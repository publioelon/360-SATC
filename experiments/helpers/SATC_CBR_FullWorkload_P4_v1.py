#!/usr/bin/env python3
"""360-SATC integrated Real-5G 60 FPS experiment.

Default execution is fully automated. It performs:
  1) setup/integrity validation;
  2) an uncapped 4096x2048 MoST-Sal+P4 benchmark for 3 videos x 3 codecs;
  3) 18 paced WebRTC sessions: 3 codecs x 3 real-5G windows x {12,35} Mbit/s;
  4) live GCC -> RT-MPC -> dynamic NVENC bitrate updates every 250 ms;
  5) strict frame identity / received access-unit PTS / queue-lag analysis;
  6) automatic REPORT/SUMMARY/VERDICT plus an uploadable QA ZIP.

The experiment never inserts an FPS limiter into the uncapped capacity test,
never drops source frames deliberately, and never turns a sub-60 result into a
pass. In the paced network test, 60 FPS is established by 60-Hz media continuity
and bounded source-to-completion backlog, supported by the separate uncapped
capacity measurement.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import pwd
import shutil
import signal
import selectors
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

VERSION = "SATC_CBR_FullWorkload_P4_v1"
WIDTH, HEIGHT, FPS = 4096, 2048, 60
FRAME_PERIOD_NS = 1_000_000_000 / FPS
MEASURE_SECONDS = 150
EXPECTED_MEASURED_FRAMES = FPS * MEASURE_SECONDS
WARMUP_SECONDS = 20
SIGNALLING_PORT = 8443
SENDER_IP = "10.30.0.1"
RECEIVER_IP = "10.30.0.2"
CONTROL_INTERVAL_MS = 100
MODEL_SHA256 = "51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa"
MODEL_ABS = Path(os.environ.get("SATC_MODEL", 'saliency/models/most_sal_144x192.onnx'))
SDK_ABS = Path(os.environ.get("SATC_SDK", '/path/to/Video_Codec_SDK_13.0.19'))
SATC_PY_ABS = Path(os.environ.get("SATC_PYTHON", sys.executable))
INPUT_ROOT_ABS = Path(os.environ.get("SATC_PREPARED", 'data/prepared'))
VIDEOS = {
    "basketball": "basketball_4096x2048_60fps.mp4",
    "rollercoaster": "rollercoaster_4096x2048_60fps.mp4",
    "ballet": "ballet_4096x2048_60fps.mp4",
}
REP_VIDEO = {1: "basketball", 2: "rollercoaster", 3: "ballet"}
CODECS = ("h264", "hevc", "av1")
CONDITIONS = ("low", "high")
TARGET_MEANS = {"low": 12.0, "high": 35.0}

# Exact transformed trace samples and code helpers are embedded below.
_PAYLOAD_B85 = 'c-mZ=V~j2g4`83MZQHhO+twZ1wr%5%Z5wxN+qSKJcmI8xG--PNwrNh2D$0O>q5=Q_NWgvuo1Vz@mU9LW06+)=0Kor8MdXaE&15Z|U7YD%JX}(hC+#*F5xXyG(1j5e{=&!LY9EbA+YtUsM3H9SLK#T73}F+iL^b<I`EXkT2O+i9EGWOFcVqI0u@MF5D7Uo}hv(*gwVbcgMoLR%+L|+8w(mshc3UBZs7a?esVWyOH;s|x`8Y^-0X%A%S^*qZ%TcbDGM>s_TRA->mGL`?!rB(?Hgqe(dn!syO8az59K!I)O^HeSBvRk}xf`srGOIQ4Y4NtS`{4TP<h&Y5&)$Kvsx4H7@=&HGCec4KN+~k(>XTIgz>ORBoqJnM#v+fE%_14US%4D`=vibJbU?&MY{m;>1V^K75rzmSLHsoGf~Srm3V!Tr$Et}Ud=#Y|zDGL7{Fgg`BDZ7eEr~P-F^(JI<8ze?F7)ONtE*!v9{QCTGO`bP4Of*`Tyjew^BcbRJu5bMVp`h>^sOx^8_&-8F$2>#iM@c{t-sK@*K~nHas{z;#GN&6*RpWHf^Q*Y<FAqdYsc(+oK2!pbbZqPk<V&$+Z_#q6?!mnN#_+I_#1Q&>g#bwiAtGF;SW&!pB`ddr>$DU004;i006{))WqJ&jNZYsN6XfJQylrH)}SH%K)#rcCCT{HMsJB&&VZhiP%f+3z$K`jSYu@nb^IzRF@^B=J1<jDx!9~rEU*gb(#+j$=QLC6Vmz5b0INeun1Y(dwSxe%$j&7%QnbdURK}-S_)p{-r6o-fJ0Z%#nNyEaNnDZstt<V?0;!H=%2@2%yPhQ~C-z?sF*|7<q+@L@GTBt>df?O<WVP}b(o%0FYMlwQ%EFyw3I#8kWcz={4VBW6I<&kf7752CrnX*wmL!v|-|OWc$#imP>CbJm75XdrO;##bq0PiJUOfl|`rm}5#vS2L5vH}v^k)XKNAK^$!^_K7KECu44JoT8RKHyK^y%qCRWR!a-nTn_Z_j-4*l5G=rE?X~Y2Uv$VIm(0z1ed*Z+Q9=gnrK-(-6dN(qgx7bY}trZT1E|7vY27z91@gF1(2Bb+Xm}st6E4JJ8hq)j{rG;QY_NANIe4r*?Xuw{A>I8aWkE<D^vhLs!NAKpYH1jJB+b<QT=sqFYf6%sM@R6RWMub-+{b8C$2+EG&G=V3WKC?w(#NXxlb)C}nS5oy+XOw{QEjG)BQAykEM(<u6Ci_TW;4{V<yoxkO0P<feKG(wg7=GsR~w_99C_CFa4ktMPi6LC+>;bGjcj@cWq0*aG#XkWpkO|E_<9Jg-EEH*cU-0}lV!Aw_-=jq><>j$uBkk*-81zB9L2G1g6!xR&`^?*5B@T#U{vhjUBOn6mRaI&(ao#$b7`55a8kvUAHdo%zu-Zm;$tj#240lLcMYAGFwDmaY*kk}P-PN=#?lpvayU3Y{>x6ViQg5_kBZmAS=k*#PX~lVmA&Cb%wOX?-Qxmm9?-l33);k2c$>E+|h5Bb>IU#gU??<!+I=9oSpydRt~Jip`m*;($mUS`~I}=rjf50J~{d-q<&Wv`8X>n=Rw-2p=*_{k-}{m`tTxAcF(wTDCja11!bO!}s$nx58q|lelDoo|{=M)Zd5Tx1$`vf`Mv)KVx{1?+;iFwpdR4lu8|Y{!73JDdwr;Y`2u>p|Kc_=37dT9hk0}$>1fX2u10h;VB_qtUZ0zE{R+$LKh&UzD3Q|GMgBV(!LR<(7zL0f^nVZQCL+p7<*$s^Z#t!B{uC8zFE{wI^LR<{~lQ6fiT!7bRDE*zi(=YZ#gtfEse3^&iEzR|7}DNzYcOE%F}ajhU~s_zz?ccKT6TkgZm7J<OY~hC?XO_Mh6R+4|TkSd$1E97_uT_?BkU?y`KcLI9{EaHWd<nK6mOo!Job0WL^heKidoR`caMX_5Z1ai5vr94)+z^A9Eoxl-?H^oHlNxf=@R@ZsHv?wS8SQE9a-ItDw3N=jp-E?(}&4E(}j0Xwt|9E0OEZdEWDZmx^XafD^n33WqXfbIj;`LG^vxQqxn!WAM8oW*Koait&o){%i;#u39(O5#0#I2t8SrQaC4F2Jfc&RdujiZgB<`k7mw5*^FRdjOyih<~^Lu=at6JZ)XakSg{Moana8+*?Yf^-*rCuNsQ#Pg?>NyvzMgZL&$OUM<LLoq2ITkk}4#+qq-pFN5QxW;?S<aXmnqt|4E)F;|WK+T1&%*vc8x<Hud2ba9g^-+@mDogICvrfwERVbSCAgJkdP+&Tv#6Vet2*V&7n1+r`53Wm?^SuEh>Hl9w7o*tA6r$M4Uv+vX|uKuq`Ap*r&3GYqI<MBR;=A6PR1;mDA54J3U@hd8l<@_CSj!Wx~>7Q$TWt70}nAR|vX$os&C=*F^k@r13d)~z`KM5vk_yYS*FhU_hzt2r+k=2>)h8G2Z_A55V{q5Ku$z`FAh4X{=V!lLO?0&0RA^1`N5N`y(>DzLV7Rj!3Fu6dfqo*p~W;prjdQ|9S;X_U!pBJyzILc9s__(-ul^L4=z=GMo%@07cGLq>DtFAa0<{}u(Ex{{3%t~8<mZrwxk_G#X6#wCDBpOv7Q;KdjlWHsyCYotrF#NUN>`I`-!p2|9!il}!44u2I!PeaTRVQkDz1cwm!{3B~C+2cA=Ib)f2j<2=LCoJAhG85#vEpn);U8V8ae{fMuJIY=P^UaD4=L@=pjw_>FoqV{0&Cywxs8?jB6bcb>;3_G9WG#neIXO$G>dRQ@t@f5dvRtdkw0*H9)W71yQ2U(>k@*)Hm*%$1{@<iZR(;a=Kf+z)7r{gwy$c2Ds~@tRt4UXQ|GPhC=YPW`pR7t?`FN;Wyit*vnl`zk!5xgGiq5&UOSqOQ;mFSTEpxy|#W9RDPYEekJhM3C(J(pQ1Lb^I#uTVafS#_{b8@%2*K|p!m*R4(5-lO<8R0gTXDtqz2XiwTHn!DW-9&U2klJbLr86vZKFaSyi7X(X($}2~%Bk3CGCW)}VBs#}kx!fGgHUYjSML-BKSFCfkjw^IR2;w=fKbKv=VSdP+k`+)10Fis%O;uA-<as9>r*x|4?{ZR&P8ye`l78_sdmXDgPMy|9lLVs_mgvB90sTm?(T0v=7oqD{2>soD~9<()Mt!%PIuvyT~us==|$){5meiEbVF?S1}0L|#|j2@Ul5$`AzuJEeVcbwJ>2LR>mIgB6#aX-h3}4pZi23ZPrZLddaWkD`6ym~{{8f#(8W9`?tNp<Onlog@6iVj9ovcLgBvr~1@_JHr2a6y5)@;qQEGvQEBI4JNRqM!{PiaYZIaqHA%|z_kyho5rn6r!C>r26V`P^kv%&zwcqq?7dRn0&MHVtG5`~2ongodfI)l0<BbrusI1qK0B^C#T1zoN&T*YP&h5DCxeC&BPC&p{bT<6CtsSx4_qy(D5&KT7|s}3R_I2Fqg_0buxc)f|H;sVseb=@V^v~{r5&liE9iGchYItRj|RmjoIREx1jtX(lbV*76tC!Q@N9!;A9d8Ir)l|1Ds*M$((w!}NnEAX0@sD@n`wPGp@KJaSofyB73%m2>E9*UG;Le4aWXIdVdhzoj$XgJs{W@fG65N_!*a+D<BeCAdr*pdVMyKpMl4m+%d8Cvp^@rwWU`=LeX5_S>X?#HR(jh`FlBP0;hbTon_Od|EnYOxHtpw6=Qx~s9jK)J+ncN|J*@ufyNbP*AU-QfYkIP;WBpz*$tmcYfi<aK}P8jpDFVn$MA@F;)}8T4ulD>9Shw`cyw8&j*_0Maq8wdwqI5F&pSot@<;LRwb|R`fAfS6AV@>W101!ieS(*FG=pHtCbLJw7iSG`D8sBQ}gNk%RF%HVnuvc|HN9kf~>7{~V9VHMI`9E8rN%@&#z*qPiL6wP+y7kFlQ({1&2X%bmHr)bq(|{^Opq0`@n7ggupRSedO!g~i}IoS9_ytc?h^iN;niouP&~e`^5ufQ%2t#;?g~5gLvXvxm^TBB61jiG*B7SL)H0RA101(aI}|;jOH+hBdpX5|Qb5FeKP9u0q?*@8&>i)2K?NR0T6!M{XpY?6hm<F!Kbvh%UYY_5S8*z{i{o&y2~-rN3$(j^4qRh^Vi<wvvOF;nPpM69-_qANN4E*eOxqnnU^LX)-#V^3l>b=;A(+&l+A)1hc}LTzA2_afYOF<!bgFgC-Z8t?z{}{p28l_KcPFR@nsja_HEzrsrHf8Yp*Awh<FP{P{O9BgL|Itz2u)YqP)f9`uvaCCCs8Ty`z(q_@}Jbl)*a{K?R5><xuq)BI=4bH7o~#o(hI&c8aP%YXAH<aSejNAO8WF_(<7{G0g<G@lY<@_SGjM%Ez=79>~|oAucV2oD3N%OYt@F0#ORPwW+f8N^}5Y_s@d@R$@^kV>U4S*OrVc$ZUeu&(D9=zo2D?_g#R5)uGNLjeHL|D%?6E@tLVMlNQi22N%sW|nSdPXFn7ji!O~COfkKYMI~ygfd@JV@|t!Vl+g2RcD^FP5jl#3Hh@FA##dhG5{tZ%@yDO^#&MKY+Rn+na?Z<0Au)jdjR40`ezA~FAIk2;306yx=Zz8yI`)8*ixIdb)Hu)vfQeK&PbVhvRZKgmYuYxJd>SwVoJpHpKv2{qs+7x{Sk3v;X$6(^i?D3BV3tvhMgwuKSLtq>0b*_NbkLkfB;5L7WRlBM<+yBKA5%&b#cv-RUC113}gI(l>tJUOeWJ27W#luQ^CFTN8l+$<uz6~IELs5xNqp6gapQkxbJe+b-rsC1$skl_)Rt<x>bbgVhFl{Ad*%VL|ZECY*lF8;^MK+BCQJ8+4I?I)^4(U2}Md3u3JpgYW8z!80(HF9UeM06RJ)cQz+QWk<yO2eq3@XSA;`E;QkV}Cv^^9;GL1ui=An9ZbF2&E^AMvImCz}!8U#wm}zP|WJohcS=DJG=qkb_A2sN4OGHuKpp`vbY}VF@UnJ4TQp$Gm2@jGgO(nyG0w&$?n&SPuGqtW-_5ITTOS=9xOOjy|YB<rJ))9mUyE%(=55JS)(#MwL9!sT0F!`oZE9tfGex2x`F247ZlJjSSmd)<w&8tmYx2us5rZhFz*$278k}z}W6Q1Ig3=4_N-#-jX;z3}D`qblXN4H~TuZsAJNw~O^lWayOI|*p{*4bFOiNco5T|IFTdQMP0)~)ntlW_a-u2jrMldcP=FYs7*-J=84TQK7aIiH5t=|^|ruOwfu7A_B#vV@+VVKHN=qFf<tTpkp+_k3LqiV0||M-trf@Vm+W_wQFPOV22rgS6wCAkW8fJU_+H@C{i~hgqN&#?_VHHX6WjA|)e93@K=Pfw<h#!_M>EMS%R|?EG3HyVeR6Vfn_GRdix`%~*JcbxyCCmg~o!)9xE@q$iXW#PdQCuff)dX*Ck(($&!!P#0D7>@4#_M6Gyd44}EmP*Ia0%&KfyP#fMaLZ@0{i=`c+S~lxJ8r2{HW*7EA-n08@Rkif3pxXgKi-Lrxw3vwFH3O~l)4l$nZD*1rzV-vRaklWUJ-kF(p+aitMH5)L@v1}349M;4l%Q(h;YM$G?hZ&lZqfaEHDLuXFqS*WH;Zij1kp5pR~6wPGEnXePzGV74n*bKFjyY1m~%M+-2OV6Lrr1vc>_kR9YoI1tsH-&D3eomOdj@rOv^;aNX+1Ev?f#?g)1o}n*E(BLGOty!VRTl0n=Xyws>x~crPpbp2Cy)-(qra%wILShMe4_e(NF<*jF{sg}gAUoo>}RcA>J}NAtAy%Jxf1E)Fa>$%L=kkPz|C8jdm;*r0MqUTi8L7+Gw(v(41Z7;yz!CS75ql0v_{y{TJZ5oOxZ)3!Cm)Regl<EjH;3zI|bs_yajQbjvlcFfH&OllY}M&D?ZJzoAVZzq8NNHW6Mp&X*o#$dqL{9Rab6yJ#WSNQ194Wo;X#~jIlnz%P;H?E<+lUrY@GSR4Q2Nw&B$dvY<X2EvBZoK<a#Gca6SA9CGn;X_{G+mp8BsON9u(O35au6gJbN|LYbh}N|<1itkpr4guR}IYO!Qio)mL@(l=@P4`a=Q1iknR91;u232{p7U_Hzp&`JTH_*w59?FnOoDL7W1{Bwnn20cd$a%CkownDn?*ZGfcOh<}PiT!&%RddXV@UIDDQ`@+=9qNpD2u%G6QcH-xRdef3I8-BE9#Tk@U>5(k88cc{0aYgAA!3e#9|k91<B8hDO~+Hidbx`uf4z33r~isN|pC)HG8Pa&^x5mhM55R*aBbdB`I^tnH8*u-MA9YtjR|27k3xBcfB$>-*$TR*!lTR@hR@gToBxNl4>IG!!MWWJ!aLE|O*sdFd*$hlH>Pa`Ey=flhA0*jlE@r$T6k6|N+FM>OGV}196Fx(tv8|SO>GkQz)VM9if5K!<r|K%gt?#<E+1ul$bedECXolB!apim1bHv@3JhihJLL>WISir0fil!Yy23;z9#Q326@t{Z&zewBq>#NG$Wjd;UfBham0;9w<#vlWGoTvCE4^Zsql<RZHr?yHPu`LiUXo^ZNBj8oo9w{HzHH$CTzpyzq_a2-O@#q}$zR%@PkC+!xvMC%$>?#30&9^yCN>^%8d0Xg%3x!oiRU+*06DZy4^>lw#0GS;8XJiI&X-Jm<Itr!p%jt29$7k?T@FSju~I=+BRFZJ$xxpUg}DV)x1*UsD!Y-i8JZzyu?Yp-1~Tbr^Un;s`OTiJqen;&34?aG?sdtw=oJFKt3tA>E%Am{IJaLHCND1Q0-cyYYUIX9@9cfj&H5-0@na}2(?v*jcH09V?8zact#Ul1?53N1JVS}UjUBD8XMM;z(3trEd}90`DV<lJxZIT{C4Hb}7GWZ?4fULjJmPUpg6K6!tvJf2-%YZD>bHeX(s0rC2N!RdE?Js%W$o3)j7&BB(FxHN8ftxuN>08k2ox;Si#3>K#)a*jk9ut5!1Z?^ZM`nS8QOTvY4z28uyxg$30+22cn_1CxlR$<Pgp|X_niXP(3eAMAF@gnz$CBba(j8N)O1Jz-FtP5ZgBaD$ic--o1thpX~EN&c3Jh_0+Hk39$-JSJ&cRb&n6sc&o@quz0*V&se+>xLPm$8<xN~*Z{Wr3sQPjOiPH}X{fD4vwC;D12>*B#J5w)f)+001RN008xW-ND(+&h-D8!#1C_#ujJe?mOB}CPHhZtwj=}6v`>>P?F8eNsHj0S^c!b!`9_({jJtRh`IBj@9VZo(n*F6qD&qGa%XTGE$US160ZtUzn@8cwIlB)%5{_R*tD8Svn<~2%+qt}$8JwGb*@KVI&CvceJj7D;t~UKOXaI2v3KIf#~>{sM-wr!$w}5&n9Dw~EKRXy$*(o)dMPDwA)t6qv=nuv=|)~SO(3$%LF#m_$015BO|xcW^~_J8?DqjS36dunfr#%yL$TO!+5<M`KHZV|>YooRed{Om`59s-r<ax}LqZN(27srCzfMYLi9(L=_7P#ZavHi)=E)`Y+~({2R8S=k&##xe+lA9nuJec4SzR95ug@zy;Oa~5zywflPwU9=5S7WJTI@Ts=!x3Y^yJoAG|-1EIU;S3KENq+Qu9ZPDR;EO0N;}41Vivn@ja1D2z#)N_+zAdz0-7<Rk~aW^Ic!_n3XB7T444Y?J=lXT3TNz0r%``8mC?>@kpPu%3%Mq(l4&5yyCUx;jmoSTP)!u2BY&6o>kDLGn9viDBGH|48Q9!P+gGgq^|nk5o$(9L~s7-&xYg_$@Pd{1KQj-xpCVFEHtxwKJ^ZLpo8`c_5StUZp&mKSB}}lQhUc#sYYov;xwC~^0imxv7ajA_0wYZBCmq8Ni7J_AhjV^nq~HtpMCPyrQGCIq_-Rc7f7oDYa!F#e@6e1Lq;^oMa?NV8=X}C#9L%|zx6N;E=B9$vnp_);E1i;l{B8)^m=EPabSX21_rau;it=}^zQ~hE8p71AdN1x=O>%oDGY$A*KXK~7S^asx+Lw9d$PODp)ou^CKI&h7U(gOqT(sLom;j#rs&tYizHbH1?$d0Z71f{Bpz7E7HOli`N(f-ESS|JWV&-!@?u0r$k+(2VDXvn+jEOtqL!N^bMVqjf3vD~OKZxetchY5zY7NAv_r!Nh6)SufJCe08zmr0Skozyy4XN@Y5eAr1}3x{$g11M&pv5J2eY1>o#UCb|H@#n<T`B3NOM%t4+7k*9-H+<h2StsXO=ZYJ1q-+O|V7=?L28nxtx-L)rAP`4KsFW{?U!-2+B7yi$(@1p-WKc_9_6MT3HIPLtHQdFxN~lPUy|!?=(W6p&q+h|7>gN4Jm-^llE&+YSL*EFt$EvN36rEG8xC$qt#zB|C^R#0f*jGZ$oy&k$`e`-6o!gt^z?<;cEMDe^+PCYQ>SEbjaK5S=?@&lpn<i8Thi=iA_JPv@a!=3tp<JFJh-4M1NNcI?e!OBgn>5JN#vY4Z0^v2q&QZ;Ts87E^JZLCOez;f%6pUT+JGwy@p%tpwj#r4T(}hXd@IEfFZ)Gzu|*`$iX<XUk6LdP%%^|JnKq9(Mp<rh5P2=&U6H75ed%#bNG2j=h~@AIQH)|TG~oRdA8Q%;Ml5F(8ch?3%82*QcteS*<<3Wh!R+ID_e`lqk}*@bN&GxUb%3d)tkQsKd;W2rsyHGlwfPE21C5lyv!6Ex~wM7nj;`552thfdK+V%%<<?jX(;}GJ!F$V9;>0rI9@8IZPCF>U>16ell)TFjLBxhRqZ}(savEb`y@Css;SfHfP))RQf?ky(uS2LQ&iwq`!Wk~b0dk`ST!HP6t^Ez57(BR4T1hn1*=_ka;q-m41xZB!=h8s7!ET|ul^}W0$9^YHnpV9upzQz8>nxL?#c_~IgmiY{vuXi?PaBQhvrK#!tvGLo?v$Xccj&eX^Wf2TN%EMHn6{v+(AQr3B<SjdcJB!L)Qd!L<T`-=zy#Q-JEj9@|6n;*xg$qX<|`o0?uqf{iH+Tp`u20L3GR$i_c6fJX2keRFpa=b>AHEkHy}f`VWg}aql9q(X(5sMZqMw#1MkNdyIwWy9@*QWf11QK~s?O!ZWEby=2*(83wX=eNO1s^|H(6apUH=|DxS|$QM;8I;LK3=Vi8V7=<^w0#b_*YeZj}DZ<8pj7Pr&SHPnMkI8zRx_VG1PvlomJ>fV97!aoSjWb8~)d<|(k9V=)$XD2Ox*GbjLVME`oF(?VG9y$Mg-s0kt*a=RHX>g4DZ?2VJ3@?T6jfYz+T7bzw-e_Eid^dp1C>RzJ1w(1&hEGgz9>Br>=&d#84{4vjfZ9rhJjYt`thkiaGr00KcQ^tOP_Znqj>ki4k!BlfJ=*0T8ftH6|3Ph>b*z^7cUzZF(AO_pR0U`8AO8F1M5F^(-Uij3wREEmmmgf5SBufUJ(OoIPDKb!IWUT<2!4f%$b@RWLw`hYQI~IO;`J+f2pkj$$D)x+l^ZK;F{2E_Ap>dhC9-_epr)}Eq^6K@q)z)bxc?AW7V?%^1LgOY4Xz$sME@?<9NBz@Az`!*&~*=HWRo+=k$Z`=eP9n2!8bR<^Q+|JA!E{Q1-9mowXBE<6XbCMSj&+Mkxp52a}C|hFn#n)7EOX57a3I5bE=2^VzPE17YAh#1+}M7Z4-gk4g>RG#jubz<@+JdRdoo&`LBE`+hu6?hIpS&O!FdUb-Vwnu%{$zh;d);#YB;vp~xR$+j}Dh&2D{snpF0EQmyQuiX_Tl6k$cv9NYj`}272=xG82-YKbm=98YP=B-Vm8HNs};C{&S<X}&y)vTGO*`I$ym$cCSR*z`9`MvWGI?c25;ZmOQ19;JxKO0~6l@1ZgZLHDBa&&pZJr-KM?!8cL3TF~Q2)m4nQ9QGbtXS)P@<*nJ#W(><`k}b#0>To6&K8N|LuIbaP=xyqa#<^NZ=tG4RZUqHD@>)IcF1fP3+ueX_5;-}ye>>{$#mz58~vd9FMsdrSbUbBq<^2pFQ#s?X543o^Ib4Yl9)Sft8e42%>(Zi6l$_L#wSoyWNyKwSlq@}nCqGR)6Z{?+BUXgOtW0L@vMJ@u^DKkoK)9%eux|10vq=m(Y3@z+S4)=FBfL&D<^)~o=FqRk!>dQ46pPcdpT>S1WZ>P%o+IwH7s<|0Lyjr)&88d#d`AS3zj$-7F?zQs-b@Avhn+ypIU5<zC*N4(v0$tGrbA1;#nHz+2M&1I#@fL+2c2cM^Xkdy#LzZZxh1@e912*b`svR`O2d$Gkc@G>`h|+y~p^dM$fu4UhfMaVAkr8J!e`39B>utq9b1Aq~P~E5M_~NQ-Q(q>lM4}{eVYzX(wk4<UMIvlD(Q{9$U7mAG)w^eXlmTYmEf!T{%Jdf{2Bm`O?qDPMU@;{SC&+uavb$95-EbOSfqyuMaVXJqL}MvR2T?Y94^@DUIHH62m^It`ykd>>Ks)xf3bfN)7A0C(WvQL5+MB`UJj;B@NTysLy%<=Ti&U+NBX%5yql~aI{Ya;y#1U1iJlz=Ye;%42Po>QnoGIGhTKEr&+t5?N5Q@JL}RI;&M(+T?7_ND21+^fg@Qdz<bdxZIjoD2x4gqoqEC-uETWY2jiWooHo!a@_W9!jDGE3nEQ2LG>?uM!NwRSn%;G{Sqc?SMZLJ`VsGzs2V=yO*?Z+EAkxZVwK;9I&a6(d_I9Wt^I|FAMMOG^AQJ3Kw5w$u@=Z4pBQeN9ci8ZafPueZ;to&fY+F5@2!^g$7#)_{Je@0IJ?Yn@#q+JUB0Pk`lOYV=*6PWE^@}`05OzXIMSjOD5K=L=*$D3N4;^o_fd%p_D&#0pC%Dkb?+wd<eg(_f@B86o`W%ZwILiZ=FE0IEFG1d)=i6id`+BjDKhGd17vKMEs5q8tIS805i%~YIf(7j=3&duNmcdb{Jd7+E>w-G3d;yTB7P@&%5@n|jHbS|N5gW*Wc8d<8;M(!e1WY}l&rq)Ju7ymE@g@=$;Cx0D&F($o<m{-XnJc$jXqveWs^;jv6ul^Bk05B5g>E+XuoZ}X{q}3pT19&F@fYoF@C`shLPVWbJtz5XxhD^b39n<FvIz=Xyf1GzC0!cG4J-t#T`gU>pd|4o+L@xR3cxPMd%z`0j9XmbCv=9Uqu%!~Mpo<ti&tzQqxR118|em0vvxveyKYOLTPFtAKAD60Tky`ZweES*z4j|cqYbbuN+gbIu24P|u^csEJ@iL_*MA5p6MmGrb};z-<VbD>SO|hPf?ZTRPVWCzhZTfY3<pku!t@g_Qt*aGjPFCu0;-XPrKax$8VfrhLga!FlY+9x1}A-F)dMOx6sB2Vo&O5ruI|t;G6ko$9^mQY<nNlZ@SGRNqM@^ZOWgxY!Cw3W`v(;QhIvq)B?CLnp&DwnT#i#yZXhXC0-G|4dl<IHrh!f3q60u5xk?vg*ePGT{qbHok%OLY;uS&ITqPIAq|*Bf#L4=}T{Q<b3EKwEn`dC5)CpMwBhzNvxUaZA50X|e7&i{y4?(-1oNJQl@lW{zbMg#GrX3NyHMr7Hy%M9y&RDHMLpd9l{o2q3X9{AYF|m=yD4bkqtbcy+nrV|9TRl_mso}C240NS5!;?+2lE`O^oM7Qywh_nzyhe!Gg}>+1C#r-ISB)tGgNj%&uHgCR<=ROWBp5#1ryj<O`HfRgYcv@pgnBl$xQPp}4ei@1ovaTI4~uTWvC1$T&mdph4k_p#gMV|3zxfBowQR=Jug*7sLS|?i@%nmvd}lrMd;c-A#8+~b;mR3NUsjTXz&F`b7fmKGQg-H&hCi=3`~yAb$aA9+X0Iz+Ot?((c+}#J;>MJeeW~hg!dJQwQooJznf{!UaA1aw=snF=PVV9Eve8ac^MGXaL_u+%qL<x2OTwM@>s((()S7JJ)#r3~yPt<cGawiCQGe))pE@M`lm6aJGy7DbbvlxT+KK;LgiVC>bc7;eoAq(1$}NV>ZLySEDQlaPY>k2bT2pz1B$I_=G=H{B6~bP>hi4xca&TsdV+dDOc@{%meP%^<bC07;*4lm*mk2-hjowhJB6oq|&M=JAxcH9e>|4Xmt`;+5S8)b#tupfsaL%*LZHt_-#7Hjn7`o&f0?&=~=(nx$U2Z!?y%E|T9k9j$YO>Xi7HXh-o0JniEDiO{(ViR|#(TqA9K8O*sV@d?o{VhrZ^2>t3ZrBw(y+^UaCNlYT6eU%!J35hcisi%OU|ZQH=%e;#ALUA&M9jP^<IbcoKl`=w!5|+e`1^`&u~B0bc?gZ^N$XCh&ifdl1K;Zbo_!X2oq1%wY#HT?~>6YQhTOZOxjHyhtLe>?5t&z<p`(7qug$O1PW09@oY2WpePE?k`%v9F%tJG3s_h{IwCyBp(--v$E)6Ps)@|Rancgnv;}YVmz@Yd3fB*#aT1LtXbhoQ3R7TkmVi~t-E#^R@2f;ZHtLqjX-`M_tliHd0uXOthE$PUXIE;FmBj!lQNS9sa=N&?H+5z6<Y54dc;}V4&!C>E5HZ<P6<-qlb!$3e?(|yt4rpGZLnW|?>0<J0cskJy;{Og4d-ai$K;NH9iW-gP@#Ox~v~<e&Vxi99JY|4g{X{UXEG$UUA!W=+0wo=!(qKLn*#6akTJ*;UHg_Y)7wyo*poV+DTtJhUIQO|9+fMH~q8s1+z@?BY2-^U{F&!yJA4RC%C?3B~^(X_H+Ci%z7*61Jlvn5sTxK;NEzIe9Bciv1BETeOzAQyJQC^J<4g68b_KHUo(mN)_TNBD!6Xj3el~Ew6_W^OB8&Eua{$`=Qu_EGupKh&@8tWG|&77hs+K;yXzvgLNlE3pD^Z}PvKA`0Jx!YM4V?s4=RD>BKWR~E<NQsGGgU(Y~%Y^GC!&DjIquw^^2oDQLZ&hMc%$nxUfpI@iM4y0DLXh!b(?JPpDvDWP%gL`s{6*&JB74gPnEOacrRbi|96re5zP!K{Kr?h81?&0ZAmiQS@edk%|LAV=skwGj8vH;Gmx45Lw3TdlUZ_KV4sEPSQCm~>g2zKWN**~dM+=tQxPq4{ZNBp}EV+j==CudPvnV{}EF{?F^{P9D`6x!&26(DZoK(N~A>8gS#eArVT?R?O*cuzQ&Xl5wQF^H?OzP;Em7*^kPyMcE1T@tmw=NrNe!9_3>ulNcJea}QAd7!9HPCTtUj(o<exL?CSlZaNTZMG0p;H|YVYt39>#5xD!OT;983x2hgeC8zh@AWy-N;UPX6t^<<7d$ci$uKnx)I|-_%m}fG6Amf6)+_DCHoN&p5A`DBKbpbQCVQs$l^1N=M=;<7UJphB8Zo&sct+^@3TOj$rbemH<?J9Mys;gDemN`B|cRZq`nXAn2<566Ocq23&`!6lh@nTgjOTZP`C0B*alH-UPXA_9TgBnm(&lo<MS<2c~>C8f{8@+4O?E}wH^4G?LE(>17xRPpE53B(T7hUZV>>6I)hKpEQvS52|#3Je*wc9(JFN&oR5Uw8o0cTcEhFcbaBctWV0Ct5<8EMV<!(4lVAzgfl=ijBuG;98&K^+_CQ5keF2$HeoB+aSJqfc;y8Fe(p-bGPUsx?aDr&I6Q=44>*#6gmPU5r8n16Cu$OgWM%36L;{Ogk`9AtCOZNB3puO__db&?>@9sl;J0nsuKBZ)B!$l>t9ZcNh53-|7cSZ?xysPf8dqBHLy{cuc{|8m2rjD6iBS<9{YW#~2j>G49Vx(PU$G63AF4;4OA`w`6O~hBpl^d7E?rny-kKp-~yaIz|+q%WnpTIn#6xU~1Kv~-_v)bn7j1cD;VaWz@g7Pg;P#I1XQFHC>6p47qBg_iK3r}N;7z#0bMv%aY+-(@lgg>8WA<l>QVj=crbwdo1pP%RR@z&Ne;*aT8BR^_O0<wn>BMT~`_m(g7ul|vknAEPvg8=BB=|REK<1I3}QOh^rEs_T-kMM@oBA?i9w3W=vfxOE+WIWB;OqfDP%y8L+FELW|GjnJk4q@%D{IDB(LXC;B9u*q&(?sZUHfK-j-x;LMC;|?mr*lkf+8Q{3!<j>6y6Au_MRZoUzJDkhAF>}>Lut)RN%!F3-xiQCb;G;I#mm8cZ?==Y`rjs!UL?FwrXoEHPeusbv2s=&a;9D8V|b7zV8d@q0d(|061W8tsEmPXb=p-_ewRbH4qeAcOxhXh>-W2Q0e<Xa;uBr8Gp{m=>hlZF%>UHFZSrXZ5CYtSoW_A&Uds~DfE%li*^gxFq{(P)Ed~*P(M|LiC<BU`@E6P)87#nM-iI$2UTAU&$im!iv&+-lsW9@y^2SaxI<AI~5sVz#QwpB=t=Fg096_FKTjSshn=IHePTD<e-Qp-LC|>bcY*TbU4#jONw|cA7xd?fxziR>0V}@(j$Wuab@hb*RP4_?=^)n(!?iTWT-xJC8v1#$q-=8;AxQvTtManQ^C>fPSF>4hrka8e~Re4vanohJqG4yi1=g>I2IW^NOIGTB>K8rPDe;QkID_3G&Zr8aZ>|+EWy!z5nz;kOc|2)M9pg!m(@WVrTf*>_?iXbV1C2=MXFo>Rpcd=!GLdhM3vbMoF%*7j=Na}qKLjOqPliShqMsQ~JjbvEY6EJWQHod0N=@&keFNNJRshS*rvg(hmY4Hn(0zW||+Oe%!;cv5)H?P^Sc&`e50?CMR4NeMI?bo~R87pSIdMnpzzrI;c4Ade?6&YdHxkBIQhs|&!8no%!Ep0Iz(H`MIW><zQxH>X^w8%Jdz4I{$$i)7>AP#(Raw@UnTHp0oU7QI3SIh%RNNX-YvuVwY7XMd=5oCEL^*e(qM{oqF_-#=-mmLxFx&$(Hw1m9P5omiAc(Yo=|G|wPaMq(O$f~K1J^6uTg{ipx5J!b!Mc)A;L2QaD$#3H@7&cEFT)-1;v*f8`OwnTSyDY?a)b6?IERw6={0W5~?QWm{U__j18AQ~R*N!cPO0Jt@$kJdc&{6ioFtdB^pC^NDB95&}YN>&WD!ebIOB_j;xYw2uI;y4It@geT#Ke<F_HPaYq04MPH_@0o`&)D_(EHf(tou{+(#z;^zecAm4E8$VH|zNo_tWj!4Ase!H9FrkE(it7pwGQ<MqKfeDo3Y#_9yjYgb|<L!Bv&qXK<JuMsJl^pR%s5%Boe1uiYR>14*epLLLo^B;7nS3I&31=v!RP!~sg9Elsf3tZ7vPmlZve_Do`&yN#lmUGXBw&({>545cEoo8!pVDIB3lrw^U_M(*?l<_Bul@m^>*!~5%RUNrt+rJeSrJv9x#@h)%fXC|(e-gJ{o=~ewzq@RIfJP=1oTKLl0AhLnZ@EBTxAq#KajDH~mCDi(Giv_a+`-AnPwHDrdDQ6xoC8-lV;UAmp8pnQ-(Y#wb+qUKL#ncCn)T#*+^hre;qjp*Pey3tYHS}b%WE6eKG7}>`R&2^Gx>)U0v8X3W;_&jfu&E5>A{3Q#)tAafQFB7p*b>5VyH$zHg(dDL1V5Pe^IiE0Xdi?Ax*-p29jPn3D7Hs}?@PP~y#O=Uf`*Jvi6eICd97Pcl>S{L{}lHn#DX7NR#nbCuJ@C-ufdTaGG5ch3NpNj;R<KQQ4$W0hft>=Hj|-_yPc)0)r~MO3*|6>1@U8ogW^_t^8)WR-QX%8Z;x-{8?U}Cryd-8i|Oas05(_s87AZ6S4fTY9T$VqC25u5N3}=#2$?!gv!F@YXobL%#Sip9xemnd!V-q6Q#$s2DhlAJ#Taw^#*ZXGxt;Pnz*N+3Avj{)uM$l}4&gVSvQ4&^;L{apM9a~Fl_qH|^gVj-c?9M%1Ymn8CdLT-8C}wl08G#_p3-@xPX{+a`T7*XslZI-OC>uz=FhncOA{u4#_9-rDVi7Rz3ldS)fOGcrylpB1=X3YAMo%uMt2Av+)5o~9<QK>;vWkY<rnlDdzp!bg?0b!G#fTkVDuADIq@v%y%hd}x<*0!Txc&4lJMn`+gdtS0dwutfK#w4+|ah79+{0@VRQ)=X#C(&t$*jXI@?R(O<BoH8#M&?I^RByuA!zl(Ytrez$Sv(HTUf+)L^O`z2mdk3KK)NmtgDOQqV1cSkw<GKtMhsvVLmIG<O*Va0<(PR_V+593rqitDz}RyUw(XSP}Il$N3gCFQbo{f?)=nO)eqWX*ze+wJK}iWn30IG;qvD%be)>Ur77|+#vb3_F0yzElNJS#A43~iC5i3!Jg?9-;|1!tyKDJHq08t;Ij}33{*q?NhmaCS+=o;DO55+K@m?*%PC~1zXLPvd>cK)*#1kT7|!LlF6`nXny+XLwS!^lG~FDI2XO}OgsqaM`xiM%E1k^JtCps{Bl_`otOR#h`VTuVd>$N&AJ(%IxaWpfy+er?zL!qL+9>$Law7CsuNP#%re!^%%J5wbrd$`Ce`&~zg4%6|_+|oNyufo7yl^n>q}jHrfFJs-e_gYx5LTo_<Ym(>i8c*UNOIQpd&fUHQ61iN*>Axq;-_m$_01F&4Z{c&5kz%m67pmf)_J;{9rmTOf3plSC5CXiA+qIc|0Ku#TOSUerSQg)4ks}f+-}~6v;%?4+UN0tFH+sOk$IYyL@FCCI{W&<=GCe)o^R3oTi{A~4I&Udqp+5r>&tyvYxXCUo_%k07SoJrZ&$Qj;xjFOSb%S_{64kS==E2MohYg8vaVTz(LoPh8^b@TX<=s}x{c)f+-D>B#RU|5H&1Sue5BS~o*?^Bk<527pWqHd!x;16`fGdBoq*v&FNJj|Fm9=flkan`l2f*!cJU3`Q2?Y5X&*N~XZAM!_xt8hCs`$ULOudX>|;Fp(EYYAC%4kM;k~6iD}bm`(dBqvmtG|cs?MGHogm-hg$az7Exqt^7ZF`*eWEe`P8ey~t^Q30HTW$)Waa1$!0g72@ydRV(7ut|6cyXPD}Ehsu+zt}EE!{`?f>!h#&<co&xJ3PxHmt=mPZZL7H45_$Je*mMLVH(F07aF1q?(QQ4yj1G+r>cH(HNr>=awk1Cmtfg)41$K)&97`rt0W-d6X@zcjjjX>11Ylqu|Be*JW;zmREa8?2&%GhCst;1*8$)xt%-bUkoBS@9Pb=bFJlyP5SXP5uXrzReR#epf<oCp9;`4cXa^^tWTB|KOGBQ3m}D;@`}O%;<R*O%C4uN+;#MRn|Nhb#H(FoUM+3lI8(+LOq#o7}_uzY;LD|y=)QS_I_nk!{eT(OK}!qU`*FN`xh<>JgD7pwXn*xj+XGR!Ei~phXzz^5U!y?!3!`EH4fjVzh}8kzkQ-gGv}KBi|~5T5Ablo$NrC|nU0-x)E`uLOE#C6QOT{2Q0uwle<0}yzZx%FT`@9Dy*1N38ctN=T|Kua>NI}2%Ygo9^VvM=?3Hfq&C?fYvFmZk(F4Pb!qCpP4~HD5KRM~Msj;Sv(<cM0u7?;_uuKl-Dtq$VR~4NKF-U9eO~%oEGk>p?x7BYJc_H^4bWZp5?eOjxJ`vdH*I#xBfCDCtcno;`Aw}_wu*|BE!bZGDs;&lhmTTUL%y8n_{8x*k-CSUQAU5t7A1=a7E?;n(h8>k%hsFati3I+3_EdDv{y45&3%=yXNz>qSHoxx{DK-0gon>4LJfuMhz&Mu+o{PVkZ!>DFKI&?`@^<?!JsvgmJhIi_UZApe4!}8ylshNute?#M|IdG71BWF~|CQ{5007*7)W*`y%)r*()Xe7p{`~29J8z07{qE))&s5L^KtU#^&`v5lUgAn*yVjU=nR2zbQ)5^`!irT30R}{nuOHlZ4Q4?wc`3)oJCu<EGtY0%{|k&2u~svo`r4{aM4|oz+dMS1L(43+?x?0_pF)Gwu9MTS^lVc7%u)YzEh|6s>0;`gC<<%f{Ku$LoVD9~3-Xd@_FX3ZMpg~yp=v}pq|m*Anut0MwR0eSVqhdam@jw~{j__p(+0Jq(0q`0T4^xj$wxwlM*rx9GmicZj)61{F<j&v;c79W1w~LYBJ#0y_EV#h{Cr-n-__4Yjr7w#P=cR(r~9i6y39wLnRxw>OLL<);X8|7<VzAiUKEDGB6ErPuQ$q^=mY`*0bbo%P=hIWgS+MNd1`0`jknR&EzgG|Dbf|%=_opd0|w%?KZ%k-t^;j@slWL-{6=rDPv?soT3#d(brptA*wKsKwHY}hl`(2K1OM<O{|NkgIo5l;z-CVfbxJP3zblZIBWUS`2-DF-Hqm`R_A|M3AyEl!GSQ-ig5o}O9zkvXGfPF|$=5qb2i1tMbpsLXC-VS9{9D{a8fI?Y(Mk&ZJOOS$3|e|!4>j$vYoq$pvo95EDH9PqE2EF<Q-(zxe%O+?pN^9U^I{-#g70PH=MwVUqbaMYo&lxQqJjdbh+Di6byQuLIaIEX<|c#UV~DDDSe+SF-8xP{un}wM5P7Hiv?BRAVVgys<&c;+Z%`#h-|XJ#c#)3RuS<^62#4t*e=T|=j|<5&pLKA|L5VkWZLtv&5%<&@K7U@QSWK!571xI>cd(i-7?!msh4gF>%%RXy>paoJO&qHNqSZo1!I|$;1ROIaUmBtl?-B7C_YcS@4%i)9p_C~LwzO}JSf*@$oG1nsdl%weB?ZdF93;fWXE~2#xQ<0<a&)``a(5dM(iaf~^&<o>Fdynth}Rp2t0_hT;Gp7HnU0oMr&Km|q_uxH4<uPIP1~qWOeR+CAC9pb$<07E?~YI***=j^=@a$Mpb|Pz*cSM8y86kdsd0cB*1UQPck?5lz_nnE2pIf)e;d}TxcNLa_oPshkM^EeB~rAp$)8>-mD)0FnUc*ZUfJva0B}WG-0Pb<s;(?(RQv98Y7SXzK_&altTL|FmSMTS#z72{NxR4flEjN=oCTvbNgqfzZ5<sp3VLi-n=oG?ILF7w>fWg353DG#I*xf-gDOx~>q9vP%^Azgh8Lsgq0GkBfyoW(84AaaWa90L))tBk*~?Dh*@9fuvNRMG|9O^7t?4v@E3S%SzM%X6?LlNZ#kkGvoIZ;Mg&IVe6LVK3Zmk*j8zax!tW?6cWTbtbXQzlgpc|hFq9th<TO*!(^9Z(pPk9?ER(`quyGTe*T0B+4ZECwj&aZ&{dQUA&4F1nD%}Q&2jWv%>-|n&oBIxD}Vw{qxKSMi4WVw4s#xFcQZ067Lf}nNeWwv8d>;n)BeUkE3;NC@vjg_L@8jv#_*tL#BMKOr$XFo*gHcq{uHBotR`a#lYNk<P5*oML5b2+fIo6d)^Rcrn+v1_jJv8*J>%JrYIh2!poG-z|BcBtK2V!w$Me%$7<+*x#qD+zJ`{>Riq*-mn+!&OgEiUBghTsYgW*!F1Sb{EE2F9@y<2CVKTW=Z=K7!2*3z?Je14;zw}GB)x%kA5lLq<>-5TU#b<xSA{Vl|*{`;4^geZ`Bc^e`CjI8L=w%z#yZoj@buE@#z1!M;BoQ%8vK72N&%(J@WL!o4o0y|2(e4P~M^IMwWsp)GhlOy$gSl7ttK!d3;{D&k07^p%V<%f8CDm)|}TX;CR&Mu<5H)nMroGFwvJc>JQWe)kJ9K-KpkK;SduQPi)dHYe`YSB8bIKqw7T%@*Uj$AtG>>o`4#!)E{C{;pK=+KH-<wY$7P5PwiKm_0q+&@qP5|Ujv$%y^M%{X3(jVQKGvYtrN6pzi3m(*}!B!r(y4mK-a~k3XA`PxRlYo#Vb=huD8&=YR@ps5X4cpo7dO9Ru%0Oj=uD9@(Xqq7vbt&5B_)v^&(wMF#hrtDt^f%4C|s%hCk%rq`>O#XsnP1y(XU-ub{NL<N8+~de!J{y$ARwe{D;^GYs=oVz8A&`|ExDQKVv<jSmkmf9iN~`Yq9ta5j%1caC)Hgi-$3iTidR7NuS4O(#H$ekaK4fFMUNnon7ONthE<uhayN9Xd5-=66W<R#T}{{4bN5kDhr;uVUapUBQ4^Uh1Hpq}K^bqZD(qsl9{Duef7iBcEw`P(W=dp@U3fH6I!qsHt?x&`lzPu$^y%i-(?B9L3=RS_a*M#rYYhGxkY~ErWL@6(4iJ>%FjJfv22zh*Nh>TRIN&P)S#Y%PQqil-b5r8>&3Fd^sCNtuVu$@7iYfR>l}>l+?bAS?w77YJD$nP(xISZ;*hOIFUQuAvmU{Lu=TrSel$zOAS$z?*hRZPJohilmp0b1E6MOEFF2VD!#9q7xL0zr#F;OAFaC}FvF*ZH1UnjklHvc?~cuQJG*^M`CN0*zX%89l%xkwQz3+@Vh5OhfV+?WtZi3-UP?XvJ!;{2O;KyVyY<f$!ezuK!4VG!GKV8&TGNh{c2cm}ii$gS9SJ1q6`kjfnL9aN&0;Z6&W-RG%^w*9fB?n!wV9=tMmY<`Dxl8hJ<xfes?y<p!vwz8-dGcnOlsX7lyRtSbQ>L|Z{KlHBNBkedabQ0^hnQGX<9etl$pEmD9~@vf5`Lf#5p}Wew-bU*Lsg(Nfso!C1|nyVPN@|%k~il{_T#DKJdPoF3`t?px^7s0)VbkaR9*mcTK2D!vQ%ku`8htMLzy9ZW8J4`7mu-jTr+W)p9gt+}Dk2o{B2u?n5z>)7-g?gnze$yX~YXC=_yDDyb{UU%Tce(AV#Urf)dZU@-VT<b6AS28%%tS+)fQ!H7^nOO+<$OddCd+}ibvI|$O{vrz5mrnTanRNf4_YVx_Fw_?9}%we5%y<N{7?S;R=lQ@SRR5?wlZY6cdos9=cL<YH2M`qI1Oy@*t9C8gUP@^U!UksyTOTDO&pdb;1%$BU%=eq!v!MGd)W$wq0_h~?&#{Jx5r{Pa;r%+`Et~zNt4WErap@=TBPmV)#9|MjX6Ru>2^*%D1%8LgkHqk7rNy`e*B&8Ph>8oPD<F~c8!n#xM^bhj56%s1j<eBpI+t26hw0A(x(YO_WpZB}aGB=ZI*7LA^NZIIb4;ku|{7Z;hVZmo`(RY1pK%Z@p)}RtN))=gq)y*UlKBNXyyZ1n(i?=&Gboy~~z3HE+VMCV`GQTXi$_U64QdFf5D*PcXMKNt-d}$Yg!L9erYE>fkqOPf%U~Fkl8rZ0tjCZD`%!U{RMJ^@O(`{{!<js5$>USIZ6ipIPlGcdvi(%4{OXTYONzh;B3kAJz&}RADGd)hGIJ;(>UEQ5Elp0FZh`gBJsUl~k?L3nj*nfnC+u+13*C@&*{}7}#r1%6cZEu>=-yyb_KU7l?Pm<1?`VWjs4LA)Q<ZQSfq~VXYDnc3bzfqZXE6dv+dQ_vSTAYsP_4)Cg@+vJKie}U+NPitGr}#e%agllw6d?JsXo-xz#mz@?9uC>EUWXHxDkVVa@Bv9T<f&!gY_W_aOoCB#xTibI|A(u03KE8iwMECaZGL0hwr$(CZQHhO+qP|c=KlMf`*L>mdRmq0mn4<c>Qv_g*lS~^l}Q`<KJ;`Bt-4u?hnEa!K)Ez?<pq97XZ*Ps6t)2|tgE_;DQ6R$m_fWasW(=MKsa1^)ETlL>TAd5)e<%tQU~et%ar9Lz$J&|7LOh;Yt{s9pb`mYKLBy~OF>i#v|!2@@&XSnE>QJl^2E2hH`@TWPr6AXx3@Fy+N<)6R4dXcXAL;up*~6I4zOdvA@~_x&TQ5=oS$l&7v`xicEH{YWTaQ9^|j^h694NB9g=yBP4HZ|!UMyTXjSL+k#YXD*N9L5SXBL+9S1a{U5qfpj}2)n;h&pDdw`x;McQsOssp@(^PO)@6Mx*b2Ekt?)%uMNYA%#hcg9-gRoF>E<&$u`&WVEC?(kJljQdCsPVDwE?`Cpn`_I+IoPt@>TBU9n@6`)M0OOq%I>~lH5d$U~SuyP4Nb77@0jbFg@Z*um?q<&1r)fA2p9i`v;<AZ-v;D8ZJY?;cY?Iz{g%lf;_Q!5P?j@}ERjq0=OT!)nBdbcW{5=Qy*v%tNEwv#^iZK^G?nT~}d;U=s*FR)YA)Cc2FZbA#>wfM?z(D@8vmkyl_U5Qqg_EmSfOJSQ3i|cYZ*Qit8%%C7bV?wNVgQgZN!)I0=n%$=6*Nw9qS6yEw{Dfk^8w{Bg;4F#ve@am%11~+Iwnb=h=p9etsrA+a*SvlkjVr2AFAdZ<hq>2Ztdqm4zbn?V+owwaI7ky{0mrvaAAzNRtJ%-9``p*4id9fm_$P;;e+hV+;o_-6!~9hWJX0ZGmwVaNWi9-<unj3jbagh@sLp2m)FwA$IMJ)+balJyWE(9n^h0Ub<W<Q-ONj5ognb4Vp=$g6rGg>b^^0x6MazdQAJh?lM-&<%ioR}^^OyW)94C5qm=w+Tnr7{rs~b~M<<jt!foz?%rbQa@<2n+#AYkRmx>r+45^I10|^|KV<a-%Ud0ccQ_Pe*@vd<Kt%=*|+Xu>WHgTfjLxyxvSuy4H0vaY?8C`N&xY|9Gt~n%3l2#v$_z03}70o@A6%UNaZc9ra$(qk+?5G;4(it%AdBKhp?G>G@v51O^Ry#1JSqi?LR`~TAV6!02mg@#26s{C6^Cc5m>tYorLDRv1`|&Qjl;wswG49r=Zo;EReS&V2=|K_*cyg~;7?74(aLUNI)Z}G2Bl5D`IS(g^PnBi0a^7L_`7;zj@1s}=9>|me9O<|`@Cf2yesYak!O;UV-09$bdf9iwCV(|XGk^H+Z-AxHM)j<^Wlx9GwT7+4|1Rh<L|O=?9M6rtX3&pZ<1eSX&5s576E+A{-0$q;7)@{{&i@q|Mph?77bHjY8k+R!Z1Id0iNm^K$If2+0#pIZVZmSaCUq6m;@Zp}3RZd$wWtJ09=6Csx|ct)q%mYS0IGLST6XJAmu6~bpvdRy`+XTN_oHyut=2?^j{uJR6UfHQyG?D@A&Oa|uOMD&B_dR+u$EcHRKLI|RqSFK79|Kj%xCHW)3QNgwdyBU@+xgiBu%;U&tU!4Kv7OgYqO2hJs|=y)(oWCA-NRcy3%^8%BoOR>+7BJu~c*+`?Uc?4T<ws>v$fGL#R!ueIPB57oRbZdMuv)X>h)@<l(tEW~#97@U=uDEW5T!di^}Lz|(ONVqxF!6nLr#<L5-2a#R}}05gKRRIZqcf#c7DB6e~32fq9`b2}3NLu7@Yj+6#f539Vwg(9C2Ybny{T~Q^<O61OZ>5`rK!1c&ChTQ|>FJRH2)P0Fk;0ae|391E`>8P`BF%W5tuisVN3GUq1l?LFI!#*LZ^y+w0TDX?_)|q?#Hu~}JVX-xu5}?#U72MqVlg-HLnigI+)<4lj{v65LGabTmRYeq(t0pN@)72yr+_nhnGhNc_9a!GEiSElHuquNs$1AyKdRrET1GD<hnXjUQ(GW86y(#{OfEe*B7^F0zu_S@p17R8$wjN?xc`ggb9J1SLSK7096H!5p(HY{#75MG1ft7Ll1>|Ulz#HD)&?IOi%)T>8Ao1J*u+3JVe#Xyd$67_(Q&+R0MZ(-SUZuW8`{WHD;@SCCD=7ZrZshtP`{9lGB-0CVl6x_)isXCHIjc%b_SIU{Mc^%Q8{BZaJ^8}Cg6ufDb?08abyK6C;xNDZTw@E>iZHdNsIim3rb%_A@4~C3w=+z$>W!omFUh!@vt(6k#o!A9`5EHr=;DDHx;$>+W~0bJsL3olue46Xt_QIcG0C+0Y5yPN&F~jzMiwJu%V}{71`O}L1Akn20z9a+qr;pT(585_(u;gtwaL@(bQ<uRi5%*Y_hT)PQSkYrs;DsAdg4|#*rOGOqef`R>ksdsd)}K+g`&*duJC&hNQSp1$t{qInCll;nvMDwgggC(Xr7ESlb`)u@7k$k+Y>`aeP6BrkF@3-uG<aD$vCU@<KpI+IIfd?LgSo+`<B-m&f(DXSHO=_xHcylr$xSyRwD};xIQCn^eu#V6Vw-_g8XZx6eChwc9JRdCufS*MPANjM(5q}>rR=Btk<xCt$9Xt{Cd~hm$3$3cdmtXX?#uJVx5|A++X(3I^56jw~(i5UMWkLvID0_+h-N~TD7^uZ-P}9WzA`drF}LnuS;JaM~1dbZ<*7JUYv%>JZs1|y;aNDqW=eR85w*@+&@m0PSkSdiboM+#zsiM%>z|be0d)$4)JC*rC2%A!*XD&GEuOIK0<D!aMtw*N`rX_DZBW+3<2M_nwN2BG#L;^d3Y?KF4;^Igj`;qPq<9zxZn!^ALS9yL^p88W*+*0ma#?>L$;Dxpys{Rj8HoZThgwZT)iOYVs>8<qIU_>u%8T<Y%-cQoYSAW&y+;D-F%4r@ORb#LRK`><E)7O?R+U$mp!q>nkhlrNZn<tGX;j{;yM~hlz}jt-RDq>IuG&CvGS}FF>$}X5V4iI%@>&;QU3vKM)9Iu->-~l<uQQJIP%)V7&D<<?!qcD(h3vWaPx*mP3baMuDuvnzTyN;Y5z<*bAA+3frwU>3s&&mG(-^orCQu(NtIQ%e*x&!Z$6l;wxX1%)imdh@sQ?(Z_{oBZVfnFL7?`VZ40a`lOJI@>aTF63{_~w5oMqgtuD!&I=(4j0#$%7_W@ucIZprn++z#d*-UMWiV5Xgxv37+Mmge7K!h5dIwu>4ZF0vvp;CS+2z5qQXpASo>dPb+H>bymB3j03Pv^UH{iFHEN-MUH<&n58vO&56LO`&5fds9URQ+oTa*c!|lkH+u^4CtKu-yb*g}785=hw@EY)#4p?ENC;aqHGj!QaZj3-%-{5Y(nA*6U|ee`01vp4YwO*6tq{GHaE)2s^i&_wI8S>qtY$u5)|fHd{VN8{K?{jQ>S_9P|Azc{EjWdI1;!02&ei0P#Q8#=u_B-qFs|#K_sg&X(5RqeRul>PQsfw^om#9sgemHk)kDd^?6`n$+Y#LeTy$y3w#W*4CDIm!VsV_}dj1)kMVgrcFK>g2&lS#)cW52Q${rW#MwFK8@96e|*ZG1b?EMD1WSM=Racplz=1-KJKJ}7-j5vQa%owYjy}UEsKD^?bLolfp{qq^Zb6K;=|%}V62N)P>u|!e6wjPXYIQXc$X7^C+#wA85^KHV^0Ss9X(xXzY}}0(tPPgu5lnX=Fl(gwpw{h)*x0)O!2mIR%1WlN(PtsQ*)bmg?}~6lY$BwXAhW~RzmRz;PRit2=+mW4GA>XDwHFvCg{+bfM%5#T@@&899S`dEeoK-51o!F1?;B6C9E)dls6d2YEYlVDySyF<^m}?$#S2UbHovP;<l}`k*<vlCegeQ7>`(zMSq3MWl_-1qNT<|=ouY1+wh|OtufhOfDfb>Y3)q1wzOS!JwA*%qD=trOKCEBbX#>Sd-g)~_{z7BcZy;)Tk!T_7idV$zn(wW1q970_;8Q*mb|N0@0y97Kd(hQGF3f|if1?6+%BKS5^f5mlOJ$@{gavzm&C|1LUor^^^QvW-$1$J6q7u8)G6ll27jy~icH402HOqH$|vbqP<r451Wqy%PP}?vk>8G(kn1e3m#7h}8wY0aFWYJ#-}UWY(S6>&?O*p8H^3N$@JX3f+K}n?Ezk>X9ryg*S7UW?dF(S~U0?8iz8cRSnMpCuPkg%*^SgUXf9E9}n^HqAWI(mMJ5qCovaBR7>qpo{FEPPEqMPN}tFc1w>zvYggQRM@htwLlVA^s;O}u4oQQ$S^s4*E~|DHjz^@PR50J=~8i5@<mbW#n3rZzr|N#FmBTO(C<k{=@8>`7i6$z%aAm2<MoJ=I%pr&fJ57@}E+{}p;W7*8n3+WyngU<S*YhY!EoFQ0Ly+O!xcunv*T0&*%<89h{hVkgAuVXQBwQMZ>8noET4@0@({UDIH8{3q@*NXD(LDp{AFvuG7WSy!4iLX2gk0SPA&>jMmq5O;Ic_xPm);s;zmX3uN8)?)IWb+*y<G4*XOu^yrta#Q6ljUD8nHDIdzxcu{R)AsJ^iGLH^R`9{6>_*YKL-RRPR_H7<E^kM~pl?uLlz~XdMDwUj?7Q{8#t<UbzPtBKOO0wklSym<u8kYqbD)WpP<vWn=ams|>axpwaHT|axgz(Y(G%$lH*}0s)Kt$=SH!dEcJW`Y@{Jv=hgXKt?FIfC{%slV3VzBpMeo$bdC&FDw&$LX$kkdA*!RDlM@n?~8RroI08E(wZ~oi5n%ElY8GG0o*jO0R8rj>|X#F`AjwAi?_>JJy=|UkHF26W>NsLn!nHuR7=IS-x?XIAO(EmdPK?6g!WBdEM$^`@kIqLONwGj!+&At8a$i3B{ogEK}yjU;ew1v00#+2CP%GjV`L?`Ylm8}si^|ziTPj~yT(-t{-8z=Z<&0;pCCDUUuZsn-O&dy2ffe!ZjD{PUPu?+|9m9{<OMmoVP6&r{Xt(gu-sfRpsBXK5w7GTXa0>S(BEA|>Vmj2c2EJaXD*3OJG{)@_54nLUg={Jos@tlXr0Cl{D@%TU-%RLPRQfJ^X_M#V!f$>Tg*OR$yy4Lr-jLe*dUL=<4d=7N+1gS|=fuIKdW-?N=TY03m%zabET?@7wc-4y15|%`iOTP&|dqDXm{^Hig0BmvIdKvh>Kc+ASYmz|<7@@`DB(|lXI^IGIfR9A~Tqe2R+CpW(o%3EjOFpRI_B?O%UBdtB`EB33YE^cePNw3e%BP0fh84<CS(k*bsK{QfoG^R5=a+RNp<UF*?IA^~;l|FnwlW4+&h^X{D*e}<h|Opih(vCTIV+%OI2ZoCq)cdy=uQVx?dE|gObb=9L~sv;c~LtaN@oU|WqHvw?qIBa8vOnb@}`gCvJMLBa>n^iBMVbj964y1q*WEQvX|OA3~8g+6*&?Gb$Ud_xS`q>i?Hi7aBsZV(XxtVg{Mj7wyNReeyGJmwn<9VMy4w9=t!OZn>DDeQ9qx=^z2dBge(n9)bM`mcwqVm&^=vOUxLzMw>2=wa?bFZre~%<@RSW!eZp-gY!))3!f*(Qn4DGiS*q3T6-Yta1V5&57DdKM#lo^nw0@L1pwIzm9u8&})SJr?r4b6k&xRB1IQapVGRZ(%^_+s8St(9U8>)U=^1!wsPfmP;Zbk3pf#GZ*L{~+%Rir0@#|_r3`}Kk#pU`C^$7s4`Qzi3s7CIGdX^|Gd;DSD##7sOH`@@Ws3pLxxyxnPXL$ywS`x%E0YLl56c+0Hy42_n$94w%R87M2%B<gIO-+xmSK>XL1J?I(e)-pXBEI!$~DOm;5BeX>QcEO+fqf?JW09nqLi!+c!AT{)6@_hv2v~;rQ9Hh_gRoMCk`aO}gi94^e5%uHIeEU{zl#}=R1y#i}gRFOE2xem9?G8lzN_-0w&PbDm>#4IdqHcW}hT93dFao>CIdfIMgo63>IkBDUGSkz$Jkc)dQ{?C?uy*=xwDMK7v*)&b%X{9g>&U=qd%+Hzn6c9ij?3n!HU~}+QHa}rghPn_BMS>_e{M64n{e}@X|?><vtZZi#qy|#^GWEYu(M=n_^SXe03+{7b9yTZCxfZRPl6`ToE&Y1L|I6%%La_UWE!BHJOFXqY|YDIq+Ge5UiU{d3Lb3Q;$gj|m4jM&0<2vhTlNt~Nf3fi5N7N_7a>AF{TfR;8;KyQ7X;GzvNkx^K6t=QLvXl=Ezi=kY3=I!%QX#<58S@@p!3<Ka>C+Aex2FwJ7<%K;|B&QQ+ey;&tlOg$-`JnbG7*oXuAaXeo^bkwX9@sD1X+1j>&`PXTnw_SKzDym1&%p|HPqtJgC%t`wl|23jpt9gm>BCzEEMGK>@v0wl?s$Mxv`Cj;u!2j9HQNfgoizDWfJrM6DH)J-!yfxtWXrv8b*->lzgdD%Jc6GLIrD&LB~<c|EjPe}tw8&C6yWVy3*ea%!wTS5qax2(S;<1hDUVAV<Iwq%k1co3HB4vKz52sjCaFIOq4ifn(K6u7%S)o~O@fl3mC7Z)pL1#^A|8n4Y#bg=Zy?bEr*Fw8r<Y_A4U2>7#90d;mtfK(GUhW-e#L=?0_J#A4q;M7>DHO!T;3{Rj{TY)YP26z`v>N8zLXI;P9$CqH|O=H(J-Uiy&x;?*l}z^t3kPr;vtpa1iQ)lUHm#9UP*A;el#D%SBp)wX?B=fpyW(q=lRa`__6T5W`w{pE!I_Us(+l=5lrAO8-AmiY^qbU$IQGI}<<I_7M?xuQOlmI{4WK#s;fM_hevDu%T6IRfjYYr9!zLm^-`R~xI)Of$QVu1Q4ds8w!$_K{_0XN;+Ra~uF;#5LqoUD6ZqtT~IR<F5Sc(?ul7ZfR`8FLrk7zWytKzoudK*u+pfG#OURv#Z;3x6gU=n>2dX+P6-XE-DWOFxoTF&;iJugU5ZhnGqm@Usy&Gw}ntf`}))+9hSQgRCyNIAae^&UEOcdL0QkiTfpx+&iT#kP`gKq5~(qgYj3o;>Km{fm<RL_-N7V$#z1{sIQYApS4R-x2L|F(Y(xcN770UtBoUXgy~=}&Td9V_c_imVAbd{QY>@iv8RxlnLybPb{6draIf^uVIlP7`GX^Xev2AH&Icz2}%m_$DmUeiVMA{a7>?gtW#}amdJh4el2>C{wI<(w!pB;^Q%eJP5S9jGPTv!c{eg^QhL;Yho)^F=AKpdQkPuTu<Huuk9mn<PlTXH?RoO9(YHD{9!e=q?Ng-Dt_jsUC&Stfh;5Xd}rHHw2Z(I|1-Fj%fUf`+jvI6w%@nqc}Fwm^6cHUBeUHPm{D2gv+c1~@Rrg$VUmThh}5$d{=UW66D3qW?$sR;S=7fk!5=Ky&z%P>Df4nh13a0?A>i&LbA;q>q_p`dkVS)#z@3=gw%zIodQ)D)6!@hPeZ6^MbmL+bOC?nnuGl!i4DaRS&B97m=zw3L^tLpgB?TRjbzdYCh>8L1DtE$xovuGvwv)8`L(pN+re&<5z*^8h8HcOrzeTW$Vep#x$$V{g~oxS8|DThh;ZMeA|~6By$u)2EWnAlS*=0Aqz4F-KX=+e6=_KZ*}%|o;Q1)*;%wZcEePcXkiHjIZH8(<T$(@ipvmn8xdW#H#QEUWZroK?Ojhi>;;ZBBqNkB>XiOyYS3opd0fue*RMJEwfAN0C_pCo&jAMfH;&+3?Z^nD1w#Cl6;VVm-GSS03Oo}27lrf_cCZL7dk>gHg^}Di%Nhv1+9Eo+l;uV!?W9EoNV{XKy?7K*B;80~BslcEGJ1~wgEX3}zkBj98|0wkBjA{UvxMXaVw_5O;!<2%u|sauhnb-+(%w7PE^{Q4k#7o&+;324suDkO=Jp!EAumuUv7QgdE7r#orP^n8Y4#J{WCZoz*_h*$t>^sn3i#;qq!(K;*mS1cIzI6?GhxxI9SQTjuYs0TLR$a5&pURAzWQ8<5YXb~B0YM%>JyBJ9{I6YD11h;pr>EMMxb-vEj_b4EnH&LQ6fh1kJel4et|f>%@V;}0-gBLlTbZLFS5WXS<5G;C({|tEf)xMHt;@$`q*;{_uQN_Cr;ZeU(CDjKybS?D!Be@JF%gX%AqHWTIKa9-e+)6=0&$a5$ba^lk#A;Dk83}#tl#_bph(<FeOQw&$|CeZq`58Ex%<FRDa%#nSMrS-s~r^ZrjAVS||lfV~F*+4x8VN_r@#&4=1~=gR@{up%K-V!gxw<kKLSR2i~pDP1bSxzkI1GJ)Z~k8@T=;O;3ZD<Xehw5CdD<A9tS%+jn)bod22w5?Hriam7e?l0|%>gKtEb6s!%1mGM||#?w$YVO4C&j^)-83yE)os~9tPIw<M;O;%xr;1~wDkaV87Zi?hrdMT2XW*MR}UJdH<QlA&9cahOu>-~<Wb!w+eKt<S9FO9R7$!0{iTvMz_qGvc;@>^S|@t85>@t%cz|6DJ6Lzc$Q?@KD|1+`39)iK`-s5<B2J<eAt<gTFy0qRToB5|naTilm%In6>jh5l<-KYf}~U}o_2mZb#=UPu{>umgX6iqj-Z!S6Y4xO187dtSQ7Z^hTUHflIMg4W-9;%h!k+fMg@znW(Q0V&&~Xx@ISWO*D<LCbX+5O+R$;-#ISi?ImO*L#5K<m_fm1yDrfz^|<4Exc4@LS9_Oq&i#yeH?J#h>zg1+ca1f@IpbBDYI*g{A52z4afX3(*TWn6{jI_{ULUL4$q?qiRi6Wy!#gD7bdZFl>FwZ0~ud|m;MD_(-LIHC3|pQg;n4oM|!_7AQYJ?=0-X?t=v95ufjUoQ+++Im~Fl)$HnP4jr{3ZTX$Rh`uz4IXpr8SX;8aZ==weUUxr_!`31!1s(bIrAsv*ccP*5-pA8rsM(UGG(J3@?9QIJ?5D+m2E}2*5Xt+*^DO5|8r>$9Y33(tJ&|kpJV*f>=B#Vj+`4(wSqaV0ax5zTbEB+bSB*0nAa6#6g8wl;;*X=?qtcc!E8*UmrSiP-mX|)b=ZYi=3sa-eghcwt~zg27%_S{KXWZe?CtsOE%p0xO9@}mLc>Zo57zAuDS#<{LxR|y<ufWWEsnD{}^bO$;Dhd<XuehVfjp#Ik5j9QU!WvGn&L846X)`+x215fB%yv)4&+>(`n?B|?ge3N57lI=xQ1o${!yY_qgb_vS}9Z?KNqEzAi;owV#@Ve`Pl6X)_LCyhGhB9|EmYPuU^D_76xl42hRD}Ymyc)pHWZiCY=Gj{`5<+UC!!El|%k*iDv)zkGz4`s_k{-x2cGk#L^b4L8!<gDDY!LJgfKkCXwn`+(cnDC61Og<L%Ls}?Bg{nojUDX;I7CoB@G4A&$RaYWj5IJ=>}ylT+YN~l&LEHx@cuzWT^FZv7==FnFzW!GKKkf-AR42{wTf}!W|qXE(6Sh)0pE)|8AB}IQgSJflmpLN8!TB&F-T~g2ygMV>aEH&H_;?dG-Y<FwI%}x|HnaUC9hPC2)1}8kxKW4bDqs&<r&V|nAd_?Bf{EoMz1<3o&pNR$&aGr4wl!5Tr^50MDiC$>cKp8Jrr>fN8k4}y2+Q9F`h@`T~11M2o~OJQ&h?Blmq_!qu)`jt9vTxi~bj<(pWBj5k<<f7NL^O-b}Jj3W}Xpk*8d9jq8-1cK#;!Z@{bRRdH}#Vc;>M9k@IigP_q@9|%(&ZRqpwk&?`3xI?eE!Bcx=e$_QL)AiO^=LOD1pK;drO?_rPfAz2`*YJC>ks`6bu=*+IMs%z@6M(=E^o_Hmkv4)m_75N)j+5e#&|O~e8C<a@kymO?9UR6Kx3c?J0K~Rdw=mXxyH<Z>^v4-pOAZ-bWRG*P(q#w9o5QeE?|zP#QM@dhE~k1Q>gCsKjJwW#Z0R}v7w?~k%jai}h_i&d&DR<u^>&9(7_b4_kFg}P6Cb*8HE`!z$t{|*I|*V-@>K9LRwA-;`Du|wHe}_UR_A`GhB7iPXbRdR53{HyAw_dZ_UyDYsqR($wo>Hv=Un3maW;TH_TcEJhs_6=C=4braB$i#3EQu9H2r#|=n6Me2$EEBM;B-HKuU4i2t}=0lYPfGEDmrsx5{{#v@VW{Rgp9=CR42oy>3}Kc>c@nF$ZKOt%ekszINbrtixo>$X*<)GOdSrHWf;NS9b0DQmsc09+}V7uUn!wH7S_(yI$2*2O0&+iTCbvUjij<ZO2NR$=MaI3$UsX9v%>MZc!Lf&L0I`w&Z-+09^O0<Im=OR$1*$%8ECLqpxX5_l4V-)7h<HcS7+(Set;HHrHc2yip>9A<#QH=`m;vZ@)woO9xkP;OMvm+A>`Nv_BWn$UvjS3ly`1KyieegL=Mh5}efe$*Dp8Mn%U-^hgD%-@g8>1;76#e<avxxU?ez0JL%f06_o8+B@1AyBPhSU!88St?ahgAG`cQT;GC<Q&p|ou&+5y0-9aw0BbEWi)qk0_~S`8S4HbC4~DUy`}U^eJvChwpYi$+pq5i9XYJXu*~t68o#l-$2Danigqv3rWqKov%z{L)!ZK~>BD=*yY^;3Khb?M36j07&Z^sl^DZ&;_4vRDNGqY+kUd_<Nq@$QheQ1OMi!vaBv&JZ+jei*(f!=4A24@bs(#b8ZBVqPz_KEdM1|Po7A>(`@`aLe&4R71o67z>A)B^bEDma4jf)ZoPA(CQFjWv_UCv3AL9Zx6c{DWg|yxJr2A++435+))7`p{f4yoe~u{%}Qij{ueul_{`p%z1L@X(E^AA;0@j!h6JX<i&+ICM811#Uz!{nO{@MV0-J`u5UI#{dhlr;$=kV{EWhi2DYA#-k82t@Y9;0e@`a5$mihT$by<SJ|@E0UN1oLw~7|Zzy0h1_JiW`O!G`x2LNjaE*?%M&N6%)CrOQJpg$R((dU}t$jQk&HDo{TFB<J;$KseTrY1hB@j<(BV7MFR@7=8(&ibj+DYD~5VC4SMI{LCRmI1^5ddS}1e~_WeW^!T^GsZQy(Aj!W2(klvGuuPKkUNZuiDiPT1M?x_C}<eKFrta|TBXaO+6sDIomFXOR<ju%pa}RDfWaWo;ps|Y-Q$1PD>eidp<77U!t#A_bM-XOnT^>^Lmj$;76vHygj+Jd)1ej|i4mlC7rU&Nw6N^yEfdhai$^KNXJu0)qF}68IaQX#=0V;l-nyig)@OypzVpC>F_W*|29(-?zb&x!^vuGheKi*E9ybJB_XKV%UC~6w2WC08Y=pI)NZz;G!D9GM+JPm)>cbJi&aoz$QH&h2zBn0h3CK5Q>`I`$;{D{Gvcj!S$4<f`N^Z)?#B2&F-yW!7?S-FD%LQ*o=GMrcKr}Z%7W%&<$<&b!=pkdXthxe|PwCTLW_9E*))IN+4d24{t*Ipqoqalck3XlVcYQu@1YgX0R?ETX`E?QbTV?u@uY-xSGt+{Zcaa>%ibbFfT7-$1_ZKB`E+}G*rj<31+FRq{%qM*W_a4)y1HOXpB1X=@q1@pKKhz2*AY)>hmvl|~%MCK!4zIX-&fKb>dIerp^l6`Po}2V;WOgPfvE5Vzv)i8%jUx<B6i-sQW~zOhT29IyiEB4~-$LQ9KYw;oN`ah=D%{dVwaec(3wXC@)+1ue)?!BcyZ@Q$PK@xkgExn}-X0&a_?I;-dUNY;fyth<+W0rpVDNs)>bFu44Qg@;5VGsDOhSr1g64RgL{|3=$(8Jedr>+HZt<&Wl0kCPw5ir90-8ak*Sz_=m1|LvI-Stk6hp&kLSdWjuOMkvJGt90;^eV2Y7ovX$@EMDS<7E;Tlxm~u1-3xXS2|uK_|4@;A$AATLc&fIb(BX$K72?gUey~bFIn%$o>rb-;vxq6|Kz^q*5+ii$wy)V7|XvI?r_cULHSwzdM!+nIBjLgu{57IPzK`i+D<36U;k|b+Vw<exE@1`VNe>+)>wznjKd&VmR@M@fK!?bgYV3J=<)vPYe!ewe3aZw`4?39X(0;Y>H{R@Mq>B=D5plF-dSvRWu(0L;JE{Y1E-WN9>D3lE@&t)rvdbq$EXP@(5C^5tkp*P)P)A>|g2_0iwlFB_LWvYAy=uSssdL7uT+r1Y7CbO(XH&ttfVZv?lg9YLUZ61;?qwJ(%>1>6rXXQf4>v9+oY@_)FG^6n76-kAmbf(&U46V-!V8wwPHVvT#T{g;RXX(?>FYl#G(3JZk`U6^zwAj}S=@BQ={~j1(Kaw+GH_IYH!#MDhi>V}`dt>}OC6H*{S}aPo77AA$JALJrg%b%N}p<t||~2skDYwV`9AoW{K6*ZYqXv}WCNIU{VDcjm_4w+K0#Ja_2!pIxp-*%cAEa+5hYz_>lU5KT}lhn#`TVz#5BPoefod~<mr9#*bsGb7K^)I>fFu*i*lx?~Mt$USksw#^=bJmal{qj+IRu<qq#&!nSthr%${YhecO6OiLBQ|M*$QZ?|K46@fC>uf5Rrq@V_9Pdp?ybIh>U89@~Q?duPw$gaV#c2}L6m7v#t3XhgtUNMhJ%>@}DfAE*4EiI=_mzuH`)UQGmY{5wnh+L?<JgVc^F`u)B4GFX+q99L+Ln_6*3iggp_$>M<ShS!`Yy~45C!bastI!8!0!EkTI54-R&|`y1~?-waLs1|;L5576Zrs?v<FoaG&M_!!!V2pam8EICUsRAB$z)Dg{&PjPPy}W0T}fLB4~ohCJHRyqEaGJJnaDB!#5|bF**N<7~YILLloc+cJNjp4UZXKCEVWe3*Q1bhEz2@!Q1in<mTupZEE9%h#R_rk^^85D+`+Tzxn@W&2s(*(F77{JxnztnBFDo5+1K|Rl9%Ew?IgKU0;78=Eo(AS5Sz5`&PspZuxovd*IW#LC_w)5}pKp#PiFxaD>0j>`pXmqd_EK#8c=8deEcwH6^fM$xQpIDsjqLUy6?+hDLU0+@ytWC9-C+RH9vPKmcU790}Ruh3k+#H@w=j8F|TErMCOE^b=wsw7SEv4oGI`znE%>RB5Qt_0Qw-^c$mcPVgVwAc82t;u_w0tN&2q!+Z9-3G3!iw9b{X3TAT3Ce3C7u{0fJ%)eTfy6i7Ol9$*OsFx!m=`1DzS_cY;6-^l-QZTJCLESlv-w>w7760O~B#8T18B6NtF)hrbT*kCiY@Cdj!p1Gb3n*p`3joxj0ATY!lc7_V1N*<(?R+wUnRuAOWeTnWkkhTZ*T^;?7ACu7bq&i6I5X?pd6Qh0N7Y28&sA<3Ry)?``Tcr?ev;v$Xs#~J9ZlSyRQ2WHOZ)z$R3^lK7M<%8odH;W(loD`<<UE+Ls{{s)6%mL!*G1EdzO%nLpxg==RK7jiFzcJj%s$JQMxnS_;@>Tbju&q;0vTu5UIsWDbFXj8T$O|dPoY6Y^^=2JqL7>Q9zJYA9&LFhpBaf=XMOCI9&O>|K;Q_Hu~j*ITVzT+b$KH6@?G?_DO>wk&_qX6fX@x6&ey6>Vm~>o@i>2c*hW5PKU(SP$3;Q7X_h~0O{Zq_yHkptZE6}vgiAX28;#vWZ*QDAk?+ZrvDz1R<_*L6~bt^Ie~2h$>^&$`^)N08I<zNb5_Kwx68!>(6{F$r0-0N1hY-83gJf0;rpD3f%D{*quSsjmq$ePk{Ab2CP!8J4{V*q=}H1N^v<D?H1ty15@ZE|#P`Ki$ld7p$bgrT5RYR!stYi5+1hgZCjD@$kG=dH7H6LV^YL{LjW~JnjSr1tIqW@MU!Y?!X>sVf5}y=SZ$iAzxZ$_&h$fbqXrM6q4ABJ0<!}715BFybM<U7S56^8;)1XahBiED{>#A>eAYZd=Fh=PvtVcrT;>{C<6kd(~Bomygk$-Q>{pb)2b_xEg?6Gia6g!B1qCMFN>OQ(Ek@_zahzH+DTKyw)K#Z@q4sy>g?X_rECIJM{HxLFR7GBJ`pqZ**B&a2Fc*y@xHw|tb^o#5#e)~B^7qq!GoZFdxZRkdAk?(K{oJZ|u@HkRJ-v7)gk>;-KXy13w^p`w)YfYGP;IC)v&7+Ne_R-)JB<|7X9PU1|kDG}6z@#hMC7U)10cRTXR}xh~){rCDjO#L|06wfTH8-*j^fav*Z;CBqfBiQwB+Ui`X>f<WD;1~$y#JXCotWotu)S&>d}zGx_@$$hD-Hc@jSDh94O3|DPZw($_Evu94E78X1Rvx`jN{3;3Ibnlva3G_MgBw1434x~TWpK?8i!*33q2QjPPHIu7l!ipR?FoXp_qF1jbAqRjx5=@T0B$&>axquA%Z~16sJD^oDi?_;h?RAHs1eC)g3qwY<j@5{g`Wo@cO+}p$>F~gkK~;24$Rtd`Z{5=dIdj+0r24WK-Lmu7g$4w=ufzvyh%!eYgFVEgZ__v6a&UTQ=90V(RMtcdFONYhT^=rYk7r#9PW2^KxxuVoz3<PGdjVX>K8ReQEZF0;vi3*3tFX7k)qA06bz{(h#K}#%c@KwQ7g4UmVJn1umT?X!YOePvJx%N=6E>8|iiy2^b}p|2g=KlBGr65&Zxq{bQNEiO;_xE6lQcg4xG@II;-w79!+aLC(E-kZ#|HBci3*Tr1jleQ*rt;t_><1(hlo)n}+qQNZk5sJn9>8&F+MwB&GV-Ir^Lj(AEzu|W+9`ZXs+0q=}cPNBGHUJxR{+Ac?YAvVfn@kHnpg~YDh{f%t20%IrB;Mp7o19e+IV+T{f<eO2@NX*oJvQkcwlP60*wJw<rrg3bWK+&S+S}f~BQZ`O4j(OsEv$a>&{WR~jK#op}1({0UDkjF0^}15@55xDYS>rPB*m75b^adYDuMoG<=EO(PQe%S*%aX2?*xkt;b?CWn({>RrKn81`Nqe;T6}O?TvqSXP)t-kJt)6gxYY)1I?eB=WmnSyA=$GAc9ljrKE+1Hj*gF~%Zjpub-B#!>m6%Ro&uZk@E!kPF{l!-J%GH=0)w7fGCee~~+-;wJC_gezc4J$uz$tH~R$7c6XWpBg?v5my=2EhU$xehc9Rj4D>yLb~$0}#ao~7)TEy46cP1fM?r2+3Kwy0ClHbZX`sg>@3<noUbSflw$%<<VpS9^G<K_T6Ps1-D3>b9-6|1?N!D>NkiqI+q)5#2EmR6;sk6KwTtVsC0YU94=Z-4J737716As-H<Iq|-+G!Cqi~yuG|>Q+<WmGHVQ^Yc2(vs{{D2*7ym2l$T=lqti(O=(=aig`tYpy*EeE*`<ngS~KL|%q9=YtJ!+9{0j0yx%oxI5E9bT&H#~!0gZ<)Bwtr2kcp#6?rIJcUM3~th_kEvRqZ_7{;gDF35yYh$ya^S3>GO@svudC_?u+U=l!jq5-f&*OY+50egM=Ut=wYC9r@Uq;X$1MlNR#K^SRq8Ec}&QD%zn|<gnSP#U@ufTB2xA)^b(QYO!5SVs|<;77Rk}f;3yyZdIfhJrS$2d=ydkFr0pFJ-s1Y*WP2e$`HX3%tngyq!W!%k8}mWkexPMAKq^BRd*-9ajkyJK~Osn`*S#__D&h*1Hb^N^T!=PmoBHV0Yrv}gB@%26b6InygrmI2#=Lhj!t4<TjhPFifRJ?0o^^upnYnD&v5Omy1-mL;$kay$-5fcZ2{qU=qO#Xg(`h`g~?<Sg?(NhqHL_1VV9BWbWO?ntU6WtK5H#hU|nl2QQ~UHN#tOWVhZgKs7x75RIaz-EvSq#$JoU*YXNz)4_cxdaIomYjs39H>NPL5Gj>l(mw#QPBndcD5=GuM$BdU%3ZW)JT*j(JjI+v;mkkGd!ZkxPKV;&E4n7$GG$NS+dKF<RYcD2?@cl0DZLBz;_2>xj+`#%+pU2I)GL)Ufd#3HM`ri;ZZV?n5?(adSsGd8!R#eL?y=}RI8x)fb(t9di0x(V1i?5!6{q#)D*v8$g#^bf7Bw)Sd=C`Ee3jEl*c_@Q9T)pH5*MQ3W;Pm;*xj_iBXe!29GRKf5Es7*(`Gb^;91ckar?!x~nOMPSu9xaB(=hq0wrT?4m|H+%v8ujsITl4Ytm|Ub4Bv@u5>2?%&45Eq<g>P7s@C4RXsBni4!ZUT)<Eel{Ynd_{nMBk%+r557To_C81XXqIssThH^)8zt~Rv=uNSq&k2OCg<-^e^vcZn_%LN_i*)HT=DUmr}2L=N(5AX{K#{^~xmIM|Kj7F!HZLX+6N36Qjg;L+rC<o&CEyl7v@aSC_7ZVJpW`d=QaqueoT=eEzs9-lg*s~LiC2nCuck2isPE`w*FdZ*4@D27&lbbI4<SPtiA17O=`7mt*UAF!OoHwVxe`rK6a@;!AZnL4z*m6zj)3auTXft)oCvHr)h5Lp4e+8ab!Ap}!AOHYZFaQ9||5!(78+#)?2Nwft3ull2nRGL?ur{GJa&omv-m)cPKoH(bct@Uc)Z0$06CjZ$Duxx{$8+uToeCyw+KdP`oLl!Pom}a?p}SkMd2{U6d2O?JdvNScjkECk>68xh%abizn*+wvjq@wrnY!CzM#GE!+BtxZip?wAY5C6_f9dw+>B^bvZ0hXFxd%_~u8Yl@Pfspw=b&VNLhQqZH@9v}rq1S<%INxnbyP4Lvxj;Q+E-X1sHY@U1hH<@lNhT`OU7E|7mrPm1v%wr6pf`?(X*|64H`Y>l`lvu<+hSWsiCse%Xm7Pz~f$1%I@FsRt%k`zV$TjmzDzQL~0eM=TKW1CF+FaXpD_%u_ZTVjwibvTSGZwiYFBONXZeG*Zg1fDNFQvP9wTaO3PciOFP_gkio#^Mq8Hla=%!D76g3g0W)Xar9Fi70$I?)wvxMPc-Slq#w1GTj=UDiK(JER_70;+#yqN|ssl3OB!whu5x}beWCD4K5Q`OXNE%9QVE=R10Nh1yY7JU`ek!JtLhUu!6&?!C*IWm*syPWoXR*18g#ZW6ROsJmm1bvb8pK0>aRRR#p5l20=oEG|s0pTlG<(MPEd~pe^{fiRD%Z1nQ=udJlFukb#S8|kjPsW3&{WVtLYnCtaIoL%!hd%N2+IRs$k$M>w#;D&g<t}ccTDwD^9&6yC;PaI(u1wOcUlC84^~ibE{sfak5L2p{0>8iL;^Nlqf2KmFdVs32A4-6*Hl6hF{4<bgk0H+o^64;`M*aCTse`KCz#CAJGh{Kxyr*jXhLHhZr=kED*+oVcFd#;``>^Mu>%Y9ry}-~%&&0MMhq@8W9+n*iYAa7p_V^DjUmio3cX=p1_&@ogk|>qzX42Jw5T-#2mlZR3;+QCAM0dp;Amp3XX<ER^FK=^Z4_s0x9DMduPE8|5&>%Q3H;$T4hun7$)uWG`{^+%rCETiC-RTU@Luneu7@kq9tq^}C*t}PP-<8LZCcio7B1n@_`mImBh>I{V}QHPju(O)z3jb$$Ih-DAX`0ZCLo$w4Xn=${;045Ku&Ql_$LcqZYN(!MAK;OFS>c`^U+H0ISZ`Bx8SNE_fjmVkBH_4i&CWazrdhw+3j`s^_G7oL)mMc<y@3kDSuYApvzTuP>-VNO$G}o3@a;PIm-X(=D8SQ8WkS%&MB?!*@WjZ-7Dq?bF1<C<jT>cGJ;)!GhNjP0HuRa!_Hx(idb4J8jwb5BRGaFMV!cP!x-fZ?;aeurqK1(H05nLvLdis(n$omi(Qm25Y1<wAJwKOXqi1w!nRnPGeXXd^QDsdHT66gG)&~(qHS5D#;NnB(NFT)HtF0w+O(?w^myyFzI9zI!r+0f!u2{vX_Wmds^df|X%J#N5)G${vR)6Qcgd1sT(N-l4aRTHCrK_Z1N=^eqsq`j9%Q1!o(_kRA?IVpvqGumdtQ<W;g1iU9rNv-v+qj6{4$>HJMB}#|0`H6sQQC1foE^=A?(T>{$C@xJq<K<PXBp)4e0-_fc^hHZeeTh;!JCvqM&0nNRQ%kR~HVo7(4q>txB=ZU|^&e!jqaF$1-0^+L7aQ>HC#<uI)mvj<>j*<Tc0fwyRrnZxx|H3i;$Im{?c<BlE1fmqMBv<n0+^*>4{LWO5h0G7lCZKzXDzM+~}(MdMVBku%i^*!*;9ssq9J7|&sslFt@y*q>k>SqR4B+Kz@#4g3}nD)wImf`$xVy63X#0WA$mfc`Pkcb>m6mxpE09_{LI4SV<yu|IZ-`U<VCBRm%tYTnDL4hWrYbRqhp78k7#l^F2=TS^kNFKqyKRI!mJ#EnrV7D%Qxl2n<zvpmD5?Cv;$N}KKwE}16<OLPmu=dy({y>Ic$6gf{G6#49)TqMc8+1Zog)FVAm>K;#pcgUO;H1fsJc0DO4%5uV=!%xb{7&}S+Gg{Xa#EA-2Fha-8UVQkqwi}6rxNQbjW?ghO44QZKrnOd!m$Wn|{vjx=*6;Vd>5i++q=ZR{{H12P`T_cX-O~gS8B+XTDS<}-{67il>}X(QqGxR2Y(Q)2WM_Mmvj=4}W)pVzghIAv!1PJ?q9O@tVVKFknO-U{Io_qVDIsNEE@~yl!!!CCGLRkA{i>NLiskc^#rJ)jW#{|3wfFNi_xG}Ow_oP>HnVr;=l3*&zxy}K_wm}+_kR1f_j`HP_xGCC_m!si!}~Yrk(2kcmiPOJzV~yO*7wo&_x{26cjWeB_M7;I|MOtRMpbVJs#ZN8iK^Q`M5XFH5?Q1EfJ)VH<kIs525l4{H*@38;B1RS$9ITP&CDX!b?S<P%{ezK7E8c>Fs<a36L;qJ>J%F~zMk7+7b7pL4kA5s)6OIOr7V3A-{<_Py#C?m$rKpE5$yxq1t8vLimnKNb@r4K9Lt4?%QYwG2L^Hv(OS`ya~#XHfm`$E3Iun2ZnSsFjrT7QVSe4-lJ2vuE)tv`*Sd4^4u!Tk>B*wPdE>!(bK;@-X#YPfn>nW>{rZ&NigQ8cxI$s>I+*@n+(M^S!;J?Km7ccOv0Lgq(&wl`Q2UG&6qpAdDtUQo=xWDcT3FY;z0n)b7IVy;UH{C9KuUWZt;BsPf>H3<s^Txz5vD1Q_BT$eWb)?n|2e@!TGq$q*^!eOlMhTOy8ZK+@fVW`7A$*h{tI_FJQl-2e=zLf;X|uDU5ZD#QwlfjBg#k_m2FM>d3o-EDk}nuBi1=sM9m+Yz~}I>iu9pKC<!Z%MJ`eN^S%^^5G#}FxCIHKy;-?d=GyG+HQ|s8AYh<LdpZ2X6z|9mmQ^$ANq!QJ&Yol6&F(z@d#)omzBt+-{W04S=<q*$e$l`N{*JV6gtX~iF<$nA1LOVRFwdmCQN&RJ;>bi@Th>K~ao5$B2%aya3bNRf!E<4!KKj9E`u$?W<0W5vtLWL91Q>>SI-D_F%YBm)=9H{mfiiOPDbm+0V9+b;*?X8?R7`i)zTmn6t9vS06q`1bAEUD~E4a@I8Qc=TN3ZB=A9^?O$XlnX!sd+5yIUj=Z^AFTI2ySCp5lQ{h_0^SIQO#&CcutP8(FtNHY}5YxJ6%aVwbo1rV+B$kQrc1EPZj();yiwgF9#S{YvAJcNRqAH8OkJQDLDKK|W2)v~bk^807HL5WGfU)KfiZe#objpJD_*C2n!T3=L7f12}4!E%*g&LS<m#egf|v7EeKJQDE^Kl#0ixzaQ`Qw3cs!-5jY&;926H^>aigKfhr4*a0xVUDL^5p8az#^lF6}NG0566xIrLpNhA}{4Wfxs@kBi3LG@8$_75ZK=0py|F*HfhmgPnq)_Cz0A~Lp5Fwf#Fj&pW`%mbwFwBou0qz?;guT028^>PT^TxrAJwUkqdZ&~Lsm`?JX5F7)Hy`ZDl3-G=zv9pK-WkP+YXY}V?d&Ph{zEF*P(3qdXvqRS^-O4hXIvNrnsClmk`y56c}oYAY^`$!kuj2?8g>#4f7&&?29Jhn%MpPd7T(a}GsfnF0~ziR`^H$VgMb?*NfizmhtVtWW`v`PaBhlb^q1)V6x@OWUp0Js`xHb#Q(%4ipSH$lZKv^A>b3jcw1dmg_db)Zefx7Ga<pG-+TU+j-Vf+*v>y0wb_=OgD_d&F7|>bv6eXT8H(4tHHy=e;+NOUA8Nb`~dMxH1L(|G?txJoPYpbW`<ri;-N~VTEkjwAzuA;7<<W(AGdeYSZ>1c;GDT!1C{ic6T)NVN=oTG_OLCzf27a9ZC(T<@uXY=DvKsJ8*$8Y@%{PcD5ja1A(o$U@HA@kjEKgh%mPe^v*Mj+<Ctf8Q5yj}rTGtkT#qZd-DV5buKS<*qHZSm$BtM2%X=W}<qi$Lxs3iHfn*i#O!zi&8i<7oBY`}MQ$ffqm~GBu2Ia5)!3j7y49M2x!z?Ts_$^gbsTfBU<)Qi@zO18^T->s!@zdE>9sqXw!w_TFt*o91+D5gozYCY<SQ8>h|}@Gx#Z`X;6IaajXmbnvx#WSnqpZKs+|0dgk7Pq6)L8I@Pci*(l9J{?FiS+Dp0-4H!s4o=&mWY!p+^I&SZVej3i=6I*>;9*9gX})W8{~Y}Ejfy)L@}4bO9V0H6m>1gI6(L^_ygFD{?4ixC^YM4)gom4B2rRDXxVH>#*UFK`F03}()ipMIoNDm_R8LI31f2xQXtPe9p6`I390>D*6L0F8qaoQNhQz&DXUurS7z3F*=<MjQq7KFy{VDj}P1%AcpTcCOUWOrtwGmv&k8WbdQ{W8f;_Q{+%KRJt$+C38O7AWpN;Su6Fws+TJzzWRvGU2wb^$6s!@Jd+Rd;ERZUsbiK|ox*U=QtN$wfK#zS?U3DR}ieQiEI?lUykXcX8LEr<i2ujPD46?!gD!LU@?nS(aQkRmB~_B*dnMCyoyn+A*Iw4t~j4E@Q@CfB4S>2i3m#iTDhX?|sZXr^|`sN33Gbj<ZfdGC#xmoD(85PL2#tn-Hbigfd}DllQd{Dn0?zwCAc;(~dNQS~Ju{k&7Vs_Tfu!MB$Fs$Q^0ocvbb2O>IFpE`~Y5G>+PkobS9*q$Y1&gWQ;@ZuF3n1>3oQYo_uxdk|fB{gZhZ-MA_O9JiCbKvZc&m|0%5^bNvPVy8ygYKwf*<{cQ@o(~Lb=I0kD&oC34Lgv2p{_kW&0}!%L1U&}RbFjLR#F3VUI5yx$v(nXJA<o?uD@M~U+KgYyr^u~?OlgNo%NN@m@y^(=MyY9+hE(cLT8|Oe#-L(c+D@Sp3Qn14b0rQBDQ3Tw{HKGgvlEPlxg#Ho<jatNh^4rV9fAXuaRO1O%^!8JzjCb^zR9+~3Y{O|)H~lS60CfbS*eRBe?7uaYa&#9^iQC{>G;MyxOBe>!v6zJiDUedWIuhyN%-mS@`CTb%7X4;__3;3rLY;$mDJ6_lA{`_GA)m>tUd&sY4FAYlf8hT3s5*gW?pNeG~k^BQhCWE<TDzQ?=#|0C<flAy~V)~@6lTlC)BPi#YZErw3x>?+Gp(@6<R3pNYnVgYk-kwLnPS#2ie?VnU*`@{Rh<icLAr~q$?NlwJX`$y95KD0z!@lcj7LQP^k+x!KIh6a<eIzv|;Vig3XN(Soii~kEM_QudnkAYHEAecmmRk6d@wb7C?}yGyy@RhYm{by#xZG1Ze_tKtZGisX-7_dJR>YNGAu7k`O=$O(CHk14z4`d*_b#aQ<`mp4n@!S@V1Lr~P5gv)<>`erO+a9YZz&(tS&mOtQU&TQ~os<(u{T{%w4#`+eXARP#?nReHI2+8}pCkd{+qRIKXEGmderEf{gpuQzVpa$GJ_T%`8atA;^uk3QHtq1wE{ZxF!eok4Z&+GO6Z@<B!q#Kx|>s?K*khnNnG?*&o{>i5w<*IkBlMRv$VB=e8h>2g2y!zpM*zvV9Q>um;YhphU|v-j<={fNoJ*~iF5EOaZRuqaR&5-+L9PCD%}7Fkc2Asj<$Xs0Hl(cSEs)Y7Ug>$%vBgpUuCF0yG__v}!bzdAfV!PP2li9GY(lMt=E+Up^v&=(eZ_ml@TB{#2nBXI83R&{Uw9F$y0ep)Y=StvB*R?5H-7AK8tSpqLD0bBf#uKI2s-vW>&zXsw&Ob>GCIcurnsBRL3?bdxjfY_xOeN$zj^w9Js&I3kcKzKoSLZ)E)izgCR4lk6Y5Fb+H2otv`&EF(`DWK_<j~A);3r`D^bI65<z?$!OOV|~TG=rLmJP)QQkdHv5?LPj?J#|o6!UOYYmM71b0|;plGo(<>Ctj3U-oVvGn$)h$_nj?G*tO@yyqk`5(D=uM^y+Bdwf)MvPgy6cI+>(SS*CGGwR`#^3w}@740DHH?S0Z9u{2o^_M{;4OTYQEff<=H)?}YWRy|8cVJ(ES#0VxxaLMuvi%45d$Nr-YA5M#|mmM;b%Ys43-LvujgE%_YNT_GO0WtCTu!C;V?raqmVG}QLToDhOyO@6XbY+;z@F<YmlUDd%{!iPVKnIN2x5U6k@xVW=zDxJQl<j&|%nR4=cBmW#Wn6N}`(aNpW2X47#9vb#1X$W94GEDKGz##vFE<WXgN1Db+#QS^120yB)_*i+Fu6qRyGAZ^+C5NR1GbLxH&<|fDD_SaInf^?BV-xsmHfq2^LddXi7R|FeDVo0evkBhyO{jtH(=9@Hg_z>EHIcG3=`YL#lRRBOzr(6`hihiWjp*;x0);cTz>UkGH$mr!}EEQYmPPWU^k~*>+yw4H*%qk)~#c#IEADt72nX6^2`1o!-I#KqLim?Ban;z@Db_>mrxs+X)GzSn>8RN@iC(LaEG#&JH9>_6z}S!u;o|@^`9t!^eRv`is6!SLxiDL%k>(`bHL9CAxHq8_pa=6&89bA%B@<xvgI&->0S{uPh{_Ho!n=0heV$-zg>P#D^{4k*^5tEIe#Lg)H9)PZ<ntF$||(vBJH=vjq&u)@-|xkSbOncDYXDv0aw@NA3)J9ZOx#uS^l-=QEA|0pKaWs)O9ug44$<p^a|!`&-#%nmy8z4dNLiWwI+L3FN!GX88(c^1@ut2=4p?`PKHOM!0@R~bMCe}`jW^o506b<wq{3ja@rEk8Ui;Vow_z6A32vV%m(T?tpX-?<Sg8A^DcAA-Z)i9;-8`z8*Ry_yv|`)3qLpC@t{0>lRXUGp8YP){HSou{9WTp^PQ|>(IrL#cA>fMLc~{j=pLQ^5ZtZJrS#7*Pp3$qu1!ye+EQ_WJ2JFew^Qf$1IOCL+>N%5+Zho_8%?)7v$k#@D$R+9!x-7?hxY}^0q?}zEw<=Y8_UVgvonq5vePwbjw{eMqsiKNYiDz$2@X|RJ^x+s=C!Q`_n04*jx7>O*UXJFFzWLYUfyybzD+_jq#rwebKymKK!jbmflZ%_<<JwiVOV=dz-vE+iQ4&$Pw8Mt=HXhyeZ7=CHm?|_0|hfCI&k$NR6&&LH0mQK&AS%+L|ygX#{GIl1XNc@#rat$HA{Y3_LMf|xF6^X+akZ3FXWk7nQh<QT@C^@>uJ}Nb;7f04G#h3<U5s?mNyk4H~U&j5zGb+S_t&bi0f$?!z7y)=57`Yijg}R;o89_>D(>J`DL77{G{GgldYG>(E2@=cosd}`$%K7R%vsIiEGkH-?V-fv8r70@jjo)XHgkjUPlg9&E5pxgqLn6YPd!IshU~KgLMMlafu7MGvAko@*>vwVFaS?;AjSu)B+<|?#N4HrZ~KF*KPR=_TL6maf8AJ=<2LBoT<14!|9goq7E|{m6S2yPp@s-yR8ixsRop#+&@}Dk|(776s}A*yAIGBpY|AvV}A6ocyo1&KCE|%Z)6-mzQKGcW4|sP-g(vl_no@UF9kEd{204U(+YqYZDn(J*SWlh2TX#f-luN58?HIV5sJz53Rfnw-qVd=RW!QaZVfVcQ=G!vL=nr*8B~xhl3!uCXu2Zrs+{Ujry9FvfRWYe+?y+8;K&!p*4e+5&x`X@7_{?mmdyOxuGd;r3ESOML391I%S9cSqjePw4>YNdHjcFN^bcap`om4kG$o5=k{-i$5}l>X?lV5F-pSVUsJ@`M&{k1ZmUV45B%<U!>WPz$uDI|;%={RM$eXj1h1zuQn>T$8jkp*Nv#rmIbh}cU27P|emgY{ga2QcXkLjLT8@5#dz8h!O6b4e+ycwxMw-`N2M_0q~1Pqi$(V+3majC=J2Y-VCw^EMVO&8S63<FB3)I@oNPC){^XF)hm{AdM|jZ(fxu7kqoLzlUyI5In-cuMu}oj_J`P)M1D?M6l>(K~RkLeNcEVvtDa<(@bYB%{ZVT>3kkNBw0q&<_&aDpqu8cb{&O+Z9EKfLhBpOqPlaT-$eFCuo$95`GSL3x=W(smu}bdMTntNKKha-%wjc?ymu@<9t0Fwz;;rB4JmrB<@bcoM~=@;t)cXHM5%{^F@AwJ-R~3HFNOmz)w{8tj3Ru+P3w=UN~$v^h*U5Eg*eor?TB3VzU97lWQ~>nBn}xh1(uMX7B2$+(WmLQok-oXY`C7n0Hrbldat^teW+c^`x8)a+LNdAARCu>;t(k=15TA{60=+kERHG?XR7NU-qKtMq!(+-8`tpTnXG)57O3J3NC95Q{Vfw1t**ZPF;eve;y-uQmIVEPT>l-01N!sR#WY6f+w7crJX17X&V2-M6=~KuP>W0fdv|h5m550YM-(k{$URwto#V)6$ib)ky6~mVp|Jml@z0MlH_Hm{_%5;A6wVHK5|9**(&NwX!@ty{K+Ak?S6|Zb2n}TSuZ!XHBNXS42`*8xjvc4LzoTLsVaq@toiFOS_^@zw&(e=OAMtqu35`+`E1WyBwYe)bT8f5dE!x8bEWAuYVS+2m};iLXp1Qy&N##w<li7cOpQros%-T<V=v_>;<gTc$<MgK`4wDw)P|J|?K=FNBZ|%IkPYR!7dW;}jkQ_3KT@mE-Awy4R=)J)kCw7P^{FHF&`h6uQmyix<|b9(*5=B-?IJO_q<d3d+|9j^$w<mp60O%%TstsGS&OpWRP=BKoo09wzz_M}B!+K|XLhcH*u_Z3&qmKSggE*6hw#PlMsO8n^mGHIW&q7)GgWozTn_a>|Bf$i0)q7%hg&n=4)<`0{q%h&&p%qhD}L(tHFf!|c|Hz?X>t-<fc|T1qTq#?rO)p&&yWQ2ct+l1xlI}iF#7#RxI9PjTV|PIhgKy1w1>}DKe#Zn^HRAX-hx!;ZWA1id=HIuf$>(QnJ7$<Lr^`@$5z$qP^X3*$9gp=3bbw|=*&!so(KYexVNEwv9k_2f7)ZOoK<rDZkg61IQys#FxREjE>+5{Ll{=k<?DCcLw8m9Nq^HL4S50NJ9G*%WrrqNYsXh<a8zOV1+bKsKJ{jdiO9+DjGgV8^GScw1;lrd$|Y1&OyqR4FMTF0uB)TwtqjNA*I_hgNxGvvSyAqq?8i4^X7-ydzwy0=<~Yu&!St<MAO0qC5Pa>gzV$z-YB)}hv{T=RYHam}@(+MT!;%!qF~r2X7&&;Z*H~j3@g5beZ^c`hFfv%G{Am)~<P5M~S@QkI8<b^v0nRqp+7p0-I-*YdM6KBAH%h}ZR6vi|gGwfO<{d*k>&3o%FV~kXG`8t?Z4z=PALzU;8{Y*R^GI#4_VJ0Q`ScNke^gg&&!ezM*PDXh2njb|>N)l-j+i1fSn#r590awi#TAQyf<PkJeLL=B5n@g|8}W$F0PW<;aAuzEFRs^!{KFWMGX`lju*xwAcM{^U0Ae6A->Rl=iB1qlG4v;S<n+(}eyFPJ6L^uNPDZo0pO(O>LnVjD#F<R{PG!XH)-cm)c(jAM#PPtN)-R$eOujT_3}vRu+a^V1*YI8r-a&kTwOujf5<mLLCf5;vtBAXI+Lf%ca2##2|AZ<+uV9~D0fL!FcW3#iv*>zTdpGd<*)((sSTH9k-+(@nf)q1~PD<6M<2^oPzX6|>fX_~CS!Z?zgXTU2T{gSa3IUqMU+RlBs=}T`DYcJyFN#XE7Gh79@7!)y5N{5}mrczL=X)U+t#QYNiecU9e^P?K!Er%iQvFcHA>S_VN~;68+Ekt`uX_8SkBKc_JSvYniN~LacC$Jl7s^r0tBc8BWy0DN(FJ5da(!gvFj($M@23;a2~1pHjHTyj%>&PO@KcXzgTCw{(GPhTHB5F#u;R>k@ydft&u|;z;J>6K$t{TuEGtact-S25Yhe;*B}m({`d(G41p;+S9zV{eOV=f~A5-8oY3#yv!$&j=Dm}$)mP1Jz6Ok5y0mt0b<A443Hge{3H7@CP&OTKP34i@aMb#K{&jZkVR^P`8EC*zsl6H$_ouv5#$vU$K6vL=cwa4jAC}N0Mh8}tXj9tlgpZd#k+8kJypf>W#of)p}+LTMVtSaaoGZcX;EoQtx=I9~~9yD-YfG-S)8x;ihLe`w^hdjKrgL$9d;V*2+4Jx@!D|cFO;3-eim9$8RFYBa##{^FJoUP4lcA-(ZdfI;mg|!mSCa{4ls#kYeoS0HLM=^wFhQ}-~&8mLA7Bh|bf6U_N_W+VCg<~r8&-SPb@*u2Q(lzTMI(QV;D?u>aqr`|g`)ecqbL`}gyMc0hQ46*ZZ#SsDYq6`OS%OyzY1mY_@32c<La8o~N`3WKP#g`sTSjA-*Av?nGW=tdTtv4zH9_>ITWWyGlz=ge$-G$=oj4r5gFCN#uw_&oY3?z<`17wEaz`-~t#NgAoc)NA-9Go&5e!Yb$uyzxh`#6~<-w-%?f~<P_4YUpjn=^r!P56YF5fLkQm>0uE^j2=3?{!1EV-*JRar2@Q`ORXYwjmdmv4v~Z?UII?Eu+55&$nXFQncdEh`z~dzc*4kC4LNnPC(cfVicESVA>NaTiyv<nL_2)enNG^I{p?Fl#Nfh}bbS#g0YvA<&p~k_E!d2*kV&Vj=CH9<hePs-KJfOtYYoRB9&Wqn`=k2Lc4PTDr|3mm{erPy4U#*PXC=mQSxKybqe&FOP7H@Fl3{T73l(C#mVS3TC!)RFt-XW0MJRFQ_T1yutdhA!$(M-DACM%EPc`0sY$Q;3!_d&3YPJ8tPSo2#g)95U`sw?AzKN507+AK0<}vJ$cH<6StJWXA;4#Ew0$23axq`pez-DD=*DiOd?6laJiKlre8m%DmqD1K3!G@8bW{+mneT*IuA!?;je)Jz*Wlsaoqm*dJ6pe`B2Zn%SG1%9sm~$Km?p;K_uj~7Xtvm!Uq7*{DbZJMLYTW|LxMAhZYE!1zK7F0LcyjaQ*`_W%@^taeo&l7muL-u17hqalSM8Z;fP*|J<kiyu$ft^xq0b(&vKG|An554f>73bpAt#(0TlP>d61W_5VW#$$9*I)y{9+*X|!RJm<mlr$K&$hJOFg*^u)x=RNt~GOPjra_G;)=k49!FlEqrv-doE-h2Gb76$*zi8O>zQUATj`0HN%^)YzL-?RS$`&u>^'


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(cmd: list[str], log: Path | None = None, env: dict[str, str] | None = None,
            check: bool = True, timeout: float | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    p = subprocess.run([str(x) for x in cmd], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=env, timeout=timeout, cwd=cwd)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(p.stdout or "", encoding="utf-8")
    if check and p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout[-6000:] if p.stdout else ''}")
    return p


def extract_payload(code: Path) -> None:
    import zlib
    raw = zlib.decompress(base64.b85decode(_PAYLOAD_B85.encode("ascii")))
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        bad = [n for n in z.namelist() if Path(n).is_absolute() or ".." in Path(n).parts]
        if bad:
            raise RuntimeError(f"Unsafe embedded payload member(s): {bad}")
        z.extractall(code)


def real_user_home() -> tuple[str, Path]:
    user = os.environ.get("SUDO_USER") or os.environ.get("SATC_REAL_USER")
    if user and user != "root":
        return user, Path(pwd.getpwnam(user).pw_dir)
    # This experiment was built for the audited Ubuntu host. Keep the exact
    # historical location first; fall back only if the same account is absent.
    try:
        return "mininet-ovs", Path(pwd.getpwnam("mininet-ovs").pw_dir)
    except KeyError:
        user = pwd.getpwuid(os.getuid()).pw_name
        return user, Path(pwd.getpwuid(os.getuid()).pw_dir)


def ensure_root(argv: list[str]) -> None:
    if os.geteuid() == 0:
        return
    sudo = shutil.which("sudo")
    if not sudo:
        raise RuntimeError("Mininet/tc require root and sudo is unavailable.")
    script = str(Path(__file__).resolve())
    os.execv(sudo, [sudo, "-E", "/usr/bin/python3", script, "--root-run", *argv])


def chown_tree(path: Path, user: str) -> None:
    if os.geteuid() != 0:
        return
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return
    for root, dirs, files in os.walk(path):
        try: os.chown(root, pw.pw_uid, pw.pw_gid)
        except OSError: pass
        for name in dirs + files:
            try: os.chown(os.path.join(root, name), pw.pw_uid, pw.pw_gid)
            except OSError: pass


def ffprobe(path: Path) -> dict[str, Any]:
    p = command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
                 "-of", "json", str(path)])
    data = json.loads(p.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No video stream: {path}")
    return data["streams"][0]


def rate_float(s: str) -> float:
    if not s or s == "0/0": return 0.0
    a, b = s.split("/") if "/" in s else (s, "1")
    return float(a) / float(b)


def base_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "SATC_TRT_CACHE_DIR": str(root / "trt_cache"),
        "GST_PLUGIN_SYSTEM_PATH_1_0": "/usr/lib/x86_64-linux-gnu/gstreamer-1.0:/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0",
        "GST_PLUGIN_SYSTEM_PATH": "/usr/lib/x86_64-linux-gnu/gstreamer-1.0:/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0",
        "GST_FEATURE_RANK": "nvh264dec:MAX,nvh265dec:MAX,nvav1dec:MAX,nvh264enc:MAX,nvh265enc:MAX,nvav1enc:MAX,x264enc:NONE,openh264enc:NONE,avenc_h264:NONE",
    })
    return env


def gst_env(root: Path, label: str) -> dict[str, str]:
    env = base_env(root)
    runtime = root / "runtime" / label
    runtime.mkdir(parents=True, exist_ok=True)
    runtime.chmod(0o700)
    env["XDG_RUNTIME_DIR"] = str(runtime)
    env["GST_REGISTRY_1_0"] = str(root / "runtime" / f"registry-{label}.bin")
    return env


def preflight(root: Path, code: Path, user: str, home: Path) -> dict[str, Any]:
    print("\n[1/4] Preflight: environment, files, codecs, model, and input integrity", flush=True)
    required = ["ffmpeg", "ffprobe", "cmake", "c++", "nvidia-smi", "gst-inspect-1.0", "iperf3", "ss", "tc"]
    missing = [x for x in required if shutil.which(x) is None]
    if missing: raise RuntimeError(f"Missing required executable(s): {missing}")
    if not SATC_PY_ABS.is_file(): raise RuntimeError(f"Missing exact SATC Python environment: {SATC_PY_ABS}")
    if not MODEL_ABS.is_file(): raise RuntimeError(f"Missing exact MoST-Sal model: {MODEL_ABS}")
    if sha256(MODEL_ABS) != MODEL_SHA256: raise RuntimeError("MoST-Sal checkpoint SHA256 does not match the documented experiment.")
    if not SDK_ABS.is_dir(): raise RuntimeError(f"Missing NVIDIA Video Codec SDK: {SDK_ABS}")
    if shutil.disk_usage(root).free < 8 * 1024**3:
        raise RuntimeError("At least 8 GiB free space is required before starting the campaign.")

    # System Python owns Mininet + GI; the SATC venv owns Torch/ORT.
    command(["/usr/bin/python3", "-c", "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"], root / "preflight_gi.txt", env=base_env(root))
    command(["/usr/bin/python3", "-c", "import mininet; print('mininet import OK')"], root / "preflight_mininet.txt")
    ort = command([str(SATC_PY_ABS), "-c", "import onnxruntime as o, torch, cv2, numpy as n; print(o.__version__); print(o.get_available_providers()); print(torch.__version__); print(cv2.__version__); print(n.__version__)"], root / "preflight_satc_python.txt", env=base_env(root))
    if "TensorrtExecutionProvider" not in ort.stdout or "CUDAExecutionProvider" not in ort.stdout:
        raise RuntimeError("The SATC Python environment does not expose both TensorRT and CUDA ONNX Runtime providers.")

    gst_required = ["webrtcsink", "webrtcsrc", "rtpgccbwe", "h264parse", "h265parse", "av1parse",
                    "rtph264pay", "rtph264depay", "rtph265pay", "rtph265depay", "rtpav1pay", "rtpav1depay",
                    "nvh264dec", "nvh265dec", "nvav1dec"]
    gst_status = {}
    for elem in gst_required:
        p = command(["gst-inspect-1.0", elem], env=base_env(root), check=False)
        gst_status[elem] = p.returncode == 0
    dump(root / "gstreamer_elements.json", gst_status)
    bad = [k for k, v in gst_status.items() if not v]
    if bad: raise RuntimeError(f"Required GStreamer/WebRTC/codec element(s) missing: {bad}")

    video_meta = {}
    for name, fn in VIDEOS.items():
        p = INPUT_ROOT_ABS / fn
        if not p.is_file(): raise RuntimeError(f"Prepared input missing: {p}")
        probe = ffprobe(p)
        if int(probe["width"]) != WIDTH or int(probe["height"]) != HEIGHT:
            raise RuntimeError(f"Wrong prepared geometry for {p}: {probe['width']}x{probe['height']}")
        if abs(rate_float(probe.get("avg_frame_rate") or probe.get("r_frame_rate")) - FPS) > 1e-6:
            raise RuntimeError(f"Prepared input is not 60 FPS: {p}")
        if probe.get("nb_frames") not in (None, "N/A") and int(probe["nb_frames"]) < 1440:
            raise RuntimeError(f"Prepared input is shorter than the documented prepared sequence: {p}")
        video_meta[name] = {"path": str(p), "sha256": sha256(p), "ffprobe": probe}

    command(["nvidia-smi", "-q"], root / "gpu_before.txt")
    command(["ffmpeg", "-version"], root / "ffmpeg_version.txt")
    command(["gst-launch-1.0", "--version"], root / "gstreamer_version.txt", env=base_env(root))
    return {"video_inputs": video_meta, "satc_python_output": ort.stdout}


def build_encoder(root: Path, code: Path) -> Path:
    print("\n[2/4] Building the dynamic-bitrate P4 NVENC bridge", flush=True)
    build = root / "build"
    command(["cmake", "-S", str(code), "-B", str(build), f"-DSDK_TOP={SDK_ABS}", "-DCMAKE_BUILD_TYPE=Release"], root / "build_configure.log")
    command(["cmake", "--build", str(build), "-j", str(min(4, os.cpu_count() or 2))], root / "build_compile.log")
    encoder = build / "nvenc_dynamic"
    if not encoder.is_file(): raise RuntimeError("NVENC bridge did not build.")
    return encoder


def benchmark_matrix(root: Path, code: Path, encoder: Path, continue_on_fail: bool, av1_input_mode: str = "annexb") -> list[dict[str, Any]]:
    print("\n[4/5] Uncapped capacity benchmark: 3 videos x 3 codecs, MoST-Sal on every frame", flush=True)
    rows = []
    env = base_env(root)
    for vi, (video_name, fn) in enumerate(VIDEOS.items(), 1):
        video = INPUT_ROOT_ABS / fn
        for ci, codec in enumerate(CODECS, 1):
            tag = f"{video_name}_{codec}"
            out = root / "benchmarks" / tag
            out.mkdir(parents=True, exist_ok=True)
            print(f"  benchmark {tag} ...", flush=True)
            cmd = [str(SATC_PY_ABS), str(code / "producer.py"), "--video", str(video), "--model", str(MODEL_ABS),
                   "--encoder", str(encoder), "--codec", codec, "--output-dir", str(out), "--initial-bitrate-bps", "36000000",
                   "--benchmark-frames", "1200", "--benchmark-warmup", "240"]
            run_env=env.copy()
            if codec=="av1": run_env["SATC_AV1_ANNEXB"]="1" if av1_input_mode=="annexb" else "0"
            p = command(cmd, log=out / "benchmark_console.log", env=run_env, check=False, timeout=240)
            if p.returncode != 0 or not (out / "benchmark.json").is_file():
                row = {"video": video_name, "codec": codec, "status": "FAILED", "fps": None, "at_least_60": False, "return_code": p.returncode}
            else:
                data = json.loads((out / "benchmark.json").read_text())
                row = {"video": video_name, "codec": codec, "status": "VALID", **data, "return_code": p.returncode}
            rows.append(row)
            fps_txt = "FAILED" if row["fps"] is None else f"{row['fps']:.6f} FPS"
            print(f"    {fps_txt} | >=60: {row['at_least_60']}", flush=True)
    with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    all_pass = all(r.get("status") == "VALID" and r.get("at_least_60") for r in rows)
    dump(root / "benchmark_verdict.json", {"planned": 9, "valid": sum(r.get("status") == "VALID" for r in rows), "all_at_least_60": all_pass, "rows": rows})
    if not all_pass and not continue_on_fail:
        print("  STOP CONDITION: at least one exact uncapped case is below 60 FPS or invalid.", flush=True)
        print("  Network sessions will not start unless --continue-on-benchmark-fail is explicitly used.", flush=True)
    return rows



def _read_exact_timeout(stream: Any, n: int, timeout: float) -> bytes:
    data = bytearray(n)
    view = memoryview(data)
    pos = 0
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        sel.register(stream, selectors.EVENT_READ)
        while pos < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not sel.select(remaining):
                raise TimeoutError(f"producer wire preflight timed out at {pos}/{n} bytes")
            k = stream.readinto(view[pos:])
            if not k:
                raise EOFError(f"producer wire preflight EOF at {pos}/{n} bytes")
            pos += k
    return bytes(data)


def producer_wire_preflight(root: Path, code: Path, encoder: Path) -> dict[str, Any]:
    """Fail fast if any diagnostic text contaminates producer stdout.

    The network sender treats producer stdout as a binary protocol. This check
    executes the real MoST-Sal + NVENC producer and verifies the first complete
    paced access-unit record before any 150-s Mininet session is attempted.
    """
    print("\n[wire preflight] Verifying producer binary protocol before network campaign ...", flush=True)
    outdir = root / "wire_preflight"
    outdir.mkdir(parents=True, exist_ok=True)
    stderr_path = outdir / "producer_stderr.log"
    video = INPUT_ROOT_ABS / VIDEOS["basketball"]
    cmd = [str(SATC_PY_ABS), str(code / "producer.py"), "--video", str(video), "--model", str(MODEL_ABS),
           "--encoder", str(encoder), "--codec", "h264", "--output-dir", str(outdir / "producer"),
           "--initial-bitrate-bps", "12000000", "--paced"]
    env = os.environ.copy()
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    with stderr_path.open("w", encoding="utf-8", buffering=1) as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=0, env=env)
        try:
            wire = struct.Struct("<4sQII")
            hdr = _read_exact_timeout(proc.stdout, wire.size, 120.0)
            magic, fid, size, applied = wire.unpack(hdr)
            if magic != b"NET2":
                raise RuntimeError(f"producer stdout protocol corrupted before first record: magic={magic!r}")
            if fid != 0:
                raise RuntimeError(f"producer first paced frame id is {fid}, expected 0")
            if size <= 0 or size > 64 * 1024 * 1024:
                raise RuntimeError(f"producer first encoded AU has invalid size {size}")
            _read_exact_timeout(proc.stdout, size, 30.0)
            result = {"magic": magic.decode("ascii"), "first_frame_id": fid, "first_au_bytes": size,
                      "applied_bitrate_bps": applied, "pass": True,
                      "purpose": "verify stdout is binary NET2 only before starting 18 network sessions"}
            dump(outdir / "wire_preflight.json", result)
            print(f"  PASS: NET2 frame {fid}, {size} encoded bytes, applied bitrate {applied/1e6:.3f} Mbit/s", flush=True)
            return result
        finally:
            try:
                if proc.stdin and proc.poll() is None:
                    proc.stdin.write(b"STOP\n"); proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate(proc, timeout=3)

def configure_host(host: Any) -> str:
    intf = str(host.defaultIntf())
    host.cmd(f"ip link set dev {intf} up")
    host.cmd("sysctl -q -w net.ipv6.conf.all.disable_ipv6=1")
    host.cmd("sysctl -q -w net.ipv6.conf.default.disable_ipv6=1")
    host.cmd(f"sysctl -q -w net.ipv6.conf.{intf}.disable_ipv6=1")
    return intf


def run_tc(intf: Any, command_text: str) -> None:
    out = intf.node.cmd(command_text)
    if "RTNETLINK answers" in out or "Error:" in out:
        raise RuntimeError(f"tc command failed: {command_text}\n{out}")


def set_link_capacity(sender_intf: Any, receiver_intf: Any, mbps: float, initialized: set[str]) -> None:
    rate = f"{mbps:.6f}mbit"
    for intf in (sender_intf, receiver_intf):
        name = str(intf)
        if name not in initialized:
            run_tc(intf, f"tc qdisc replace dev {name} root handle 1: htb default 10")
            run_tc(intf, f"tc class replace dev {name} parent 1: classid 1:10 htb rate {rate} ceil {rate} burst 256k cburst 256k")
            run_tc(intf, f"tc qdisc replace dev {name} parent 1:10 handle 10: netem delay 1ms limit 1000")
            initialized.add(name)
        else:
            run_tc(intf, f"tc class change dev {name} parent 1: classid 1:10 htb rate {rate} ceil {rate} burst 256k cburst 256k")


def terminate(proc: subprocess.Popen[Any] | None, timeout: float = 8) -> None:
    if proc is None or proc.poll() is not None: return
    try: proc.send_signal(signal.SIGINT); proc.wait(timeout=timeout); return
    except subprocess.TimeoutExpired: pass
    try: proc.terminate(); proc.wait(timeout=3); return
    except subprocess.TimeoutExpired: pass
    proc.kill(); proc.wait(timeout=3)


def wait_listener(host: Any, port: int, proc: subprocess.Popen[Any], timeout: int = 45) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None: raise RuntimeError("Sender exited before signalling listener was ready.")
        if host.cmd(f"ss -lnt | grep -E '[:.]({port})[[:space:]]' || true").strip(): return
        time.sleep(.5)
    raise RuntimeError(f"Signalling server did not open TCP port {port}.")


def wait_producer_registered(log: Path, proc: subprocess.Popen[Any], timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None: raise RuntimeError("Sender exited before producer registration.")
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if "registered as [Producer]" in text or '"roles":["producer"]' in text:
            return
        time.sleep(.25)
    raise RuntimeError("webrtcsink never registered its producer with signalling.")



def wait_sender_frames(outdir: Path, proc: subprocess.Popen[Any], timeout: int = 120, minimum: int = 3) -> int:
    """Wait for the live MoST-Sal/NVENC producer to deliver real encoded packets.

    This is deliberately separate from WebRTC producer registration. AV1
    codec discovery can depend on seeing an actual sequence header / coded
    buffer, while a cold TensorRT process may take substantially longer than
    the signalling server itself to become ready.
    """
    path = outdir / "sender_frames.csv"
    deadline = time.monotonic() + timeout
    last_count = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Sender exited before the live producer emitted encoded packets.")
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    last_count = max(0, sum(1 for _ in f) - 1)
            except OSError:
                last_count = 0
            if last_count >= minimum:
                return last_count
        time.sleep(.5)
    model_ready = (outdir / "producer" / "model_runtime.json").is_file()
    native_log = outdir / "producer" / "nvenc_dynamic.log"
    native_tail = ""
    if native_log.is_file():
        try:
            native_tail = native_log.read_text(encoding="utf-8", errors="replace")[-500:].replace("\\n", " | ")
        except OSError:
            pass
    raise RuntimeError(
        f"Live producer startup timeout: only {last_count} encoded packets after {timeout}s; "
        f"model_runtime_present={model_ready}; nvenc_tail={native_tail!r}"
    )


def wait_av1_parser_output(log: Path, outdir: Path, proc: subprocess.Popen[Any], timeout: int = 15) -> None:
    """Require proof that av1parse emitted at least one parsed output buffer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("AV1 sender exited before av1parse emitted an output buffer.")
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if "[SATC] parser-output codec=av1" in text:
            return
        time.sleep(.25)
    frames = 0
    path = outdir / "sender_frames.csv"
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                frames = max(0, sum(1 for _ in f) - 1)
        except OSError:
            pass
    raise RuntimeError(
        f"AV1 producer delivered {frames} encoded NVENC packets to appsrc, "
        "but av1parse emitted no output buffer. This isolates the failure to "
        "the NVENC AV1 elementary-stream/framing -> av1parse boundary."
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def validate_warmup(session: Path, sender_proc: subprocess.Popen[Any], receiver_proc: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + WARMUP_SECONDS
    while time.monotonic() < deadline:
        if sender_proc.poll() is not None: raise RuntimeError("Sender stopped during warmup.")
        if receiver_proc.poll() is not None: raise RuntimeError("Receiver stopped during warmup.")
        time.sleep(1)
    controls = csv_rows(session / "rtmpc_control.csv")
    receiver = csv_rows(session / "receiver" / "receiver_frames.csv")
    if len(controls) < 20:
        raise RuntimeError(f"GCC/RT-MPC warmup produced too few control samples: {len(controls)}")
    if not receiver:
        raise RuntimeError("Receiver produced no complete encoded access units during warmup.")
    now = time.time_ns()
    recent = [r for r in receiver if int(r["arrival_unix_ns"]) >= now - 5_000_000_000]
    # A nominal 60-Hz stream contributes about 300 complete AUs in five seconds.
    # Allow startup/jitter margin here; the 150-s measurement uses stricter checks.
    if len(recent) < 250:
        raise RuntimeError(f"Receiver warmup has insufficient recent complete-video traffic: {len(recent)} access units/5s")


def percentile(values: list[float], p: float) -> float | None:
    if not values: return None
    s = sorted(values)
    x = (len(s)-1) * p
    i = int(math.floor(x)); j = min(i+1, len(s)-1); f = x-i
    return s[i]*(1-f) + s[j]*f


def analyze_session(session: Path, codec: str, condition: str, rep: int, trace: list[dict[str, float]],
                    start_ns: int, end_ns: int, sender_rc: int, receiver_rc: int) -> dict[str, Any]:
    producer = csv_rows(session / "sender" / "producer" / "producer_events.csv")
    sender_frames = csv_rows(session / "sender" / "sender_frames.csv")
    receiver = csv_rows(session / "receiver" / "receiver_frames.csv")
    controls = csv_rows(session / "rtmpc_control.csv")
    cap_events = csv_rows(session / "capacity_events.csv")

    result: dict[str, Any] = {
        "session": session.name, "codec": codec, "condition": condition, "replicate": rep,
        "video": REP_VIDEO[rep], "measurement_start_unix_ns": start_ns, "measurement_end_unix_ns": end_ns,
        "measurement_wall_seconds": (end_ns-start_ns)/1e9, "sender_return_code": sender_rc, "receiver_return_code": receiver_rc,
        "expected_frames": EXPECTED_MEASURED_FRAMES, "trace_samples": len(trace),
        "trace_arithmetic_mean_mbps": statistics.fmean(float(x["capacity_mbps"]) for x in trace),
        "trace_min_mbps": min(float(x["capacity_mbps"]) for x in trace), "trace_max_mbps": max(float(x["capacity_mbps"]) for x in trace),
    }

    # Producer proof: take the first theoretically scheduled source frame at/after
    # measurement start and audit exactly 9000 consecutive source IDs.
    prow = []
    for r in producer:
        try:
            r2 = dict(r); r2["frame_id_i"] = int(r["frame_id"]); r2["due_i"] = int(r["due_unix_ns"]); r2["complete_i"] = int(r["completion_unix_ns"]); r2["lag_f"] = float(r["completion_lag_ms"]); prow.append(r2)
        except Exception: pass
    prow.sort(key=lambda r: r["frame_id_i"])
    first = next((r for r in prow if r["due_i"] >= start_ns), None)
    expected = []
    missing = []
    if first:
        byid = {r["frame_id_i"]: r for r in prow}
        start_id = first["frame_id_i"]
        for fid in range(start_id, start_id + EXPECTED_MEASURED_FRAMES):
            if fid in byid: expected.append(byid[fid])
            else: missing.append(fid)
    else:
        start_id = None
    lags = [r["lag_f"] for r in expected]
    edge = min(600, len(lags)//2) if lags else 0
    start_lag = statistics.fmean(lags[:edge]) if edge else None
    end_lag = statistics.fmean(lags[-edge:]) if edge else None
    lag_growth = (end_lag-start_lag) if start_lag is not None and end_lag is not None else None
    completion_span_fps = None
    if len(expected) > 1:
        dt = (expected[-1]["complete_i"]-expected[0]["complete_i"])/1e9
        if dt > 0: completion_span_fps = (len(expected)-1)/dt
    result.update({
        "producer_window_start_frame_id": start_id, "producer_frames_found": len(expected), "producer_missing_frame_count": len(missing),
        "producer_first_missing_ids": missing[:20], "producer_completion_span_fps": completion_span_fps,
        "producer_mean_completion_lag_ms": statistics.fmean(lags) if lags else None,
        "producer_p95_completion_lag_ms": percentile(lags,.95), "producer_max_completion_lag_ms": max(lags) if lags else None,
        "producer_first10s_mean_lag_ms": start_lag, "producer_last10s_mean_lag_ms": end_lag, "producer_lag_growth_ms": lag_growth,
        "producer_all_9000_complete_by_end_plus_1s": bool(expected and not missing and expected[-1]["complete_i"] <= end_ns + 1_000_000_000),
    })
    producer_pass = (len(expected)==EXPECTED_MEASURED_FRAMES and not missing and lag_growth is not None and lag_growth <= 20.0
                     and result["producer_p95_completion_lag_ms"] is not None and result["producer_p95_completion_lag_ms"] <= 500.0
                     and result["producer_all_9000_complete_by_end_plus_1s"])

    # Sender access-unit continuity.
    srows=[]
    for r in sender_frames:
        try: srows.append((int(r["frame_id"]), int(r["push_unix_ns"]), int(r["pts_ns"])))
        except Exception: pass
    s_in=[x for x in srows if start_ns <= x[1] < end_ns]
    sender_id_gaps=sum(1 for a,b in zip(s_in,s_in[1:]) if b[0] != a[0]+1)
    result["sender_access_units_in_window"] = len(s_in); result["sender_id_gap_count"] = sender_id_gaps

    # Receiver proof: V4 keeps webrtcsrc at the negotiated RTP boundary, then
    # explicitly depayloads and parses to one complete codec access unit / AV1 temporal unit per fakesink handoff.
    # PTS values originate from sender frame IDs and are preserved through RTP, so continuity checks
    # whether the 60-Hz frame sequence survived WebRTC transport/depay/parsing.
    rrows=[]
    zero_bytes=0
    for r in receiver:
        try:
            arr=int(r["arrival_unix_ns"]); pts=int(r["pts_ns"]); size=int(r.get("buffer_bytes","0") or 0)
            if size <= 0: zero_bytes += 1
            if pts >= 0: rrows.append((arr,pts,size))
        except Exception: pass
    # Exclude two seconds at each boundary so arbitrary measurement-phase alignment
    # cannot decide the result. The 146-s interior should contain about 8760 frames.
    interior=[x for x in rrows if start_ns+2_000_000_000 <= x[0] < end_ns-2_000_000_000]
    pts=[x[1] for x in interior]
    deltas=[b-a for a,b in zip(pts,pts[1:]) if b>a]
    duplicates=sum(1 for a,b in zip(pts,pts[1:]) if b<=a)
    median_delta=statistics.median(deltas) if deltas else None
    expected_delta=1_000_000_000/FPS
    pts_gaps=sum(1 for d in deltas if d > 1.5*expected_delta)
    media_fps=(1e9/median_delta) if median_delta and median_delta>0 else None
    wall_fps=None
    if len(interior)>1 and interior[-1][0]>interior[0][0]:
        wall_fps=(len(interior)-1)/((interior[-1][0]-interior[0][0])/1e9)
    caps_path=session/"receiver"/"receiver_caps.txt"
    receiver_caps=caps_path.read_text(encoding="utf-8",errors="replace") if caps_path.is_file() else None
    result.update({"receiver_interior_access_units":len(interior), "receiver_pts_gap_count":pts_gaps, "receiver_nonincreasing_pts_count":duplicates,
                   "receiver_zero_byte_buffer_count":zero_bytes, "receiver_median_pts_delta_ns":median_delta,
                   "receiver_media_fps":media_fps, "receiver_wall_fps":wall_fps, "receiver_caps":receiver_caps})
    receiver_pass = (len(interior)>=8700 and pts_gaps==0 and duplicates==0 and zero_bytes==0
                     and media_fps is not None and media_fps>=59.99
                     and wall_fps is not None and wall_fps>=59.0)

    crows=[]
    for r in controls:
        try:
            ts=int(r["timestamp_unix_ns"])
            if start_ns <= ts < end_ns: crows.append(r)
        except Exception: pass
    gcc=[float(r["gcc_raw_mbps"]) for r in crows if r.get("gcc_raw_mbps")]
    selected=[float(r["selected_target_mbps"]) for r in crows if r.get("selected_target_mbps")]
    actual=[float(r["actual_interval_encoded_mbps"]) for r in crows if r.get("actual_interval_encoded_mbps")]
    result.update({"rtmpc_control_samples":len(crows), "mean_gcc_mbps":statistics.fmean(gcc) if gcc else None,
                   "mean_selected_target_mbps":statistics.fmean(selected) if selected else None,
                   "mean_actual_encoded_mbps":statistics.fmean(actual) if actual else None,
                   "target_changes":sum(int(r.get("bitrate_changed","0")) for r in crows) if crows else 0,
                   "capacity_updates_recorded":len(cap_events)})
    control_pass = len(crows) >= 1200 and len(cap_events)==150

    result.update({"producer_sustain_pass":producer_pass, "receiver_continuity_pass":receiver_pass, "network_control_pass":control_pass,
                   "session_valid_and_sustains_60fps": bool(sender_rc in (0,-2,None) and receiver_rc in (0,-2,None) and producer_pass and receiver_pass and control_pass)})
    return result


def validate_udp(sender: Any, receiver: Any, mbps: float, session: Path) -> dict[str, float]:
    offered = mbps * .90
    server_log = (session / "iperf_server.log").open("w", encoding="utf-8")
    server = receiver.popen(["iperf3", "-s", "-1", "-p", "5201"], stdout=server_log, stderr=subprocess.STDOUT, text=True)
    time.sleep(.7)
    raw = sender.cmd(f"iperf3 -c {RECEIVER_IP} -u -b {offered:.3f}M -t 3 -l 1200 -p 5201 --get-server-output -J")
    try: server.wait(timeout=15)
    except subprocess.TimeoutExpired: terminate(server)
    server_log.close(); (session / "iperf_client.json.txt").write_text(raw, encoding="utf-8")
    start=raw.find("{")
    if start<0: raise RuntimeError("iperf3 UDP validation returned no JSON")
    data=json.loads(raw[start:]); end=data.get("end",{}); summary=end.get("sum_received") or end.get("sum") or end.get("sum_sent") or {}
    result={"configured_mbps":mbps,"offered_mbps":offered,"received_mbps":float(summary.get("bits_per_second",0))/1e6,
            "jitter_ms":float(summary.get("jitter_ms",0)),"loss_percent":float(summary.get("lost_percent",0))}
    dump(session / "udp_validation.json", result); return result


def transport_preflight(root: Path, code: Path, encoder: Path) -> tuple[list[dict[str, Any]], str]:
    """Short WebRTC canary. After removing the SDK IVF wrapper, AV1 first tries native low-overhead OBU/TU, then Annex B fallback; H264/HEVC run after AV1 passes."""
    from mininet.clean import cleanup
    from mininet.link import TCLink
    from mininet.net import Mininet
    print("\n[transport preflight] H.264 / H.265 / AV1 signalling, explicit RTP depayloading, and complete-unit counting", flush=True)
    results=[]; selected_av1_mode="obu"
    jobs=[("av1","obu"),("av1","annexb"),("h264",None),("hevc",None)]
    av1_passed=False
    for codec,av1_mode in jobs:
        if codec=="av1" and av1_passed: continue
        if codec!="av1" and not av1_passed:
            # H.264/HEVC already passed V4 and are unchanged by the AV1-only fix.
            # If both AV1 modes fail, stop the canary work here instead of wasting time.
            continue
        tag=codec if codec!="av1" else f"av1_{av1_mode}"
        out=root/"transport_preflight"/tag; out.mkdir(parents=True,exist_ok=True)
        net=None; sp=rp=None; slf=rlf=None; initialized=set()
        try:
            cleanup(); net=Mininet(controller=None,link=TCLink,build=False,autoSetMacs=True,autoStaticArp=True)
            sh=net.addHost("s",ip=f"{SENDER_IP}/24"); rh=net.addHost("r",ip=f"{RECEIVER_IP}/24")
            link=net.addLink(sh,rh,cls=TCLink,bw=35.0,delay="1ms",max_queue_size=1000,use_htb=True); net.build(); net.start()
            sname=configure_host(sh); rname=configure_host(rh); sintf=link.intf1 if str(link.intf1)==sname else link.intf2; rintf=link.intf1 if str(link.intf1)==rname else link.intf2
            set_link_capacity(sintf,rintf,35.0,initialized); time.sleep(.5)
            senv=gst_env(root,f"preflight-sender-{tag}"); renv=gst_env(root,f"preflight-receiver-{tag}")
            if codec=="av1":
                senv["GST_DEBUG"]="webrtcsink:6,av1parse:6,rtpav1pay:6,rsrtp*:5,rswebrtc*:4"
                senv["SATC_AV1_ANNEXB"]="1" if av1_mode=="annexb" else "0"
                renv["GST_DEBUG"]="webrtcsrc:4,rswebrtc*:4,rtpav1depay:5,av1parse:5"
            else:
                senv["GST_DEBUG"]="webrtcsink:4,rtpgccbwe:3,rswebrtc*:3"; renv["GST_DEBUG"]="webrtcsrc:4,rswebrtc*:3,rtp*:3,h264parse:3,h265parse:3"
            sout=out/"sender"; rout=out/"receiver"; sout.mkdir(); rout.mkdir()
            slog=out/"sender.log"; rlog=out/"receiver.log"; slf=slog.open("w",encoding="utf-8",buffering=1); rlf=rlog.open("w",encoding="utf-8",buffering=1)
            scmd=["/usr/bin/python3",str(code/"integrated_sender.py"),"--video",str(INPUT_ROOT_ABS/VIDEOS["basketball"]),"--model",str(MODEL_ABS),"--encoder",str(encoder),"--producer",str(code/"producer.py"),"--satc-python",str(SATC_PY_ABS),"--profile-csv",str(code/"rtmpc_quality_profile.csv"),"--control-csv",str(out/"rtmpc_control.csv"),"--output-dir",str(sout),"--codec",codec,"--width",str(WIDTH),"--height",str(HEIGHT),"--fps",str(FPS),"--fps-threshold","60","--min-bitrate","1000000","--start-bitrate","20000000","--max-bitrate","60000000","--signalling-port",str(SIGNALLING_PORT),"--control-interval-ms",str(CONTROL_INTERVAL_MS),"--horizon","3","--discount","0.95","--bandwidth-safety","0.80","--ewma-alpha","0.35","--fast-drop-ratio","0.85","--emergency-drop-ratio","0.65","--fast-down-safety","0.70","--fast-down-step-ratio","0.60","--fast-down-min-interval-ms","50","--fast-down-cooldown-ms","1000","--slow-up-interval-ms","500"]
            if codec=="av1": scmd += ["--av1-input-mode",str(av1_mode)]
            sp=sh.popen(scmd,env=senv,stdout=slf,stderr=subprocess.STDOUT,text=True)
            wait_listener(sh,SIGNALLING_PORT,sp,timeout=30)
            if codec=="av1":
                # Separate cold TensorRT/NVENC startup from AV1 parser/WebRTC discovery.
                produced=wait_sender_frames(sout,sp,timeout=120,minimum=1)
                print(f"  {tag}: codec-discovery priming AU emitted ({produced}+); waiting for av1parse output",flush=True)
                wait_av1_parser_output(slog,sout,sp,timeout=15)
                print(f"  {tag}: av1parse emitted a parsed buffer; waiting for WebRTC producer registration",flush=True)
                wait_producer_registered(slog,sp,timeout=30)
            else:
                wait_producer_registered(slog,sp,timeout=60)
            rcmd=["/usr/bin/python3",str(code/"integrated_receiver.py"),"--sender-ip",SENDER_IP,"--signalling-port",str(SIGNALLING_PORT),"--output-dir",str(rout),"--codec",codec,"--sample-interval-ms","250"]
            rp=rh.popen(rcmd,env=renv,stdout=rlf,stderr=subprocess.STDOUT,text=True)
            deadline=time.monotonic()+25; rows=[]; controls=[]
            while time.monotonic()<deadline:
                if sp.poll() is not None: raise RuntimeError(f"{tag} sender exited during transport preflight")
                if rp.poll() is not None: raise RuntimeError(f"{tag} receiver exited during transport preflight")
                rows=csv_rows(rout/"receiver_frames.csv"); controls=csv_rows(out/"rtmpc_control.csv")
                if len(rows)>=300 and len(controls)>=20: break
                time.sleep(.5)
            if len(rows)<300: raise RuntimeError(f"{tag} receiver produced only {len(rows)} complete AUs in preflight")
            pts=[int(x["pts_ns"]) for x in rows[-300:] if int(x.get("pts_ns","-1"))>=0]
            if len(pts)<290: raise RuntimeError(f"{tag} preflight has too few valid PTS values: {len(pts)}")
            bad=sum(1 for a,b in zip(pts,pts[1:]) if b<=a or b-a>1.5*(1_000_000_000/FPS))
            if bad: raise RuntimeError(f"{tag} preflight access-unit PTS continuity failures: {bad}")
            caps=(rout/"receiver_caps.txt").read_text(encoding="utf-8",errors="replace") if (rout/"receiver_caps.txt").is_file() else ""
            row={"codec":codec,"av1_input_mode":av1_mode,"pass":True,"received_access_units":len(rows),"control_samples":len(controls),"receiver_caps":caps}
            print(f"  PASS {tag}: {len(rows)} complete received AUs; {len(controls)} controller samples; caps={caps[:140]}",flush=True)
            if codec=="av1": selected_av1_mode=str(av1_mode); av1_passed=True
        except BaseException:
            error=traceback.format_exc(); (out/"PREFLIGHT_ERROR.txt").write_text(error,encoding="utf-8")
            row={"codec":codec,"av1_input_mode":av1_mode,"pass":False,"failure":error.splitlines()[-1] if error else "unknown"}
            print(f"  FAIL {tag}: {row['failure']}",flush=True)
        finally:
            terminate(rp); terminate(sp)
            if slf: slf.close()
            if rlf: rlf.close()
            if net is not None:
                try:net.stop()
                except Exception:pass
            try:cleanup()
            except Exception:pass
        results.append(row); dump(out/"preflight_summary.json",row)
    required_ok=any(r.get("codec")=="h264" and r.get("pass") for r in results) and any(r.get("codec")=="hevc" and r.get("pass") for r in results) and any(r.get("codec")=="av1" and r.get("pass") for r in results)
    dump(root/"transport_preflight_summary.json",{"all_pass":required_ok,"selected_av1_input_mode":selected_av1_mode if av1_passed else None,"rows":results})
    if not required_ok:
        raise RuntimeError(f"Transport preflight did not establish all three codecs; AV1 OBU and Annex-B modes were tried automatically after IVF unwrapping. See {root/'transport_preflight'}")
    print(f"  Selected AV1 input mode for long sessions: {selected_av1_mode}",flush=True)
    return results,selected_av1_mode


def run_network_session(root: Path, code: Path, encoder: Path, trace_data: dict[str, list[dict[str, float]]],
                        codec: str, condition: str, rep: int, av1_input_mode: str = "annexb") -> dict[str, Any]:
    from mininet.clean import cleanup
    from mininet.link import TCLink
    from mininet.net import Mininet

    key=f"{condition}_rep{rep}"; trace=trace_data[key]; video_name=REP_VIDEO[rep]; video=INPUT_ROOT_ABS/VIDEOS[video_name]
    session=root/"sessions"/f"{condition}_rep{rep}_{video_name}_{codec}"
    session.mkdir(parents=True,exist_ok=True)
    print(f"\n  session {session.name}: {codec}, {condition} ({TARGET_MEANS[condition]:.0f} Mbit/s nominal), {video_name}",flush=True)
    sender_proc=receiver_proc=None; net=None; sender_log_f=receiver_log_f=None
    start_ns=end_ns=0; sender_rc=receiver_rc=999; error_text=None
    initialized:set[str]=set()
    try:
        cleanup()
        net=Mininet(controller=None,link=TCLink,build=False,autoSetMacs=True,autoStaticArp=True)
        sender=net.addHost("s",ip=f"{SENDER_IP}/24"); receiver=net.addHost("r",ip=f"{RECEIVER_IP}/24")
        warmup_capacity=float(trace[0]["capacity_mbps"])
        link=net.addLink(sender,receiver,cls=TCLink,bw=warmup_capacity,delay="1ms",max_queue_size=1000,use_htb=True)
        net.build(); net.start()
        sname=configure_host(sender); rname=configure_host(receiver)
        sintf=link.intf1 if str(link.intf1)==sname else link.intf2
        rintf=link.intf1 if str(link.intf1)==rname else link.intf2
        set_link_capacity(sintf,rintf,warmup_capacity,initialized); time.sleep(.5)
        val=validate_udp(sender,receiver,warmup_capacity,session)
        if val["received_mbps"] < .75*val["offered_mbps"]:
            raise RuntimeError(f"UDP validation received only {val['received_mbps']:.3f} Mbit/s from {val['offered_mbps']:.3f} Mbit/s offered")
        set_link_capacity(sintf,rintf,warmup_capacity,initialized); time.sleep(1)

        sender_env=gst_env(root,f"sender-{session.name}"); receiver_env=gst_env(root,f"receiver-{session.name}")
        sender_env["GST_DEBUG"]="webrtcsink:4,rtpgccbwe:4,rswebrtc*:3,nice*:2"
        if codec=="av1": sender_env["SATC_AV1_ANNEXB"]="1" if av1_input_mode=="annexb" else "0"
        receiver_env["GST_DEBUG"]="webrtcsrc:3,rswebrtc*:3,nice*:2"
        sender_out=session/"sender"; receiver_out=session/"receiver"; sender_out.mkdir(); receiver_out.mkdir()
        sender_log_path=session/"gstreamer_sender.log"; receiver_log_path=session/"gstreamer_receiver.log"
        sender_log_f=sender_log_path.open("w",encoding="utf-8",buffering=1); receiver_log_f=receiver_log_path.open("w",encoding="utf-8",buffering=1)
        scmd=["/usr/bin/python3",str(code/"integrated_sender.py"),"--video",str(video),"--model",str(MODEL_ABS),"--encoder",str(encoder),
              "--producer",str(code/"producer.py"),"--satc-python",str(SATC_PY_ABS),"--profile-csv",str(code/"rtmpc_quality_profile.csv"),
              "--control-csv",str(session/"rtmpc_control.csv"),"--output-dir",str(sender_out),"--codec",codec,"--width",str(WIDTH),"--height",str(HEIGHT),
              "--fps",str(FPS),"--fps-threshold","60","--min-bitrate","1000000","--start-bitrate",str(int(round(warmup_capacity*1e6))),"--max-bitrate","60000000",
              "--signalling-port",str(SIGNALLING_PORT),"--control-interval-ms",str(CONTROL_INTERVAL_MS),"--horizon","3","--discount","0.95",
              "--bandwidth-safety","0.80","--ewma-alpha","0.35","--fast-drop-ratio","0.85","--emergency-drop-ratio","0.65","--fast-down-safety","0.70","--fast-down-step-ratio","0.60","--fast-down-min-interval-ms","50","--fast-down-cooldown-ms","1000","--slow-up-interval-ms","500"]
        if codec=="av1": scmd += ["--av1-input-mode",av1_input_mode]
        sender_proc=sender.popen(scmd,env=sender_env,stdout=sender_log_f,stderr=subprocess.STDOUT,text=True)
        wait_listener(sender,SIGNALLING_PORT,sender_proc); wait_producer_registered(sender_log_path,sender_proc); time.sleep(.5)
        rcmd=["/usr/bin/python3",str(code/"integrated_receiver.py"),"--sender-ip",SENDER_IP,"--signalling-port",str(SIGNALLING_PORT),"--output-dir",str(receiver_out),"--codec",codec,"--sample-interval-ms","250"]
        receiver_proc=receiver.popen(rcmd,env=receiver_env,stdout=receiver_log_f,stderr=subprocess.STDOUT,text=True)
        validate_warmup(session,sender_proc,receiver_proc)

        # Replay the exact 150 one-second capacities. No interpolation.
        capf=(session/"capacity_events.csv").open("w",newline="",encoding="utf-8",buffering=1); capw=csv.writer(capf); capw.writerow(["sample_index","scheduled_offset_s","applied_unix_ns","capacity_mbps"])
        start_mono=time.monotonic(); start_ns=time.time_ns()
        for i,row in enumerate(trace):
            due=start_mono+float(row["elapsed_seconds"])
            while True:
                rem=due-time.monotonic()
                if rem<=0: break
                time.sleep(min(rem,.01))
            if sender_proc.poll() is not None: raise RuntimeError("Sender exited during measurement")
            if receiver_proc.poll() is not None: raise RuntimeError("Receiver exited during measurement")
            set_link_capacity(sintf,rintf,float(row["capacity_mbps"]),initialized)
            capw.writerow([i,row["elapsed_seconds"],time.time_ns(),row["capacity_mbps"]])
        due_end=start_mono+MEASURE_SECONDS
        while time.monotonic()<due_end:
            if sender_proc.poll() is not None or receiver_proc.poll() is not None: raise RuntimeError("WebRTC process exited before the 150-s measurement completed")
            time.sleep(min(.05,max(0,due_end-time.monotonic())))
        end_ns=time.time_ns(); capf.close(); time.sleep(1.0)
    except BaseException:
        error_text=traceback.format_exc()
        (session/"SESSION_ERROR.txt").write_text(error_text,encoding="utf-8")
        print(f"    SESSION FAILED; preserving logs and continuing: {error_text.splitlines()[-1] if error_text else 'unknown error'}", flush=True)
    finally:
        terminate(receiver_proc); terminate(sender_proc)
        receiver_rc=receiver_proc.returncode if receiver_proc is not None else 999
        sender_rc=sender_proc.returncode if sender_proc is not None else 999
        if sender_log_f: sender_log_f.close()
        if receiver_log_f: receiver_log_f.close()
        if net is not None:
            try: net.stop()
            except Exception: pass
        try: cleanup()
        except Exception: pass
    if error_text is not None:
        result={"session":session.name,"codec":codec,"condition":condition,"replicate":rep,"video":video_name,
                "sender_return_code":sender_rc,"receiver_return_code":receiver_rc,"session_valid_and_sustains_60fps":False,
                "failure":error_text.splitlines()[-1] if error_text else "unknown failure"}
    else:
        try:
            result=analyze_session(session,codec,condition,rep,trace,start_ns,end_ns,sender_rc,receiver_rc)
        except BaseException:
            error_text=traceback.format_exc(); (session/"ANALYSIS_ERROR.txt").write_text(error_text,encoding="utf-8")
            result={"session":session.name,"codec":codec,"condition":condition,"replicate":rep,"video":video_name,
                    "sender_return_code":sender_rc,"receiver_return_code":receiver_rc,"session_valid_and_sustains_60fps":False,
                    "failure":"analysis: "+error_text.splitlines()[-1]}
    dump(session/"session_summary.json",result)
    print(f"    pass={result['session_valid_and_sustains_60fps']} producer_lag_growth={result.get('producer_lag_growth_ms')} ms receiver_media={result.get('receiver_media_fps')} FPS receiver_wall={result.get('receiver_wall_fps')} FPS",flush=True)
    return result


def package(root: Path) -> Path:
    out=root/"UPLOAD_SATC_INTEGRATED_REAL5G_RESULTS_V13.zip"
    exclude_parts={"trt_cache","build"}
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p==out: continue
            rel=p.relative_to(root)
            if any(x in exclude_parts for x in rel.parts): continue
            # Keep QA portable. No packet capture or large native products are required.
            if p.suffix.lower() in {".pcap",".pcapng"}: continue
            z.write(p,rel.as_posix())
    return out


def summarize(root: Path, benchmarks: list[dict[str,Any]], sessions: list[dict[str,Any]], planned_sessions: int) -> dict[str,Any]:
    bpass=all(r.get("status")=="VALID" and r.get("at_least_60") for r in benchmarks) if len(benchmarks)==9 else False
    spass=len(sessions)==planned_sessions and all(r.get("session_valid_and_sustains_60fps") for r in sessions)
    verdict={"version":VERSION,"benchmark_planned":9,"benchmark_completed":len(benchmarks),"all_uncapped_benchmarks_at_least_60":bpass,
             "network_sessions_planned":planned_sessions,"network_sessions_attempted":len(sessions),"network_sessions_successfully_measured":sum(1 for r in sessions if not r.get("failure")),"all_network_sessions_sustain_60fps":spass,
             "all_codecs_all_real5g_traces_sustain_60fps":bool(bpass and spass),"benchmark_rows":benchmarks,"session_rows":sessions}
    dump(root/"VERDICT.json",verdict)
    if sessions:
        with (root/"SUMMARY.csv").open("w",newline="",encoding="utf-8") as f:
            fields=sorted({k for r in sessions for k in r.keys() if not isinstance(r.get(k),(list,dict))})
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:v for k,v in r.items() if k in fields} for r in sessions])
    lines=["360-SATC INTEGRATED REAL-5G 60 FPS REPORT",f"Version: {VERSION}","",
           f"Uncapped MoST-Sal+P4 benchmarks >= 60 FPS: {bpass} ({len(benchmarks)}/9 completed)",
           f"Paced WebRTC/Mininet sessions sustaining 60-Hz complete-access-unit continuity: {spass} ({len(sessions)}/{planned_sessions} attempted; {sum(1 for r in sessions if not r.get('failure'))} successfully measured)",
           f"OVERALL all codecs/all traces sustain 60 FPS: {verdict['all_codecs_all_real5g_traces_sustain_60fps']}","",
           "Pass semantics:","  - uncapped benchmark uses exact unrounded FPS >= 60, with 1200 measured frames after 240 warmup frames;",
           "  - no measured source frame is read before the uncapped timer starts; final measured AU ends the timer;",
           "  - paced sessions require 9000 consecutive producer frames for the 150-s window, bounded backlog growth,",
           "    complete received access-unit PTS continuity at >=59.99 Hz, long-window arrival rate >=59 FPS, >=1200 GCC/RT-MPC control samples, and all 150 capacity updates;",
           "  - no frame dropping or deadline-based denominator change is used to create a pass;",
           "  - pre-consumer codec discovery uses 1 AV1 TU or up to 360 H.264/HEVC AUs (6 s at 60 Hz); once the WebRTC consumer exists the live pacing gate is released, and measurement begins only after the 20-s warmup;",
           "  - each session warms up at its trace's first capacity sample rather than the nominal trace mean;",
           "  - RT-MPC runs every 100 ms with 0.80 normal bandwidth safety; each GCC estimate can trigger an immediate fast-down, while upward changes are rate-limited to one action rung every 500 ms after a 1-s post-congestion hold.","",
           "Scope:","  This is a local Mininet/WebRTC sender/receiver experiment on the audited RTX 4060 laptop. It does not measure Meta Quest display FPS or motion-to-photon latency.",
           "  The 30-s prepared source clips loop during each 150-s trace replay; this repeats content but does not duplicate or skip admitted 60-Hz source frame IDs.",
           "  The RT-MPC operating-point profile is held fixed across codecs. V13 uses 100-ms control, 20% normal transport headroom, event-driven GCC fast-down, and one-rung slow-up; actual per-codec bitrate and full-pipeline timing are measured."]
    (root/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return verdict


def write_protocol(root: Path, meta: dict[str,Any], traces: dict[str,list[dict[str,float]]]) -> None:
    trace_summary={k:{"samples":len(v),"mean_mbps":statistics.fmean(float(x["capacity_mbps"]) for x in v),
                      "min_mbps":min(float(x["capacity_mbps"]) for x in v),"max_mbps":max(float(x["capacity_mbps"]) for x in v),
                      "sha256_of_samples":hashlib.sha256(json.dumps(v,separators=(",",":"),sort_keys=True).encode()).hexdigest()} for k,v in traces.items()}
    dump(root/"protocol.json",{
        "version":VERSION,"goal":"test whether 360-SATC with live MoST-Sal sustains at least 60 FPS for H.264, H.265, and AV1 under the six existing transformed real-5G fluctuation traces",
        "resolution":[WIDTH,HEIGHT],"fps":FPS,"codecs":list(CODECS),"encoder":{"api":"NVIDIA NVENC","preset":"p4","tuning":"low_latency","rate_control":"CBR dynamic reconfiguration","single_pass":True,"gop":240,"b_frames":0,"lookahead":0,"spatial_aq":False,"temporal_aq":False,"vbv_seconds":.25,"qp_offsets_low_medium_high":[2,0,-2],"map_block_pixels":{"h264":16,"hevc":32,"av1":64},"av1_output_annex_b":False,"av1_repeat_sequence_header":True},
        "saliency":{"model":str(MODEL_ABS),"sha256":MODEL_SHA256,"onnx_input":[1,20,3,144,192],"fresh_inference_per_frame":True,"history_stride_source_frames":8,"temporal_smoothing":.7,"importance_grid":[5,9],"top_high_fraction":.10,"preprocessing":"documented BT.709 limited NV12 -> BGR convention","provider_required":"TensorRT EP first, FP16; CUDA/CPU fallbacks present","stage_schedule":"exact preprocessing for frame t+1 overlaps MoST-Sal/model/map for frame t; predictions are never reused or skipped"},
        "uncapped_benchmark":{"videos":list(VIDEOS.keys()),"measured_frames":1200,"warmup_frames":240,"wall_clock_pacing":False,"threshold":"exact FPS >= 60.0","timer_start":"after final warmup AU completes and before source gate allows any measured-frame read","timer_end":"after final measured AU returned by NVENC"},
        "network":{"mininet":True,"webrtc":True,"gcc":True,"producer_binary_wire_preflight":True,"three_codec_transport_preflight":True,"preconsumer_codec_discovery_prime_frames":{"av1":1,"h264":360,"hevc":360},"consumer_gated_live_source_start":True,"live_pacing_epoch_reset_on_gate_release":True,"media_pts_rebased_on_consumer_start":True,"rtmpc":{"control_interval_ms":100,"ewma_alpha":.35,"bandwidth_safety":.80,"horizon":3,"discount":.95,"fast_down":{"gcc_drop_ratio":.85,"emergency_drop_ratio":.65,"safety":.70,"max_current_ratio_on_sharp_drop":.60,"min_interval_ms":50,"upward_cooldown_ms":1000},"slow_up":{"max_one_action_rung":True,"min_interval_ms":500},"actions_mbps":[5,6,7,8,9,10,12,14,16,20,24,28,32,36]},"measurement_seconds":MEASURE_SECONDS,"warmup_seconds":WARMUP_SECONDS,"sessions":18,"capacity_update_interval_s":1,"link_delay_ms":1},
        "dataset":{"title":"Beyond Throughput: The Next Generation - a 5G Dataset with Channel and Context Metrics","doi":"10.1145/3339825.3394938","application_subset":"file download","trace_transform":"the six transformed capacity CSVs from the previously completed Real5G experiment are embedded exactly; they are scaled/clipped imposed capacities, not unmodified field measurements","traces":trace_summary},
        "content":{"mapping":{"rep1":"basketball","rep2":"rollercoaster","rep3":"ballet"},"prepared_sequence_seconds":30,"loop_during_150s_session":True,"note":"retimed prepared inputs are reused; looping is disclosed and does not change source-frame accounting"},
        "pass_criteria":{"uncapped":"all nine video/codec combinations exact >=60 FPS","producer_network":"9000 consecutive source/completion IDs, final completion by end+1s, first-to-last 10s mean completion-lag growth <=20 ms, p95 completion lag <=500 ms","receiver":"webrtcsrc is constrained to negotiated application/x-rtp; codec-specific RTP depayloading plus parsing reconstructs one complete H.264/H.265 access unit or AV1 temporal unit per handoff; the 146-s interior requires >=8700 complete units, no >1.5-frame PTS gaps, no non-increasing PTS, zero empty buffers, median media rate >=59.99 FPS, and long-window arrival rate >=59 FPS","control":"at least 500 in-window control samples and all 150 capacity updates"},
        "controller_profile_disclosure":"The previously validated RT-MPC operating-point table is reused as a fixed controller policy for all codecs. This integrated experiment does not reinterpret its PSNR/SSIM/LPIPS columns as codec-specific measurements; actual encoded bitrate/timing are logged per codec.",
        "scope_limit":"same-laptop local Mininet namespaces and WebRTC receiver; not a Quest display or motion-to-photon measurement",
        **meta,
    })


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="satc-integrated-selftest-") as td:
        d=Path(td); extract_payload(d)
        required={"producer.py","integrated_sender.py","integrated_receiver.py","nvenc_dynamic.cpp","CMakeLists.txt","trace_data.json","rtmpc_quality_profile.csv","live_model.py"}
        missing=[x for x in required if not (d/x).is_file()]
        if missing: raise RuntimeError(f"embedded payload missing: {missing}")
        traces=json.loads((d/"trace_data.json").read_text())
        if set(traces)!={f"{c}_rep{r}" for c in CONDITIONS for r in (1,2,3)}: raise RuntimeError("trace key mismatch")
        if any(len(v)!=150 for v in traces.values()): raise RuntimeError("every trace must contain exactly 150 samples")
        for py in ("producer.py","integrated_sender.py","integrated_receiver.py","live_model.py","core.py","map_projection.py","shared_frame.py"):
            command([sys.executable,"-m","py_compile",str(d/py)])
        live=(d/"live_model.py").read_text()
        if 'MoST-Sal:' in live and 'file=sys.stderr' not in live:
            raise RuntimeError("live_model diagnostic must be sent to stderr; producer stdout is reserved for NET2 binary records")
        sender=(d/"integrated_sender.py").read_text(); receiver=(d/"integrated_receiver.py").read_text()
        for token in ("coded-picture-structure=(string)frame","alignment=tu,profile=main","bit-depth-luma=(uint)8","av1-input-mode","stream-format=annexb","av1_first32_packets.bin","parser-output"):
            if token not in sender: raise RuntimeError(f"sender stable-caps/AV1 framing/probe fix missing: {token}")
        for token in ("application/x-rtp,media=video,clock-rate=90000,encoding-name=H264 ! rtph264depay",
                      "application/x-rtp,media=video,clock-rate=90000,encoding-name=H265 ! rtph265depay",
                      "application/x-rtp,media=video,clock-rate=90000,encoding-name=AV1 ! rtpav1depay",
                      "alignment=au","alignment=tu"):
            if token not in receiver: raise RuntimeError(f"receiver explicit-RTP-depay pipeline missing: {token}")
        cpp=(d/"nvenc_dynamic.cpp").read_text()
        for token in ("NV_ENC_PRESET_P4_GUID","NV_ENC_PARAMS_RC_CBR","NV_ENC_QP_MAP_DELTA","enc.Reconfigure","FRM2","AU02","SATC_AV1_ANNEXB","outputAnnexBFormat=av1_annexb ? 1 : 0","enableTimingInfo=1","chromaFormatIDC=1","NV_ENC_PIC_FLAG_OUTPUT_SPSPPS","disableSeqHdr=0","DKIF","av1_ivf_wrapper=detected","IVF frame size does not match NVENC packet size","stripping_before_wire=1"):
            if token not in cpp: raise RuntimeError(f"native source missing frozen token: {token}")
    print("PASS: embedded syntax, stdout/binary-wire discipline, H.264 stable-caps fix, NVENC AV1 IVF unwrapping, Annex-B/OBU framing switch, forced AV1 sequence-header request, bounded AV1 packet probe, explicit-RTP-depay receiver, exact traces, native frozen settings, dynamic reconfiguration protocol")
    print("Hardware, Mininet, GStreamer, TensorRT, WebRTC, and FPS are intentionally not exercised by --self-test.")
    return 0



def patch_producer_for_full_workload_cbr(code: Path) -> None:
    """Make the benchmark baseline execute the full MoST-Sal + map path,
    then zero the QP map immediately before NVENC.

    This preserves preprocessing, model inference, temporal history, tile mapping,
    and encoder settings. The only encoding-control difference from ROI is that
    the map submitted to NVENC contains only zero deltas.
    """
    p = code / "producer.py"
    s = p.read_text(encoding="utf-8")
    replacements = [
        (
            "class Inference:\n    def __init__(self,preprocessor,model,codec,stop_evt):",
            "class Inference:\n    def __init__(self,preprocessor,model,codec,stop_evt,uniform_full=False):",
        ),
        (
            "                    qp=qp_from_classes(classes,codec)\n                    t1=time.monotonic_ns()",
            "                    qp=qp_from_classes(classes,codec)\n"
            "                    # Full-workload CBR baseline: compute the exact ROI map first,\n"
            "                    # then replace only the submitted QP deltas with zeros.\n"
            "                    # Thus MoST-Sal + spatial mapping work remains in the timing path.\n"
            "                    if uniform_full:\n"
            "                        qp=np.zeros_like(qp)\n"
            "                    qp_nonzero_count=int(np.count_nonzero(qp))\n"
            "                    t1=time.monotonic_ns()",
        ),
        (
            "                                   'model_plus_map_ms':(t1-t0)/1e6,'map_ready_unix_ns':time.time_ns()})",
            "                                   'model_plus_map_ms':(t1-t0)/1e6,'map_ready_unix_ns':time.time_ns(),\n"
            "                                   'qp_nonzero_count':qp_nonzero_count})",
        ),
        (
            "    ap.add_argument('--startup-prime-frames',type=int,default=0)\n",
            "    ap.add_argument('--startup-prime-frames',type=int,default=0)\n"
            "    ap.add_argument('--uniform-full',action='store_true',help='run full MoST-Sal + map path but submit a zero QP-delta map to NVENC')\n",
        ),
        (
            "    prep=Preprocessor(source,stop_evt); infer=Inference(prep,model,args.codec,stop_evt)\n",
            "    prep=Preprocessor(source,stop_evt); infer=Inference(prep,model,args.codec,stop_evt,args.uniform_full)\n",
        ),
        (
            "fields=['frame_id','due_unix_ns','source_read_start_unix_ns','source_available_unix_ns','preprocess_ms','preprocess_ready_unix_ns','model_call_ms','model_plus_map_ms','map_ready_unix_ns','encode_submit_unix_ns'",
            "fields=['frame_id','due_unix_ns','source_read_start_unix_ns','source_available_unix_ns','preprocess_ms','preprocess_ready_unix_ns','model_call_ms','model_plus_map_ms','map_ready_unix_ns','qp_nonzero_count','encode_submit_unix_ns'",
        ),
        (
            "                'timing_end':'after final measured AU returned by NVENC','frame_drops':0}",
            "                'timing_end':'after final measured AU returned by NVENC','frame_drops':0,\n"
            "                'uniform_full':bool(args.uniform_full),\n"
            "                'qp_mode':'zero_delta_after_full_most_sal_and_map' if args.uniform_full else 'saliency_roi'}",
        ),
    ]
    for old,new in replacements:
        if old not in s:
            raise RuntimeError(f"producer patch anchor missing: {old[:100]!r}")
        s=s.replace(old,new,1)
    p.write_text(s,encoding="utf-8")
    command([sys.executable,"-m","py_compile",str(p)])


def cbr_preflight(root: Path, code: Path) -> dict[str, Any]:
    print("[1/3] Preflight", flush=True)
    required = {
        "SATC Python": SATC_PY_ABS,
        "MoST-Sal model": MODEL_ABS,
        "NVCodecSDK": SDK_ABS,
        "input root": INPUT_ROOT_ABS,
    }
    missing=[f"{k}: {v}" for k,v in required.items() if not v.exists()]
    for name,fn in VIDEOS.items():
        vp=INPUT_ROOT_ABS/fn
        if not vp.is_file(): missing.append(f"video {name}: {vp}")
    if missing:
        raise RuntimeError("Missing required inputs:\n  " + "\n  ".join(missing))
    model_sha=sha256_file(MODEL_ABS)
    if model_sha != MODEL_SHA256:
        raise RuntimeError(f"MoST-Sal model SHA256 mismatch: {model_sha}")
    command(["nvidia-smi"], root/"nvidia_smi.txt")
    command([str(SATC_PY_ABS),"-c",
             "import onnxruntime as o; print('ORT',o.__version__); print('providers',o.get_available_providers())"],
            root/"python_runtime.txt")
    command(["ffmpeg","-version"], root/"ffmpeg_version.txt")
    meta={
        "version":VERSION,
        "scope":"Unpaced full-workload CBR processing benchmark; no WebRTC, Mininet, RT-MPC, receiver, or HMD",
        "resolution":[WIDTH,HEIGHT],
        "videos":VIDEOS,
        "codecs":list(CODECS),
        "target_mbps":[12,35],
        "preset":"p4",
        "model":str(MODEL_ABS),
        "model_sha256":model_sha,
        "benchmark_measured_frames":1200,
        "benchmark_warmup_frames":240,
        "baseline_definition":"execute identical preprocessing + live MoST-Sal inference + spatial map construction, then submit an all-zero QP-delta map to NVENC",
        "retry_policy":"one automatic retry only for execution/integrity failure; an FPS value is never retried merely for being low",
    }
    dump(root/"protocol.json",meta)
    print("  PASS: GPU/runtime/model/videos present and model hash matches.", flush=True)
    return meta


def _audit_zero_map(events_csv: Path, expected_rows: int) -> dict[str, Any]:
    rows=read_csv(events_csv)
    measured=rows[-expected_rows:] if len(rows)>=expected_rows else rows
    nonzero=[]
    for r in measured:
        try: n=int(float(r.get("qp_nonzero_count","-1")))
        except Exception: n=-1
        if n != 0: nonzero.append((r.get("frame_id"),n))
    return {
        "event_rows":len(rows),
        "measured_rows_audited":len(measured),
        "nonzero_qp_rows":len(nonzero),
        "first_nonzero":nonzero[:10],
        "pass":len(measured)==expected_rows and not nonzero,
    }


def run_one_cbr(root: Path, code: Path, encoder: Path, video_name: str, codec: str, mbps: int, attempt: int) -> dict[str, Any]:
    video=INPUT_ROOT_ABS/VIDEOS[video_name]
    tag=f"{video_name}_{codec}_{mbps}Mbps"
    out=root/"runs"/tag/f"attempt_{attempt}"
    out.mkdir(parents=True,exist_ok=True)
    env=base_env(root)
    # No transport is used. Keep AV1 in OBU mode; packet bytes are consumed only
    # by the local producer benchmark and are not passed to RTP/WebRTC.
    if codec=="av1": env["SATC_AV1_ANNEXB"]="0"
    cmd=[str(SATC_PY_ABS),str(code/"producer.py"),
         "--video",str(video),"--model",str(MODEL_ABS),"--encoder",str(encoder),
         "--codec",codec,"--output-dir",str(out),"--initial-bitrate-bps",str(mbps*1_000_000),
         "--benchmark-frames","1200","--benchmark-warmup","240","--uniform-full"]
    t0=time.time()
    p=command(cmd,log=out/"console.log",env=env,check=False,timeout=300)
    elapsed_outer=time.time()-t0
    result={"video":video_name,"codec":codec,"target_mbps":mbps,"attempt":attempt,
            "return_code":p.returncode,"outer_elapsed_seconds":elapsed_outer,
            "status":"FAILED","fps":None,"integrity_pass":False}
    bj=out/"benchmark.json"
    if p.returncode==0 and bj.is_file():
        data=json.loads(bj.read_text())
        audit=_audit_zero_map(out/"producer_events.csv",1200)
        result.update(data)
        result["zero_qp_audit"]=audit
        result["integrity_pass"]=bool(audit["pass"] and data.get("frame_drops")==0 and data.get("uniform_full") is True)
        result["status"]="VALID" if result["integrity_pass"] else "FAILED_AUDIT"
    dump(out/"result.json",result)
    return result


def cbr_matrix(root: Path, code: Path, encoder: Path) -> list[dict[str, Any]]:
    print("\n[3/3] Full-workload CBR matrix: 3 videos x 3 codecs x 2 rates = 18 runs", flush=True)
    final=[]
    for video_name in VIDEOS:
        for codec in CODECS:
            for mbps in (12,35):
                print(f"  {video_name:13s} {codec:4s} {mbps:2d} Mbit/s ...",flush=True)
                r=run_one_cbr(root,code,encoder,video_name,codec,mbps,1)
                # Retry only an execution/integrity failure, never a low FPS result.
                if r["status"]!="VALID":
                    print("    first attempt failed integrity/execution; retrying once after 3 s ...",flush=True)
                    time.sleep(3)
                    r2=run_one_cbr(root,code,encoder,video_name,codec,mbps,2)
                    r2["first_attempt_status"]=r["status"]
                    r=r2
                final.append(r)
                fps="N/A" if r.get("fps") is None else f"{float(r['fps']):.6f}"
                print(f"    {r['status']} | {fps} FPS | zero-QP audit: {r.get('zero_qp_audit',{}).get('pass',False)}",flush=True)

    fields=["video","codec","target_mbps","status","fps","at_least_60","measured_frames","warmup_frames","elapsed_seconds","frame_drops","attempt","integrity_pass","return_code"]
    with (root/"CBR_FULLWORKLOAD_SUMMARY.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(final)

    valid=[r for r in final if r.get("status")=="VALID" and r.get("fps") is not None]
    agg={}
    for codec in CODECS:
        agg[codec]={}
        for mbps in (12,35):
            xs=[float(r["fps"]) for r in valid if r["codec"]==codec and r["target_mbps"]==mbps]
            agg[codec][str(mbps)]={"n":len(xs),"mean_fps":statistics.fmean(xs) if xs else None,
                                  "min_fps":min(xs) if xs else None,"max_fps":max(xs) if xs else None}
    verdict={"planned_runs":18,"valid_runs":len(valid),"all_integrity_valid":len(valid)==18,"aggregate":agg,"rows":final}
    dump(root/"CBR_FULLWORKLOAD_VERDICT.json",verdict)

    lines=[
        "360-SATC FULL-WORKLOAD CBR BASELINE (P4)",
        "",
        "Boundary: unpaced 4096x2048 processing only; no WebRTC, Mininet, RT-MPC, receiver, or HMD.",
        "Each frame executes preprocessing + live MoST-Sal + the spatial mapping path; the computed QP map is then replaced by an all-zero delta map immediately before NVENC.",
        "No FPS limiter. No intentional frame dropping. 240 warmup frames + 1200 measured frames per run.",
        "",
        "video, codec, target, FPS, status",
    ]
    for r in final:
        lines.append(f"{r['video']}, {r['codec']}, {r['target_mbps']} Mbit/s, {r.get('fps')}, {r['status']}")
    lines += ["", "Codec/rate means across the three videos:"]
    for codec in CODECS:
        for mbps in (12,35):
            a=agg[codec][str(mbps)]
            lines.append(f"{codec} {mbps} Mbit/s: n={a['n']}, mean={a['mean_fps']}, min={a['min_fps']}, max={a['max_fps']}")
    (root/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return final


def package_cbr(root: Path, script_path: Path) -> Path:
    upload_dir=Path.home()/"Downloads"/"files_to_upload"
    upload_dir.mkdir(parents=True,exist_ok=True)
    z=upload_dir/f"{root.name}_UPLOAD.zip"
    with zipfile.ZipFile(z,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as f:
        for p in root.rglob("*"):
            if p.is_file(): f.write(p,p.relative_to(root.parent))
        f.write(script_path,Path(root.name)/script_path.name)
    return z


def cbr_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="satc-cbr-selftest-") as td:
        d=Path(td); extract_payload(d); patch_producer_for_full_workload_cbr(d)
        s=(d/"producer.py").read_text()
        required=["--uniform-full","qp=np.zeros_like(qp)","qp_nonzero_count","Inference(prep,model,args.codec,stop_evt,args.uniform_full)"]
        for token in required:
            if token not in s: raise RuntimeError(f"self-test token missing: {token}")
        cpp=(d/"nvenc_dynamic.cpp").read_text()
        for token in ("NV_ENC_PRESET_P4_GUID","NV_ENC_PARAMS_RC_CBR","NV_ENC_QP_MAP_DELTA"):
            if token not in cpp: raise RuntimeError(f"native encoder token missing: {token}")
    print("PASS: full-workload CBR producer patch and frozen P4/CBR/NVENC-QP-map settings are present.")
    print("Hardware/model/video execution is intentionally not performed by --self-test.")
    return 0


def main() -> int:
    ap=argparse.ArgumentParser(description="Full-workload CBR baseline benchmark for 360-SATC Figure 4")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test: return cbr_self_test()
    if os.geteuid()==0:
        print("Run this benchmark as the normal Ubuntu user, not with sudo.",file=sys.stderr)
        return 2

    user,home=real_user_home()
    timestamp=_dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root=home/"Downloads"/f"SATC_CBR_FULLWORKLOAD_P4_{timestamp}"
    root.mkdir(parents=True,exist_ok=False)
    code=root/"code"; code.mkdir(); extract_payload(code); patch_producer_for_full_workload_cbr(code)
    print(f"RESULTS: {root}",flush=True)
    rows=[]; upload=None
    try:
        cbr_preflight(root,code)
        print("\n[2/3] Building frozen NVENC P4 bridge",flush=True)
        encoder=build_encoder(root,code)
        print(f"  PASS: {encoder}",flush=True)
        rows=cbr_matrix(root,code,encoder)
        command(["nvidia-smi"],root/"nvidia_smi_after.txt")
    except KeyboardInterrupt:
        (root/"INTERRUPTED.txt").write_text("Interrupted by user. Completed runs are preserved.\n")
        print("Interrupted; completed runs preserved.",file=sys.stderr)
    except BaseException:
        (root/"ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print(traceback.format_exc(),file=sys.stderr,flush=True)
    finally:
        try:
            upload=package_cbr(root,Path(__file__).resolve())
            print("\nUPLOAD THIS FILE:",upload,flush=True)
        except Exception:
            print("Packaging failed:\n"+traceback.format_exc(),file=sys.stderr)
    ok=(len(rows)==18 and all(r.get("status")=="VALID" for r in rows))
    print("Completed valid CBR runs:",sum(r.get("status")=="VALID" for r in rows),"/ 18",flush=True)
    return 0 if ok else 2

if __name__=="__main__":
    raise SystemExit(main())
