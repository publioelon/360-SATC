#!/usr/bin/env python3
"""Unpaced 360-SATC sender processing benchmark; SDK 13, Linux, existing MoST-Sal.

One fixed p4 campaign: three prepared videos, H.264/HEVC/AV1, 12/35 Mbit/s
CBR per MEDIA second, ROI/uniform. Wall-clock throughput is NOT capped at 60.
No input deadline drops, frame duplication, temporal subsampling, or map reuse.

Scope: source decode/read, causal preprocessing, MoST-Sal, map construction,
host/device transfer, native encoding, and completed bitstream receipt/write.
This does NOT measure WebRTC/Mininet/RT-MPC, receiver FPS, or HMD presentation.
Network trace replay has a different, capacity-constrained measurement boundary.

The script contains only experiment code, not the ONNX weights or NVIDIA SDK.
It uses the user's existing satc virtual environment, model, SDK and videos.
It changes no existing project files, GPU power settings or network settings.

Run: python3 SATC_Uncapped_P4_v1.py
Local, hardware-free unit checks: python3 SATC_Uncapped_P4_v1.py --self-test
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import selectors
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
import zlib

VERSION = "SATC-Uncapped-P4-v1"
W, H, MEDIA_FPS = 4096, 2048, 60
FRAME_BYTES = W * H * 3 // 2
QUEUE_FRAMES = 4
GUARD_FRAMES = 64
INPUT = struct.Struct("<4sQII")
OUTPUT = struct.Struct("<4sQQQI")
CODECS = ("h264", "hevc", "av1")
PREPARED_FOLDER = "SATC_ThreeVideos_60FPS_20260920_205952/prepared_inputs"
MODEL_NAME = "FlowSal_R192_T20_C8_v1_1_Robust1000_static_opset16.onnx"
SCOPE = "Unpaced sender processing; no WebRTC, Mininet, RT-MPC, receiver or HMD"
PAYLOAD = "c-nQlQ<P}S(yd#zZQHhO+qUhRW!rYuEZeqi+qSFD`Y+nPd);%!c*@?|moL#GGqOjff;2D)3IG5A1b})zi!S<g>Z&6&0006W008(us<FM3DV>97x2CQArWn#sjedRFo?J04bE5I5jqVbW>}KcQAC}thLgZl7_6D?43AeQ9VvzUS*Qu*SBGOHDls2e3((9R-PabAzUeAXkbK+I0GZE7gI(J~hrRo{xguf*-E)j8RV>F1?q_(C@SSCVPx-}~mOO8vI-Q&YjnI^<BNbQKfUkX^Dv0`Vg30VuVqU}m*kjf=cRRSfABB)fv5$AHr6RA!K6@Ook#*=myj;tB;=qr&1PMl{!uuMEDHM4T{wI-cZ{n;vhOr({EO~0?DE6|@$tF+L$39H8<^==o-R{0|%G_DAEi8830VLw)hJsllAJ$<`*dfVfRG$yi~DS!R=bZBZqSJN8sU-*1{-hYVYuuzBJJLbxv(7u0d!h}Ddd+-;v-*|uG1in6|H|0gHk|X)<wCDUC4R^c!w&6p(c|#~$x$q#e)XCHip5q__w;(Ee{|0!xrt&}ey+7Rxn%(Yr+P&5*X=al{jTTep^<NS-0^dI*9d29_&M=6PIX9>5nR9vpCsbUu=!6{N)w52hS)PBB#w2w1-9EmUSF^3@RJz)^J&`(qZ`t+eXo-e}eYrLU&Rz_o>%}4q*G_3d;1VrJnVaGvOqO#q=7zwQ>p_u(jJ<?uSLSv#4wp*A=zN{0<@qy_u?y~wBPGX3@?Y@^bzBV|YSKuj^btDRCDy-P4EcD!EU({Bo2o=5us^vFG2Kd$xSjA`Z1)Osmbc3+j(kSYl)Qe|J9{=BOKWwl227=OzxHOI$b9!7xl?&pLaTC{i-ss0^q;RaOjZsTjE^*OC7`vWQ)5pGfl3(K3+cH!k2<~=%GhN!X#jEYO*Rp^<X_>pvbq)T$_{Y~Clq-zBF=WI3CNK^3#IOPvM1|mf;Uas$?Gq6yecyi!Df#^azLaEs1CX@aGn5igxa#Jtmhqtk0%hvPM4@>1MfFVeRJ}SnLweREr9jeHS2ny12h{q0pH0r(*lPsLgbJMdSzs?*zgp9-+*um1NNr~euL&pwzXzD*eWsSSt)s<F&=>up3hLj*<~)yKxy3@y|R)hJ2+P=lf{im?qmF#=O`swu0C>CFASY4M&To*xJS-ZF&+Cz(z@3m(|ZtHgmj$blv)3!H}OG#>h)f=M`Y3^bho4%ce*hz``J71?_cGR$YF?-{rcJ(-EgdvQWj&wo%)Trb6*B8aue)Ckf(3w1krQjfEHAvcABWF3H$a1!3AhQrU-{G9^T1Y+?V;1<-tdEWWWlayaTAv>3QFa#{I-xuc8?9>%Kvw6UE&2BW>iZ>wbSOW3L5i2VawMB}BvoD0)+O@pA<nin9DVTWzmuDG_wGIdIG8IZ1t$qkh)Rz!eoFKX_lhyBeH~tJRtM`qR9`f?;|<nuE@7U?7zoVR2^&PojdcsCn4+j(@kQPj+DS<*}~}UVvE&tUP4AbY6=K;@B<riFJTCgdvhw>cvCc&iAH!?w)C#G!jEB;gx+J-FNc!@X!{;JgdCJv$5F3$dwJrQeZ}Wlzs?=`*@>aU-!`Af{8czR#aN2|1(><#N<~->8)|!Uot?1_nLh~`5+*Oejm|h0^Zag*zGVq>1WA>r(5p~(I0DDo={hcryDa%Z#EjMg$Im7T0GiMWld-}eene}9$UeQCRJ-Z#X)tbeybnR+ye6gUPOGEyw=f)S0N~IlN|h2w=L#)1~_%T(k&bb?pS`BBh&*92et+!e4m4H9mC7!5EG$-E@9;FX;72eM?RYz?^soZ8Z7uNoC4`h!B~Luy2Q@0H?2`HY3yKnY{d#LP)tQlemV#!3PG2n!_E$S5>eU-8~W)+KEqSMfym{O1UM@W(xyh5Uf-aN%CN~$RW^@=%Ben1+p!`>er!dcCbe5L7cJ9MpCr9N+?<3>-+9D|9Vo62ye}7SKARr8Bd$(ZeAr8@*BpVXUr01Fy5s=YmKXjfBPT*pg7r$I5AACd55M{qXLNMn_;EgxX%@7BL3W+K%_d4TbJQJ3_l#}OlyusWR0O>}1h}(cGI9*IFg;yPVzAq|n_n4Q$v&r{f;rQ)LtHIPd<?>UNRy$`+ag=a`q>FxLnkNscx}X`AWx*IAnwm*ps{46tJ9X&q1Y>H19cNk#T|fR7vy+^Pc<cx?R$Gj1-zI`Efl_U@dg@Isdf%G#G2OCP>lZr0kcY=@v-l#N*7lNBsFFn>;tVKJWBg=$xqeL3;w}%jn45MxV>5`o>#aCecHraW0F#*I3Xf)XJw-W0D3A*W=)}Krf~I!Lf}o(h^B*$3udS*T%b`(S;V78fMXK=+LC$9SCtZ$!0gSy`Fc*!G-{#H^;$btM$0MCV!rNoG?^UrRs?3{wymmpYZEDMKqpXTaK=b^9F{&tpj*5{2CaKYra@z1#OmkxT>cM*W6v28NB@f%C_HGf{#qouA{h@esQDd|fT7FWV?3&G2P_r`cryJEuWoAlP@H4AK7;86M?sD82SV6E?E#qVM5|;W0j=e6m#*B}-S`YB$7wS7+uLivS$=$4Z%CNi(gB_j^=V!8<1H3tN6iXg8X<Z%IF*h)-5}eY!SRN)iGm?rCwRLDi1(kEUE8;0J=~}$o34hx$Od<6^IseY+WB0DUb}OI`mDz{xyao9M((>2XkwlecYZKs$97zJ4r%@SPTVE(z>K-eefnqBQvP|LuS(HW$uvRVWxJ>%BuH3&bG>l`=WREP#$g$Hq*U4Op{+J^^LvzB(XdJp8>0OoJXGc&-Oo@FA_(ggh`_>g4*<phUP9iI5{-!29P<H^L*ah{@;gwkv<6EV0yrAFIv{X6!p&t*mgB(BBN*riA^{e`#28bDt@<e9+wa8`^<NPyd^=4d>GIjjeAz8iwX?C>!0+)c1rK@=ybQ>tpEB8IE<szu)vTHvvG4CkjA8d%p{kJ~t(WXm$daFOTnJEXOSp5r0<LKYt5=jz%BM2n0<PBXh>dDH{qGEKAxQ8iWKEK}r{%y1IiYq4hlAXrXV&r$VV5o=Mv3#xXKtZ`EZM=nyQYHdutKYupd>Eq=y-p-UN2k0s+nc5+ukQb4nEIV&XB>3;(5h!7<np0)Gzr`0DNU1Wp`6R!BVJXKjq}sGFxtwsj@@Y>H?DZ5u$kH-xFfcP2cmgsJr3G?d<Ud)!e19kjTKV5(o^$<s_$I-fv^vb_d6ReuToEn{s4(R|N}Z9ybQeM(`CQs3QX`dXl0pDSg*<OX*l`Omh#X7as`QX9GXaJSFDV8>)9LAdo$Q7k4Z$_Si6CISH{At#Na4A<8R#G}kz3^rai28Pv>6aNF=<Po9DRYZyItH;;Rg1X-tx)T377#3fDN!f@(qY$(wxGIBECFLskLJ;i!q4bye4fIEB)Q{J!OP7vtNGYlTHQnJ)61p3cF*P%m_RdaC}*Y~6{?TPM@4x&^Rm3{j8S#66o^Hf8^-_gl%qMc=y8y=m}<u~B8bC_}l)^@F!dT2OzUGZjFRxxZnWlF-`%ivGAtIsK-=8J&!V(c6Ptf3*_^DZmArXL$_%St){<I!#?^qH!?L9|9AjKq4o-olyk$jJWU;jcHI0I&upbQlhR@{(Wj6k1LHWc$}ISSmbP!#i2hy^IIS2|v=YlGW|#=c;dH-85m$QBILO?mKAwUPnk~oJ7WCf>yV}+F;4o#R-u^q#E=E+gjdSeyXA2kz0zumwra`?+3Y_%x0t4Y>2)#$8Fx(r#QXZZ{k1tZjNJz^FdWKf`B#agXQcYAN$m2X<G$A&cquU#7hO0>dh1!`q-YzCU#OHy1;dv{Ro%=%yigfspNHLivUxYUbs3*FUg%@mr1`>zv?UeS5;ohbHX_I&v2d}008zM)yC4zRNvO##MI`W+PqrZ_n+E4>6bsxh<-RgMS4E*iX#@5w~gibIZ`^Q&oT0J1|s~Q>ijL3NV8{64aooXbGj#f5`A6HTL>X)Ydd3oJ!9Q-v16=-24clk6`cq|qiedc(1=I2F;d+zHFfr>5|*A_vWqG1m#T5er}m)QqlHX@v(_RJ@DVW1Oi9cB5%-W(+L@Jo3Tl3j8U)@>)Tg^#3jPiWG5kzx&pZ9DA$pZNt~!zC3u(+;kqSXO7u+-9TRK>z0&t}VvZagaAbt<Uh!!d#l!yY7QE;Y(6Q26}>uUX-pFg{|o#&VI?M=THoj!Iwk(2)9>rA!H4>7bBB<_T|p6_6k`JRf$%MoQdE^eS5;{ASdaPV2?ZusgBRX+!(y70~+1YY0ybiL6$L8Fm{x`ijAH_{+Wln8PKbQenfho94T<j(Ww>phF68<But8Z>E9Bm2a4_=rW#v}yx_>_&j;^L;fcXgCUeO$Ba2v?6xGpG?qq>VgQxN>ei1c3vDHwqP?x0&*kPu$+QqFMgF&evOQ-vTE)93Tc<5O~|GZ8vux|pZX63X$~nPkM?9UDdJysHV<kt2HP-Mi;+*WqK8{pc`F%hAsipQhr&JXX#`H3+Gm*Ftvj7Me{H7MIpfD7;;&aLMng?QGLdCDS%6|Lxk}_=MM>^B>3+J4WRj;YiiQDQ?mQ*OG-19v)Zr7j<Fc!)n71A4T+&=`@p<!l6(Y3Fu8sB=sW`pbq-gap=x(ytqBpYG;9T=r2ghvWI8)bV8$n?)PtD=;=Y{gc#M%%sJqQbHtNHw48G8~4&$fVUa!u8a6J4Byu}VN1O(bOO`A!9ZV}_(l13;oZLSCc3erbh%yF)AFG9`iLwyhD%<n50Wg@9u3Lfosw0O{z1_}I8i=aF>Rv8W7o&pUve-Nv}I#W(@IaQ;h-huUPK_4?r|@{vEVka4SwN6V{I%A49!TEE)|63iGTt&}Im6RqCwPOzFu4FR?uPEf&_zA?^e615E=V_OkemU-2>21#ZqF@H2{c=hLBXNN$7szU1G(0l#fmabHDa(QU(NFpVj9K14#rD$Q1y}XzIYEQMKk2fZHV{0(~6T=Pewe>Akw<gqzUC)^nM=W)q;$0?I8MiAdFx;Q3fLh7K?L-4fVx?2gd{G)C&&2B%4i4Knz1B<3=uhD6Ba<U_A5?P3=A`JY2VCv`r3mY_0qlL|^u>k)OVPC81`}$)B>FXVMU#gT@m58fb47-%MdvUq!49fvY6^;jE+w<898KWA)<iJgQN0U#;pvZ2@G{$GPhx=}`;cda+*F90D<=GA$kVpV6mcvVsUH?uNn-b@Cgy^v2^uCgh!#E_f-K-tUxrGQpD%Nk2*^lECrY>tY*xs5<Po1PsAP!12Q1RfRTef_vuO0K&nv)!?oS~m$ruJQv?B#pdbgy#gVMt0jZT*N%|p-A9OGi|;h5->lrI7f&q{2pf61=?w1)=1(6*^81a$cB0WaRbYUDQ~EcHt}jvp>;?fe0{E_wG-0VM6D^<`w;oOMd*kgInhD+auDV>UT=(vy%5X{6W+wpT~wH?_ux(>j_lk0O2{F5=&NpKv78MQ*;o<OxjLLyDgXW&07^5@pokN*C)6%HB$Y+0n!-W`7KUrhOH#R<Y%2N!VD#LiXU<BcYQt5=OPdtHXk$vDR3QuW<n8M??2f6Dgb<IWkX=R=EQT7HxLIHcW&=Z|oeNj}|CB(%<TzzgvIDH3(z;s+RsezCuOufUXf)3@lr{=&AD|^h=sgdBE%PeFV245M_^wGt~GwUBq8;Su2C((U`%auS{kl+1|iFSJAB9Q|ng~p^^8Xnn{LDKu|iqPQ9oqMh*=t95IWk5uL|#^k|Ha<0d^0Hc_qJPovDm9+h;)C$CzMQ%;lGt1#!Sfoth`=QFqhI6ZY88vZc3bfv5~9(+TEfK~ThwGzf2B=a2>L!S?fA|Ae*hY7?=O3N~jU}=}meBJDve2m(=w!fXx*S}g8Z5M{N@_qaTauXlw;MNFwzX$OqU5zvG{*fwr#vlais8o*I-@Rg<*3Di^DgkOyJ~>ogYIfKCn>65x;pbTU=a-!ARbG!^<UOJ8Ivma4@8Lng%3W4oe3abz-G$lD1ZylRO`MzsvXwJhxjlQvn`02f4(SKYAT_ErKl2UTeAOr}8NC?+PB86UBPd3QwB+%ZW0hMixkk}K233DmvzlJT;EvkDA;Y}H0X;GIJ=i7@hIU<hYoS+JC*KAhvw|SMk`iJ|iNso76c#W;>AcSSC@NlCp9WV?J>zh^%>~pfibacyYbHC?({^(**J>fG@6)4zW3H=|SBP_Gd2<prV}C(cro%efaG1&VMH^}};wV_ld7tVDXlJy0dNL+LrYIeyN982)$^rIrk{F{>V}tyy$Q;FKmC5c;0I0e3o1V$KipnMy(<b?MKsP8xb14M-9X|A2MT`$6WB2>h<YiFZ&+#_*+M#McG*<Zdsx(r>E>J0v**T;>@k)0@6irGj=@)wkqXXZ<X27!uM$!fC9#YlgWnj<e_hF)mYep@FZY*pH>WsDZ$X8u#+pl7eGqEFbm=Wh?PBAH`h8eGG**#aFz`#!8sji#(!aSF3$KW|#aF<cb6X>1m!meIwse1^QvlHyyL@GW9dRGM$dM)qC-UnGIsA6KWuQ&j+kN1`><2^PWxlanjoZw=OC@RY>P?6M2Bu!cb1@ABM7bs7UI(^upPqt1E<e4S5e~KEB#1L$-TJ>*A=&p8=z2%(0zawULulM)k=HuS;W-D-h9TzD096@!fV962!0J<@<C}TE<JsBJK&}QW<oA-1$k3b@-I2iIwiNs_TMa3gG;!5ywQ0`fL-pXB#HxLz8Q!JKxpfYNT6dcE3`_GFs3VMx78o_5n7mXF;xr{Z%tEXJWVF!HtNxgkg)S#kJ+}i>tX(RQy=IP_1Kyedr7irZX_Gq~@Lkf(gF5yKoR`B@SftDB9vvi!C*dGM81iiuUMa)FHl>|r=K~}tXBYYK)lYUz@A4<bBRW`uNlg6|#`nVJFsG?8FQ5bHM0WmmcRmwnKa4xfWali!D%0;zl84+qE)I;AjRqVF77S1Lb2kKmYLB7}g{KaeB6TZ<KdYo-uJI~YUFnQsoJns|L44~?@-VBkV76iKeIy<4b`7Z(L3#DIM`&*}VpT??`NRm|C;03c7O)8ukEeFs%*!z<=Iz0Z>*V~=%#V#Z31d?bQ?(HG~4=_Q+b`Yq|sC2pb?E$SrC>n>p7t0N`khAip9;(TO1t~Q1UON71=6n|%9%za5Ko`gL1F{E`9WcaI*jr>tI6+EZ$`7Uy8&*-W^$%e$)mIcGkT$F7;1{+sZe3pER@;_)D>!|W(6I>`{~I~hcGGE2b+3mca)%+Q3YK-a<M%OaBUs^iZi?;%MW9o(0DsulK)zU=2Nfjr;^YZ^Etq_CVDy1ZOGVynDug|{ee06$*N%lC#?HG#Ix}%{@4-^D`<ywYI^r+e>M`ETZZ0x6+#Hx}Dpk;dr-Y4o?t?&ErgJ&!R@vhhT`qu$%q*R3tR<F#=pg{gmOCGB^;7sOS9aYQ2id-yj9!hpg_`BJbO;dT3~IGkb{KcU$0#%w5O$oO8oFASNc89-f&{%6F1Y{%MmfwyQUK~J_ttf4GzBVW$us=S<)hbC*TgpOJ`WqNG)n;LY5qUpVhFe5n6jjQ=#of_!8LfGv`}%<KJP!ALS4U@<jvYIF8CU5%X71=DQ8^u;6X+P<>0!-kHo-%7WTPwIFks0svSNVrhfUr26GTmKH=6kR(mS_uO8VUSw`7~&h*MXalD9DbYC8r=3aV?`3;W5G`?7HKr`CN2{ZiJkv0>k+%4Mz^u^ngcA`-o;R76Q{bCvg;;u9aek-Wgu60p!BbB={Rx>Y0j|(auMcQ<a72S7+F1uqqhJ$cow~h!ml0)0RFVAZg&68Fu^uqWqpD6;E9Ie$+whD?EG1bt{oF}7fbKH%Er6$3TMkj|Fx$~a?sT}<t=(dQWC-*G&zXtP>aa)}V<8Tdoq}iEu-ggReFJQOa)u|^k*X%(sv#69PR<ffG-QLmE(HJAGDsa<dpJrWo=I)nqQzi-v*sPa%c}6GH4Df~l1oIy+1qql@Hbh7NKDc@VOo1SyVAvS@^<*5rz~mNH!UR+?0R##+L+Q1o@MWY}QKOO|E;<Eq?^I|$8B`2X2-6KJi<3#HeghYzVUh+6S<Kd63o@l5!;H`Y9XtF5L)o~ASe3KXqx0I&CjPW(Ac1oiiBbKv&*a7+LL4cw)tPsr-}x=YNo2MLlVk)fbdZx(mJVH%q6i3q#4LBtgICjnPOmQOAz53;qXiP7VsWWxWlu?4T{1D&7I*M^G=i~hRgmsmxyOB3Nsc?i;?W>=brfhntB4%}<Se2ArsiNsZ5X7WKE7A}oUt95pcQ8@m-)u51UcUB8*|DsT^?DAD{6z;KY?8S9*X2bL|IHK){XirZ%Z<uG4TZ}fAE(ujp6(*<>Hh{@vo2ghA7y&l$*I*lAPc)FA|}ue?N@_SK$PVZN|Oz1DB(d!*}VLZ|WF9y=_&X42fa&!$T?6%%H-Ns&bXG)pFUchK=r{4clHI+C=qHg~JAqoQz_#FLS!1801OIkNxV;EZDHwIteL-8`bMv>1FboMCD2FRIq9vA#}5{!Y~*1{TkIxc=V`W$W1aMcoG3mt_>Ok(vlnOUpfvAIXTYAf@}{i(@D}(Rb`#LH#kDUEF`e|7*?W3GNnK#25v7*f_Ru8=5aejCNTC#1H3Os#~$P~(56_n-`>4l@pQWA{tb_ul?dienDyAdRoy0NsO~i51xeSg^3j`uR49*y3DAI|W??FY-GjUn$&MsNmC=#qiAr?AN@O3ADX`AApV)CktXtO@T#e8lYS1|>cuRcb?!vb08`;AlDvu&ol|U&&)&)#w3dh!T#vI1Lbsj0p9=++ZtnEyc`8@qUFXMj$C|&fbwb2oyfFo4{IhgsjX)QZNu*wXT#4Bz^gi5tmGpm^E78qqp+|0vc1Q3P=&Anh-Hb`t%{)m^p$e5B!SF{2$+kQn+{QXO3zm3;DAqp|p1fty`xfJEL;`=y#O;WnNTzl)3=OvvJ;jb+qx=*~TO4of)9Bf@;?E_f>qU64s$V2JuPm|N3B`?qQE>po%lb0njM)|o?+<n74T~ulJ7Ytm-U4yS|Cb47d%%ia!LTR<+ft^<Usy}&@ApT{Q(a3@-olQZtR$f4qwc`5Bsi@{*Eqrh6@2aEK$nDrRj{Ou909Z+%Tz$DR%f)+Ig{w8EMWJgz87Ot4@1JX=GrYNNI~|}a*8w7I*>&ldEGQlA*h{XOUF?;`Q}D`RX+VaU8ZJEJMSqM%BX`p~Cy?YSZ6Q~Kfk6=k=jsN+HOm~8B_nEPt~(^v=uwr_LHs|3Q9kRlP&(6H=UYW4rYCMfBjX0XW!jR{DS%S(L;As2yBLsj<RlC+$+R&?<3R>D7CfT)rCvKn0@8cBk7f(R%W!`>5(|jUTCvN1!<rNJYsfKfN4C8GAX8)0@C7gCz&Hsxy%gOy+i@^Vd--nLsOoTOY1KDNTJ*%L)-~&tJmN=Ox-@79#joECU)`hJzp~6{S|d#GEXY@qd=0pSRZ7dgXo$HAuEuYH3vIL|ubY>X9eKBGUuv{$X!cR`<&&IgZVp=!rdH=RqzhEnsD<>LE0^%KjcOIV9yRACDtB>}sA(=ETAT431m+s}@$lxF0Y2ll@p`k^2)snsE<O=nR<jSV7%fD*;^~N%e!J_R*1wObWj3J&^^0e|dy7P~ykzMd<hJ<Pl8lslRh2C_c>0$t?*2R_l|(x7aViHe3b}Yv6Bl9|jZ4b`zh9z0D1^E`10MDFyL>i?uWDo{%Fga|y%vFFJXw<50I9OMeR8MSDt|(Fu#AfFJD-t#?8W%jPN(RdC@bmv>jQp9H0HiuZ(xom*rw~0G{+@0m@;pp<X<`C?7Zv%hrMw;`WgJviqPRA=Mc{&(5z!2KV%I76>$SCR2BM6Kqq$_CBk&ko))hM_Q)NpeVCVdW#e(R`@UKx4d6MZV{4HSlf2w~^J%7n*P3HxQJPfNJYA;j9`>5>w+Qny{>JAn%a3R3R@7y8X@66*tyWn${$yBpQdXa&SV?5l@wxlcJ2SOkeEm!>`}3Nq@@_`oA66~nNduqEW_xiZ^HmtAoU7$G6i%Qgw2zX3ScGY-`iMMN9}~=L$x!m9xEIBi=U`zTy@%dO5NsHbmwpKl({u~^nS;G=`(7rT7iU12=-{z{xaF`+67cx{eDSiN<3TFlzn4cslib3Yka!y$Sj3r14%tcJKDG9SFvD&<DPFaByTC3aXuY9E?3t%PznHH$X3}gqrJwX(po{T&2W9sX9rytW+tA1WcXbN_I|tFNFAMWq+32D<DLxhIy~aisko(#y7DDjZJeH!0cuos5@rX@R^9y}^;&NB$EZ5jiumZvC#0z3Qzij3;OSr-qym=f@?Rc;qjU#7zsK{xR`h}xWM-97>wLWss-oWb@qao*?pwkmXI^~LNR=8vV-%dgR<y)@8X{AtJx|0T=)4YBsT5m-wov3Ti9TOnW3EgDe3fvfRwtz$LHQ5*0R-`<_bJt(t${4HBN+8QaC)!?8D0h63!3L@UTdf1bMsl61{M_S+I@(Wdj7a?E*(j<F)J8iKRDy?^oVui#Mr`uLH6u`cD++Z)R%(wW!WziS7dK}lh$30X=}s4T@&aN8#L6nWju&LRFK|G*^+G|ie1Qb7l|}++3-b<yqfmBZRSML~#<AT5U+KA3>EzeTgKrDT1?}xe=X2@R%7Wj_f(iAeC=u4AE7lq2(0+nvMV;5bf9niRl3RBuU5B38CHV|`Pqd_>WY&SdaGlLuU{CHm!zPu}oHYId|F_#InkqFl4-5bR4G92%_>XF9=%DZ5WN&3^>|$weN9W*KqGoG-D2DJ;qtDoePgR1=E|)XchT)YaJ<*>Kw6}w9JS2g&xhc_U<enn&ddW>a9&xo{n@@(|d3v3(ZjR^4g0+27xRh!@Ydz5ypRy|{kf<&u5G&U~MJ$jKki^N)lhhxhf;~sd&uM$b0fDA%8SuN6+GivfFD+`3--lFuP@E2ob>0HXnE{n=K1J=Ka~lHhdJOQWQ>G(p3zTQ#<>;)duP5Vw>>ys6FVnz12E@)1`l-`eqhQ4r#D<9}(fXIo#2>hV(KY_W!Zu!!s(NWcNKy0j9#hL&I3599;bREFAxNn{f!0Qqa+u8&9a;;}yaJ=M9Ho^LD<-gc9(3rw!zrbJ!%U=v4Mv~x8UtA!>VsGn)fCu5Fhw_6{^MenI6_~-u4N|Dt%1=rnhyfw0ZXdrw{WQ}3fe`i)MOAnqy2gdUaYSrCi@fcp7cDejakl)uCun=mnlcA5#VhxO*W5yvzB$&L6`ww<>uj5NxXU!-XZK94XNqJ>-(yJpeY3(?!m!|Z^imeE3xDIrD$8Wvb#a)^qPmq^}|HcU9oiH9qzYpLM!5e7&%6`_JX?3NqO%JD0hrvf;W#Q#e%``mrYcO+2qD>tDZ&WI2{X0ADn=|Sys}SPv1N8%Lx;5jrHXMHG*w@-yHsVOa1+;uFX5T*T=8z^A6)07^4tADYH@sGTosWdcL*&PN3^@v^Fk}W4f&K6W-rX^T{(aDaPfIe`kDdXLs@Utb}tzdeD^&sAgwdde%sejl^~B5WDC(COAlJ<8Ss#tnk|!mrULOsk+_)jV3Obj(kxgUs-Duc(ny;Ohy>h6G*nch`2aF*U2BE`;SN6R72s(^>-7}w_lT%NHyK$`-oQul4mC}IRH$Rob10|8Z9@IE54eH(QHFhg+7iZ<BD>2RJxihU|DnU;kSE#r(LKwEQbqhLZq^QoJ&<l4iup{2ywcZ>i*JdI7kc6CPMdhOg#FnYO**{Nw^M>@#v^Y)u!hxSO-zomZl98V;O5g!b!&Z0)r#OU0?P-eCmSu1J{i@@Y$_4o4#e8u6MpqepyJag{XyGS9-``2YG7uo2fi3eZOC~zIl1!UkA4qyz{HLQ*><8eoU7Yy2y?x*b^}t7}gbKAQCduKBy4;ZN9BChKRTA>^{-apc>L<5*vc+;0AZ^YhfkSoD|r5XM~%%?(pqjDid8S%YSQjNBY4H9^n)<)^XMr@h-TZd%=}`VF&Bum0@&wgTI7-S%tfSpKwnyICpa0aeuMzdZZ(Ax0DC=0{(aY(-kdfj{pJy#Qd`c!2hE<TNpZ-n&_K38QT8OT_jtjX}e7Zn4U{Y4*f)c8hio)IL!ki&=oT2M!P-+j7%9;Ae-_0BQm^~yQHe2^0a#b1^n^2UdcoomcZ*Kj`;Zt_Edo{5AujryqaiW@4ch>A}4SM@ZhqvFH6vt@8x(z3)_*koxvaV79hwe?1hA6;mhrm7!omcT!-^6UWa_N(kHG0YspPKYlz)sOWH%CIibpw$-PN5sT&RlT_KtaLt9}SbuOixWLK$PF>A;es}G}W#L@|kL=u_h7(#H@>8vO_8e<uity9mN&rbMxXD~gfWCfIoGWy8!VX0DsZv(S=ERsK(eQ5ibgAF<uS<4!e4G6(o`z}W9$nJw0<mK)jUHa&ivsqu>J(;i|u^rZl1UrgcR?QGB<{wS;W=AX9K2pNlL@pU2=0|xGk-ZxhK00<$r}C4!J#ykSziA8;7_Hm3FT-^DHNQH&yKeBGUs^D^AZ&1aPEnq<R7Lh&NGbKg?1y6E)KNBTfef<h5=_sYuzta}E%_uWf6D;B>M+!pdP#%P^_tU>F*D_Qtaw-{ba<W~WkUqwL+M8S+Gia)lh!@@9cTDY_|f@(bgPBcfAOSXI9h#4J9CEq)!^!`r>%AVC*-R@0040R3Av@6gR2XjMT(-X^#B8k?`>^3)I#jcdzBi+8l$1HQV4HqdK~LqDQSC-^M&7M;+c*sg9hHhPLlU5=j)DM^__KuA}Qpfmr!D10gUXE+HMMIYLJgth*h6M2$1P*@bVm3gdpXi@+>jv3Kp$%6-Lfv2Vm3Vg_$k{(?dL`eM&xixKUq%O=KY$t6LiyJ`M0|M5y>*(F176@WnfB+iuX(pake|V*{7Da|;DnMxBw)cDJzmcTtBU=cv!ns#?M`5#grYtg3*}sRmb~PZ|lax=_hsPq4)#A&1g>a3@t;8A99`6=K0;8e=KdiCe2vY|5_oW2m&Lc9D`fQm{n#Abf5+7_+-(|4h-d<N?u-u89Sb-0SUKX)b-zedVsPRCveCDIsG&3>~-QzeHKi__O#)85yI;$-l;HT0%Hcfr`fHxY-N$KQ{Kmkq|dcz$$DDZbm_KZa#E2>hV%m7R28Kg;fT9e%D=b6`7PUDUm-k%$MIl{~aFbjlT_bAOHYY|Ln-9|EPwpCYCPx_6{zVww7Lo|Ie<Rq9$v9$cE7Usa|&zprodVG@5l%#09cRqm_SMx^6^oNierCb`na|OoBQ}`?1SSER}i)tHmN<5P>z*<;nawcJA^~P?5aUw<k@V7Yw8ez>FG1UWaXlMQ{C1o?N9c&|b7_r%8)-0{ckKqD<P&X0Ukf!c|^Xu1WkQH4d8OYFuUB`o?vh2mQ*z<L4#@Kig5A1&bVePS!>d9b)K@uBMh1Oq)!J_jrS=!=Um2`LL0k59VxZLyH15gZpDZmxTieX6-YEZOtdVUot-k2aT#B#pRB&Dod7xL`Alt^s&{#HSqi9G;jx>$23u<eWA9pP>a!x=U5KT&pU*&c<g9}HJ1>|>X_}An;$`0qG#6iyu8mDNBKwvlFv_ymakJ<{D|l!;U6E;3#=FzQ^6t9Y?OmMt%;t#a50BB{}Jz|l{4dE4m%5L24fmp5)=6Pwr9e}&(WTaJH?HxN$A)EKOMa-CXs{A7kN+aE^)!WzGpLJ9pch=qZJba#J+-sea~HQ1{~i<k`%j)L|vYCq)!CtTGS^zAuJd0CBt6MR2cojL~O12)+17G$Hs#wg(*TAB@VjoU4Szr%+#Hs9RbED&K*H+pA_}B+usXapPmL4SjHGq>qx8&(h-U^F9vK6$!a=6D=*b+`G<yTYBFyo5pGFZMk5*|YF-`PD&gCTMaOWE?^Gba;qWQ)tTKdNZEWFN*jy7aDT38{Xc%A=qUw&J4YzR%PBi04GPj$LIQUzSZ-|(E(7@!A7%2u${1K6cPjs`jc#@ka(gD+dXh2juEUce!IOcF3U9qmYAn*WaUcQN83RdG|8?dHZQXam=?eL~YJWOy33^u}H5(C6~3u-w9Y}s&L?TlQ@hKCJ~wrY=4$m-Vfmel{#C*hFDKOA{$QR;g35T#$4m)NSZ`qVF*LeqMfOxzTeY9lNK1c2N1!eUE}lhkn8G&le#ej*afNs}m$ai#Uj9pyj`ow^{6VlhVw2~>z%iL<KevT0-gB)`<g(v(wzgZ7r=lQAb}8WZZn(7~i)BEX&U69f%&%Q3p58LZ^2H7gkjN)lh^1D~gB?!Hzt17AOzwDIuWGrOw^#^u^)*BGv-TTB%k@YI;(<-tF}CLlKAiaz#am>6gqs{1glQyB+~l%4H&;mA1HfTd@29D1oM5pCJ^vwJGX(pC_`;j-=e$I0z0-XW7ZJ3pHOei35?Mz$6Cb>TPGv~`|B1yW>|UViV+W;Xq(XSnU_<>OGC55+;GKbUS*@|TYo-H_n@^D@ux#QP>1HDo}dtNxBe_Hqr}<a3)`tyv29;0j~e&76_1Fmn!bcO!UvYum_v{CHx~QIN2nr>;qeadm}GaG&_P%x)ArGDW)lesP+9HOKbMag*xEkG~s8-Z(f&92`41I~yLfgzugj1?!6a;y4&Yzu85nvAOQqa8rj-69^9-R4lo}vb#u93!@91WyG^fKOf(`s<V(%D5Qol@Gp<XNf)SM5&e%lI5m#bH>@-Q{C<=4xl=HESaS-V6Z|Y+VM^1GH<Bg!voR5C|Crf5_D`7KvD*A=jeYfvw#Nhfe=l)6^K}@}&;S4(_y7P%|EP9urgp~qu6D+T4i2U!bjA)2C79p#8)FZzU#Qx-cueKef1G@~5l$oiEao3x_7z`LQcbQ|YYcnfddntSzwho|)xfqeo~xQuMrP#v&p-P-42rdTd5D+!`YSN~YJrJl9&%|Rqj+S9BSPgQY#pR%us)<;KGvUlh=@;2UhI<+Y9Xm%!59!Y__015LXrV!gJb`Ps6n`42odZ2TM(TQ=t?L5a(jv^oZ(18<B?L>23Ki>C)#DV6t3r3=feD#sONeZPE6hHO5^$26%C0o?ClCdBnlyp209;MX*>`D6r+Vr69eJE92=Ylo+i<*e%c~E#O5wElwK2Jm5bi{3VAOtNOfcrkHStrx^)3$f4`bMxz;2eJ=$YlREWs$Xtk&r)o7H&4}rPI35_ud5qsF69^N*IkYCB_>=QFEED9ZTP3-EkCdejk1O5bx8BWb>Q1TYunFUimdfa19SutgqE0Ai42JV|zIAnQ!CAv<R+n#N~oDru7)d3%k%e+%1l9m*X!6RBSJcYCbPpe|A)Nd#rHZZS9wh8DQ!7<^%OcQzLKDf^)EzR)CxnBk+ltqO@5B&NaO3fiY_IxC_?9O&ZF~2bUE2Mp<VyJJ4^*rcAOD|ZNn1n*aTH-l(1teRMP~9&wp&v-ACyJ<*)t1^wUFgo5^lvw*mGbj0qUxKu5$z9zEndTw62*KIScMvL|63MxZKR2Uv@Gq>^-;JyR_VZAE3D!f!9H-`U}=H!cz7c>#EY7D{GINLMF1!r9#)-_9k}o}PJ0DU-O^ZjjPm!wp=+MdKpm1p(O)~+MW5ydk{;9^0xV{_OaXxFzCEG7ZOp-}KH&XYo7qSr6P+SuCJc*7+>juh<_=V)KgTVs`)IX~5e%R5hwj6Fq40#xY<;D1WKK<vwR7*0_AJARuNt*ywJfLQ&`wiBv7;<C(f=qsVIWX)CT}!9Xw%9=&2Tn;p`zJf<Nn!PDso&yKW8!s{wTqguCkP1Q<HFTv3dl;f|%Qa1~}h{(~+Rut4W=9fQSfRBiGCkn5|FVjG-BenXQc?aYQ=UUdKFwYpM^gy+fzgS+58$bpGk#NA_Hj-;>U~oCElbL<XPOI+I5W9Y-Hc%GC1ID(#^ih%PF^=SRqI;i%!uB5kwEeW<99s~wtx;dw+TEXQ^}XQ{4MI~dd2mp-zux6k4k2?Pkd`s{cv_%@DdeSx>$+$zWlV7$cY@MO*@de?O?eVyL)WBV|<3m;1n1-+NCx7K;AowgShvCE&;>b(C1f0nam-N0964{w5xA;Oa%z12ntlQdjO#I?Ah40OHJ77JJ5Y;RWEmr`k4e*sXS00%gxG64SVFO%;H#<PC5WIQWUmBFE%ee0dGPoHy0@0QfR&2R@`0lhJ$D>u#SA`k`p@TE75*FW1~2kn}Ql-Apg1+n%6z#E?4X$5#I&^4;{*h4O5A;@}0`>x^CfAae9sZl!;B1<bt?hXlR9SU86wD(N8rWGIx%`4ceHwQ?znMVi7V^FRTC{DF)5vN?g@B2HN>#b0O!PZJ!0`*xAi^`FDw`rVuH>qB&V;x9<f->{k=v77r?%3GYy-avQ)~LA3F#0-;W)V-X2YF(^!vMCuzvth<c|Frn#EFA!xGP_s`TAj_KJk|wa~F<}a8uWJR#1xK4bGK@<xFvxl)4(q#1luOM%wzR-g+2Mnk83=Y)eEc<x!zMt&B?E{6mL*s@%ouxmZ4p2NK+v5igoQV5Sw>nd99CjxZ$JTi=Rwz<kpN)=97o;KrE08XL5vx+JWGYWr-s)$@D2N+!lH0ggslf8HSc8&%u7nhn7re$5I%4#$qxUP}yRV?oU|H==Kbrg#V9Wmzd`95|<6u%E*rb8F)OA@K&*P=}dT(l&=@VZ@7$DEp*M*UsJu{bB>>BLwBQaGP`BU|)u&Hjej_G{`;k5ez3preISd1&20t1VANccKXL)ryKL)TYc=9(B*dtE2jYz7z+zaD_7tRGmWqWeg*4eiP%i9V<GI_?3=2x*FnM@LrWY-ikJnu3oj@E)kOYzww&`qZn}q$r2`Z^cp_W6YgmtCn-`)S`aLX6aY7X(>fB~vlI;#Z8XA(%w<>*}s)2Qe4Ax;yQOq;vmP$X4^136ebxoQb=i>E4!R}KV+NB1WJX`2j7#9(U^oS6SKzpj}@!W-ON+k`}&b?ol%9h=haOm5J;3|yJr%=e}79muO-=0=4!Eacri|NPNy1%BZI;C|~{+a*4p$^>Uk^SnvHRW{q7ylA2W~R0>7XLQ8PDWjurH}t|D@r4Sj(hehE%PnujC|c*Bx|Q=%{4;S0E%<`9pjNixgay$EArjUz3__cSxrqV{d8^^RO?L<M*!+5$ad(K;!8Eyh8pa*E)f%5Z3w1{2>*cd$hfh!=_<^NZ+!em8?L(G*@R9XgC8AGfp2x@c4k+<=^=1txMV_LYVjdZRC3drsPpo5*XklyvmfT*f{>smu7EyP+{0O%m~D`n14cpVp^_V|5e{c*X8`BCsB?TP{X7rM8JMf$+{EmTO37mP!5gyPh5@*vLkKlPl<G^j3yJ{hEqJ^sKStv!2$waEFT+Rc<r>|OpByqV)4ebQF`8|SFqQH&;pjtR2g8c*nK96dz}s#-uD55z7FLb%7&o(6p?EqUzjmw@%H=o(01HU(h=3X>b`JeBR^159Q`jA}`&T?kw`f#A5>E5|OHJajMza%vzS&u{oq7POtgTmIa2AkZL~cIoT@@scZ+W8X7c{hl&7<bh%drV<4~W|jy~2g7q#TzAsPquEzPF_pS3DWR8QxtNNYS?PYoSz4SX#rAgKR@uK6}Pc?dWI3wvZ3(;FnRCOQtIzsGeQ0LHpQ^4GF6vr6fg<C89)15{hF?D<t=wIGq%DJ!)=d5IKV5tyqD<-0Ix&FDj*|$RR>$wIIJx76y=Sgy#@*L!>26iZppfNub3BAJn_A4pCSsFPQ63;T?5Y{J<LBtF|FjGqsPWt|P$-Al#5Q8|m8sxreLJ=HzI83%YHk<*k+v#_q$V(0lK2*u9D;1Zz4yuUU#@J^u%3F^>M>YcHv#mWgQ3E2k4{Zlqxw&R{xKVs2Y9J@KS$P+A%<Ru_f}Tv=i$eAqh7tETEg1c!m)w68nu6o;wd`x*!Ypy7*_Oga;#oSA+!z4G?axMIk2$=T^~3ihvUt#C&@Sq-!#9oP7B96N6S$1@E+QcasNZ>CB}?D!Ou#3|a?@yjc*#dzH~eqZ$<ev~jz7(#PKYC-T-&l%<k?cC*Y@c<XtIzW9hBnw`*w*!ADi0s*XTiOC2nC}hpT#2>!?<I=b5JvI|@G<Kt%Ifk=Z^11fIgPaTc}qh{CQ_EZp_rzteb~1Amr;Hh{72&<?t-pJxfgM5wiPLjKhAwNP4xKygDJSzl!rUW_zL*`Ao6@2R9^SVZWCUgdo1RC+n7(O5|T(Wlu!?vr$WBmkS^8%WZBSH=E_m~sVva<#v57K4VO&76==iQICW?wi)uK|LHMloI4Zec=+CUTU>Dt{3$jOgS<Dfh-W{DZZdjFqY;+;2=tqWBgqV8u(IkSrc04*U71^>1P^=1BUqPuGdQ5TYjp5Mv57Ig<G(b`Bh~w=OC#SV0+_+R?`=dwM^UR751Zs1lRz{HY&FaGG^(ISB?+R(}O*>2*zP{Oc4{^8J{-tS*#7#au=*m70bWd6}H(W2~pWg@P`r&xK40P8+q$kHAp^rqimtk*O_UmV>jLpenMZG_rIIR=n?&)3LBKLT$Lo?vqk)JanBzgrNMPS#~!AWAYu$jgEoBi2<q!i7GHk`xgMiM=$!Opvo&fl$H6E(ADNtiA*IKh|l)=`vYUaLxsndjsk+v~soG7M7!G>9?*0RR|*|8Mt5Sk}<mRL0WT#hK2<!zCwiR(_KKW%Tx)I=2uvaWIk!E~_nLNEngSR8Gs4d~&UJZ`qM&v^<pT$9=29Vy--?5B*c~cJ%b~DJ5r(;I?Ubd(APfSL|BRRC*%QN^kJp<bzc{`_{h@CS$uW#ivP$BMi!{xJN6VdPAUI>v;SRA+_BY>oMbztn@`<9a5?OgGqJuf!tV3+h~<Wa`Cly<2)e7ch6r8+DDPvMz0+pT~*mlfsb=}WxYo?zfYHy@EJ^;nG4%_>g4-UwUM;nPbx7|vW0dQd4gQHFhllpY|HUz<&hJZ1!4ZP5yC&0&9Dew5bzKiaDo`fsI<)kcrV5TCxcHoDqCSN-Tu+t^A+J4<U*9Oy^67dTbMk0lba4<&0$_;#ogI)C_NjZ2tRnht7@5wg#HPIjvQt^rpt*<PfJ}-{s?81=z>{A8xvqp)`X0_Z3AcA^k2p+@hyALi7TLas%g=6_zFo`XSAKmQc<&Q>fnsu)1xM5<csQTToP;>c5v15l_5FE^mHaV80vcV^TH(GVTX`99(NQd6iMW>3etc;D8T>c$EW}A_XiN*KTlEr4gLFW{C}WS=zram|2O*YM$>=LV1oZm+v(r<zkTZe4^Hy`bFKdk|J(8W5A06%Z~yb(?7!`}|FCEj{|9sK-&6eixci?e-f8}Khu?qm|4zC8@XZ|mPCx}|P%wc1i-Z6;{fF`9`k&GN0<)huF#"


def dump(path, obj):
    """Atomic checkpoint, with strict finite JSON values."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    if not values:
        return None
    v = sorted(values)
    p = (len(v) - 1) * .95
    lo = int(p)
    hi = min(lo + 1, len(v) - 1)
    return {"count": len(v), "mean_ms": statistics.fmean(v),
            "median_ms": statistics.median(v),
            "p95_ms": v[lo] + (p - lo) * (v[hi] - v[lo]), "max_ms": v[-1]}


def metric(count, begin, end):
    if count <= 0 or end <= begin:
        raise ValueError("invalid completed-frame measurement")
    elapsed = (end - begin) / 1e9
    fps = count / elapsed
    return {"completed_frames": count, "elapsed_seconds": elapsed,
            "fps": fps, "strictly_above_60": fps > 60.0}


def csv_rows(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def command(args, log, timeout=300):
    with Path(log).open("w", encoding="utf-8") as f:
        f.write(json.dumps([str(a) for a in args]) + "\n")
        f.flush()
        p = subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout, check=False)
    if p.returncode:
        raise RuntimeError(f"{Path(str(args[0])).name} returned {p.returncode}; see {log}")


def read_exact(stream, length, timeout=60):
    """Finite, cancellable native/decoder wait. Timeout is not frame pacing."""
    data = bytearray(length)
    view = memoryview(data)
    pos = 0
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        sel.register(stream, selectors.EVENT_READ)
        while pos < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not sel.select(max(0, remaining)):
                raise TimeoutError("no complete native/decoder record within watchdog interval")
            n = stream.readinto(view[pos:])
            if not n:
                raise EOFError(f"stream ended at byte {pos}/{length}")
            pos += n
    return data


def write_all(stream, data):
    view = memoryview(data)
    while view:
        n = stream.write(view)
        if not n:
            raise BrokenPipeError("native input stopped")
        view = view[n:]


def stop_process(p):
    if p is None:
        return
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
    for name in ("stdin", "stdout"):
        f = getattr(p, name, None)
        if f:
            f.close()


def put_wait(q, item, stopped):
    while not stopped.is_set():
        try:
            q.put(item, timeout=.2)
            return True
        except queue.Full:
            pass
    return False


def materialize(code):
    """Only trusted bundled code is unpacked into a new experiment directory."""
    binary = zlib.decompress(base64.b85decode(PAYLOAD.encode("ascii")))
    with zipfile.ZipFile(io.BytesIO(binary)) as z:
        for info in z.infolist():
            if Path(info.filename).name != info.filename or info.file_size > 200000:
                raise RuntimeError("unexpected embedded member")
            (code / info.filename).write_bytes(z.read(info))
    shutil.copy2(Path(__file__), code / Path(__file__).name)
    dump(code / "SHA256SUMS.json", {p.name: digest(p) for p in sorted(code.iterdir()) if p.is_file()})


def find_sdk(explicit):
    roots = [explicit] if explicit else []
    env = os.environ.get("SATC_SDK")
    if env:
        roots.append(Path(env))
    roots += [Path.home() / "nvCodecSDK/samples/Video_Codec_SDK_13.0.19"]
    for p in roots:
        p = Path(p).expanduser().resolve()
        h = p / "Samples/NvCodec/NvEncoder/NvEncoder.h"
        if h.is_file() and "NvEncOutputFrame" in h.read_text(errors="replace"):
            return p
    raise RuntimeError("SDK 13 not found. Supply --sdk /path/to/Video_Codec_SDK_13.0.19")


def find_model(explicit):
    candidates = [explicit] if explicit else []
    candidates += [Path("/data/D-SAV360/flowsal/models") / MODEL_NAME,
                   Path.home() / "D-SAV360/flowsal/models" / MODEL_NAME]
    for p in candidates:
        if Path(p).expanduser().is_file():
            return Path(p).expanduser().resolve()
    raise RuntimeError("Existing MoST-Sal ONNX model not found; supply --model /path/to/model.onnx")


def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, text=True, timeout=60, check=True)
    return json.loads(p.stdout)


def validate_video(path, total):
    d = probe(path)
    s = d["streams"][0]
    if (s.get("width"), s.get("height")) != (W, H):
        raise ValueError(f"{path.name}: use the existing prepared 4096x2048 input, not the original download")
    n = s.get("nb_frames")
    if not n or n == "N/A":
        raise ValueError(f"{path.name}: exact input frame count unavailable; supply the prepared MP4")
    if int(n) < total + GUARD_FRAMES:
        raise ValueError(f"{path.name}: needs {total + GUARD_FRAMES} unique input frames, has {n}; no looping/duplication is substituted")
    if s.get("pix_fmt") != "yuv420p" or s.get("color_space") != "bt709" or s.get("color_range") != "tv":
        raise ValueError(f"{path.name}: expected prepared 8-bit BT.709 limited-range YUV420")
    return d


def qp_from_classes(classes, codec):
    import numpy as np
    from core import tile_edges
    block = {"h264": 16, "hevc": 32, "av1": 64}[codec]
    xe, ye = tile_edges(W, H)
    x = np.minimum(np.arange((W + block - 1) // block) * block + block // 2, W - 1)
    y = np.minimum(np.arange((H + block - 1) // block) * block + block // 2, H - 1)
    tx = np.searchsorted(xe[1:], x, side="right")
    ty = np.searchsorted(ye[1:], y, side="right")
    labels = np.asarray(classes)[ty[:, None], tx[None, :]]
    return np.ascontiguousarray(np.array([2, 0, -2], dtype=np.int8)[labels])


class Source:
    def __init__(self, video, run, total, warmup, gate, stop):
        self.q = queue.Queue(QUEUE_FRAMES)
        self.done = threading.Event()
        self.error = None
        self.events = []
        self.log = (run / "source_decode.log").open("w")
        # Guard frames prevent a decoder EOF/loop boundary inside the measured input.
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(video),
                "-map", "0:v:0", "-an", "-sn", "-frames:v", str(total + GUARD_FRAMES),
                "-vf", "hwdownload,format=nv12", "-fps_mode", "passthrough",
                "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1"]
        dump(run / "source_command.json", args)
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)

        def work():
            try:
                for fid in range(total):
                    if stop.is_set():
                        return
                    if fid == warmup:
                        while not gate.wait(.2):
                            if stop.is_set():
                                return
                    started = time.monotonic_ns()
                    raw = read_exact(self.proc.stdout, FRAME_BYTES)
                    available = time.monotonic_ns()
                    self.events.append({"frame_id": fid, "read_start_ns": started,
                                        "raw_available_ns": available})
                    if not put_wait(self.q, (fid, raw, started, available), stop):
                        return
            except BaseException:
                if not stop.is_set():
                    self.error = traceback.format_exc()
            finally:
                self.done.set()
        self.thread = threading.Thread(target=work, name="raw-source", daemon=True)
        self.thread.start()

    def close(self):
        # This FFmpeg child was created by this session. No unrelated process is killed.
        if self.proc.poll() is None:
            self.proc.terminate()
        self.thread.join(timeout=5)
        stop_process(self.proc)
        self.log.close()
        if self.thread.is_alive():
            raise RuntimeError("source worker failed to terminate")


class Inference:
    def __init__(self, source, model, codec, total, stop):
        import numpy as np
        from core import history_ids
        from live_model import preprocess_nv12, normalize_model_frame
        self.q = queue.Queue(QUEUE_FRAMES)
        self.done = threading.Event()
        self.error = None
        self.events, self.labels = [], []
        self.raw_maps = np.empty((total, 144, 192), np.float32)
        self.scores = np.empty((total, 5, 9), np.float64)
        model.reset()

        def work():
            history = {}
            try:
                for expected in range(total):
                    while not stop.is_set():
                        try:
                            item = source.q.get(timeout=.2)
                            break
                        except queue.Empty:
                            if source.error:
                                raise RuntimeError(source.error)
                            if source.done.is_set():
                                raise RuntimeError("source ended before all input frames")
                    else:
                        return
                    fid, raw, read_start, available = item
                    if fid != expected:
                        raise RuntimeError("source frame gap before live inference")
                    prep = time.monotonic_ns()
                    history[fid] = normalize_model_frame(preprocess_nv12(raw, W, H))
                    ids = history_ids(fid)
                    classes, _, stamps = model.infer([history[i] for i in ids], ids)
                    history.pop(fid - 160, None)
                    qp = qp_from_classes(classes, codec)
                    ready = time.monotonic_ns()
                    self.raw_maps[fid] = model.last_raw
                    self.scores[fid] = model.last_scores
                    self.labels.append({"frame_id": fid, "classes_5x9_hex": classes.tobytes().hex()})
                    self.events.append({"frame_id": fid, "preprocess_start_ns": prep,
                                        **stamps, "codec_qp_ready_ns": ready})
                    if not put_wait(self.q, (fid, raw, read_start, available, qp), stop):
                        return
            except BaseException:
                if not stop.is_set():
                    self.error = traceback.format_exc()
            finally:
                self.done.set()
        self.thread = threading.Thread(target=work, name="live-most-sal", daemon=True)
        self.thread.start()


class Bitstream:
    def __init__(self, path, codec, total):
        self.f = path.open("wb", buffering=1024 * 1024)
        self.codec, self.total = codec, total
        self.ivf_from_sdk = None

    def write(self, packet, fid):
        if self.codec == "av1":
            if self.ivf_from_sdk is None:
                self.ivf_from_sdk = packet[:4] == b"DKIF"
                if not self.ivf_from_sdk:
                    self.f.write(struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32,
                                             b"AV01", W, H, MEDIA_FPS, 1, self.total, 0))
            if not self.ivf_from_sdk:
                self.f.write(struct.pack("<IQ", len(packet), fid))
        self.f.write(packet)

    def close(self):
        self.f.close()


def verify_stream(path, expected, run):
    """Independent decoder count AFTER timing. Does not inflate headline FPS."""
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-xerror",
            "-hwaccel", "cuda", "-i", str(path), "-map", "0:v:0", "-an", "-sn",
            "-fps_mode", "passthrough", "-progress", "pipe:1", "-f", "null", "-"]
    with (run / "verification_decode_errors.log").open("w") as log:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=log, text=True,
                           timeout=600, check=False)
    (run / "verification_decode_progress.txt").write_text(p.stdout, encoding="utf-8")
    frames = [int(l.split("=", 1)[1].strip()) for l in p.stdout.splitlines() if l.startswith("frame=")]
    count = frames[-1] if frames else 0
    metadata = probe(path)
    s = metadata["streams"][0]
    expected_codec = {".h264": "h264", ".hevc": "hevc", ".ivf": "av1"}[path.suffix]
    report = {"return_code": p.returncode, "decoded_frames": count,
              "expected_frames": expected, "width": s.get("width"), "height": s.get("height"),
              "codec_name": s.get("codec_name"), "expected_codec": expected_codec,
              "measurement": "postrun count, not timed receiver FPS"}
    dump(run / "decoder_verification.json", report)
    if (p.returncode or count != expected or (s.get("width"), s.get("height")) != (W, H)
            or s.get("codec_name") != expected_codec):
        raise RuntimeError(f"independent decoder count/geometry check failed: {report}")
    return report


def check_workers(source, inference):
    for worker in (source, inference):
        if worker is not None and worker.error:
            raise RuntimeError(worker.error)


def run_one(args, root, job, model):
    import numpy as np
    from shared_frame import SharedFrame
    run = root / "runs" / job["id"]
    run.mkdir(parents=True)
    video, codec, method, rate = Path(job["video"]), job["codec"], job["method"], job["target_mbps"]
    total = args.warmup_frames + args.frames
    summary = {**job, "status": "ERROR", "scope": SCOPE, "preset": "p4",
               "media_fps": MEDIA_FPS, "pacing": "none", "drop_policy": "none; bounded queues block",
               "warmup_frames": args.warmup_frames, "planned_measured_frames": args.frames,
               "processing_queue_frames_per_stage": QUEUE_FRAMES}
    dump(run / "config.json", summary)
    events = []
    source = inference = encoder = shared = bitstream = None
    enc_log = (run / "nvenc.log").open("w")
    stop, gate = threading.Event(), threading.Event()
    t0 = t1 = None
    failure = None
    stream_path = run / ("encoded." + {"h264": "h264", "hevc": "hevc", "av1": "ivf"}[codec])
    print(f"\n{job['id']}: p4, {method}, {rate} Mbit/s media rate, UNPACED", flush=True)
    try:
        shared = SharedFrame(FRAME_BYTES)
        encoder = subprocess.Popen([str(root / "build/nvenc_uncapped"), codec,
                                    str(rate * 1000000), str(shared.fd)],
                                   pass_fds=(shared.fd,), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=enc_log, bufsize=0)
        if read_exact(encoder.stdout, 4) != b"RDY1":
            raise RuntimeError("native encoder did not become ready")
        bitstream = Bitstream(stream_path, codec, total)
        source = Source(video, run, total, args.warmup_frames, gate, stop)
        if method == "roi":
            inference = Inference(source, model, codec, total, stop)
        block = {"h264": 16, "hevc": 32, "av1": 64}[codec]
        zero = np.zeros(((H + block - 1) // block, (W + block - 1) // block), np.int8)
        for expected in range(total):
            if expected == args.warmup_frames:
                # Every warmup output is complete. No measured-frame inference
                # has started: the source gate is still closed.
                t0 = time.monotonic_ns()
                gate.set()
            q = inference.q if inference else source.q
            deadline = time.monotonic() + 60
            while True:
                check_workers(source, inference)
                if encoder.poll() is not None:
                    raise RuntimeError("native encoder exited; see nvenc.log")
                try:
                    item = q.get(timeout=.2)
                    break
                except queue.Empty:
                    worker = inference if inference else source
                    if worker.done.is_set() or time.monotonic() > deadline:
                        raise RuntimeError("pipeline ended or stalled before all frames completed")
            if inference:
                fid, raw, read_start, available, qp = item
            else:
                fid, raw, read_start, available = item
                qp = zero
            if fid != expected:
                raise RuntimeError("noncontiguous input IDs; refusing to report inflated FPS")
            submit = time.monotonic_ns()
            shared.copy_from(raw)
            write_all(encoder.stdin, INPUT.pack(b"FRM1", fid, len(raw), qp.nbytes))
            write_all(encoder.stdin, qp.tobytes())
            magic, echoed, enc_start, enc_end, size = OUTPUT.unpack(read_exact(encoder.stdout, OUTPUT.size))
            if magic != b"AU01" or echoed != fid or not 0 < size < 20000000 or enc_end < enc_start:
                raise RuntimeError("invalid native completed frame record")
            packet = read_exact(encoder.stdout, size)
            bitstream.write(packet, fid)
            done = time.monotonic_ns()
            events.append({"frame_id": fid, "source_read_start_ns": read_start,
                           "raw_available_ns": available, "bridge_submit_ns": submit,
                           "gpu_upload_start_ns": enc_start, "encoded_ns": enc_end,
                           "bitstream_complete_ns": done, "encoded_bytes": size,
                           "qp_bytes": qp.nbytes, "qp_crc32": zlib.crc32(qp.tobytes())})
            if fid + 1 == total:
                t1 = done
            if fid >= args.warmup_frames and (fid + 1 - args.warmup_frames) % 300 == 0:
                print(f"  completed {fid + 1 - args.warmup_frames}/{args.frames} measured frames", flush=True)
        encoder.stdin.close()
        encoder.wait(timeout=30)
        if encoder.returncode:
            raise RuntimeError("encoder failed at final flush")
        check_workers(source, inference)
    except BaseException:
        failure = traceback.format_exc()
        if isinstance(sys.exc_info()[1], KeyboardInterrupt):
            summary["interrupted"] = True
    finally:
        stop.set(); gate.set()
        cleanup_errors = []
        if source:
            try:
                source.close()
            except Exception:
                cleanup_errors.append(traceback.format_exc())
        if inference:
            inference.thread.join(timeout=30)
            if inference.thread.is_alive():
                cleanup_errors.append("live inference worker did not terminate")
        stop_process(encoder)
        if shared:
            shared.close()
        if bitstream:
            bitstream.close()
        enc_log.close()
        if cleanup_errors:
            failure = (failure or "") + "\n".join(cleanup_errors)
            summary["unsafe_to_continue"] = True
    if events:
        csv_rows(run / "frame_events.csv", list(events[0]), events)
    if source and source.events:
        csv_rows(run / "source_events.csv", list(source.events[0]), source.events)
    if inference and inference.events:
        csv_rows(run / "model_events.csv", list(inference.events[0]), inference.events)
        csv_rows(run / "inference_classes.csv", ["frame_id", "classes_5x9_hex"], inference.labels)
    if not failure:
        try:
            expected_ids = list(range(total))
            if [r["frame_id"] for r in events] != expected_ids:
                raise RuntimeError("completed-frame coverage mismatch")
            if [r["frame_id"] for r in source.events] != expected_ids:
                raise RuntimeError("source coverage mismatch")
            if inference and [r["frame_id"] for r in inference.events] != expected_ids:
                raise RuntimeError("one live inference per encoded frame was not verified")
            measured = events[args.warmup_frames:]
            summary["measurement"] = metric(len(measured), t0, t1)
            summary["measured_input_frames"] = len(measured)
            summary["all_completed_frames"] = total
            summary["missing_or_dropped_source_frames"] = 0
            summary["measured_inference_calls"] = args.frames if inference else 0
            summary["encoded_media_mbps"] = sum(r["encoded_bytes"] for r in measured) * 8 / (args.frames / MEDIA_FPS) / 1e6
            summary["bitstream_wall_output_mbps"] = sum(r["encoded_bytes"] for r in measured) * 8 / summary["measurement"]["elapsed_seconds"] / 1e6
            summary["upload_and_encode"] = distribution([(r["encoded_ns"] - r["gpu_upload_start_ns"]) / 1e6 for r in measured])
            summary["source_read_to_bitstream"] = distribution([(r["bitstream_complete_ns"] - r["source_read_start_ns"]) / 1e6 for r in measured])
            if inference:
                model_rows = inference.events[args.warmup_frames:]
                summary["inference_call"] = distribution([(r["inference_end_ns"] - r["inference_start_ns"]) / 1e6 for r in model_rows])
                summary["preprocess_model_map"] = distribution([(r["codec_qp_ready_ns"] - r["preprocess_start_ns"]) / 1e6 for r in model_rows])
            print(f"  measured throughput = {summary['measurement']['fps']:.3f} FPS; validating output", flush=True)
            summary["decoder_verification"] = verify_stream(stream_path, total, run)
            if inference:
                print("  auditing every saliency map after timing", flush=True)
                audit_dir = run / "map_audit"
                audit_dir.mkdir()
                inference.raw_maps.tofile(audit_dir / "raw_saliency.f32")
                np.save(audit_dir / "compact_scores.npy", inference.scores, allow_pickle=False)
                from audit_optimization import audit_run
                summary["map_audit"] = audit_run(run)
                if summary["map_audit"]["status"] != "PASS":
                    raise RuntimeError("optimized saliency decisions differ from reference")
            summary["status"] = "VALID"
        except BaseException:
            failure = traceback.format_exc()
            if isinstance(sys.exc_info()[1], KeyboardInterrupt):
                summary["interrupted"] = True
    if failure:
        (run / "ERROR.txt").write_text(failure, encoding="utf-8")
        summary["error"] = failure.splitlines()[-1]
        print(f"  ERROR: {summary['error']}", flush=True)
    dump(run / "summary.json", summary)
    return summary


def summarize(root, plan, records):
    rows = []
    for job in plan:
        r = records.get(job["id"], {})
        m = r.get("measurement") or {}
        rows.append({"run": job["id"], "video": job["video_name"], "codec": job["codec"],
                     "method": job["method"], "target_media_mbps": job["target_mbps"],
                     "status": r.get("status", "NOT_RUN"), "completed_frames": m.get("completed_frames"),
                     "elapsed_seconds": m.get("elapsed_seconds"), "measured_fps": m.get("fps"),
                     "valid_and_above_60": r.get("status") == "VALID" and m.get("strictly_above_60", False),
                     "error": r.get("error", "")})
    csv_rows(root / "SUMMARY.csv", list(rows[0]), rows)
    roi = [r for r in rows if r["method"] == "roi"]
    result = {"planned_runs": len(rows), "valid_runs": sum(r["status"] == "VALID" for r in rows),
              "roi_runs": len(roi), "all_roi_valid_and_above_60": all(r["valid_and_above_60"] for r in roi),
              "scope": SCOPE, "no_selection": "Every planned run appears, including errors and FPS below 60"}
    dump(root / "VERDICT.json", result)
    lines = [VERSION, SCOPE, "", "Complete measured frames / actual wall seconds; no pacing.",
             "The 60 FPS field in the encoder is a media timebase, not an input/output limiter.",
             "No measured result is rounded up for the >60 decision.",
             "No maximum-FPS claim about the GPU is made; these are this implementation's measured throughputs.",
             "", "video          codec method  Mbps  status      FPS"]
    for r in rows:
        v = f"{r['measured_fps']:.3f}" if r["measured_fps"] is not None else "NA"
        lines.append(f"{r['video']:14} {r['codec']:5} {r['method']:7} {r['target_media_mbps']:4}  {r['status']:9} {v}")
    lines += ["", json.dumps(result, indent=2), "",
              "Stage/queue latency here is measured under unpaced saturation, not geographic/network latency.",
              "Network trace replay, RT-MPC decisions, delivered FPS, PSNR, SSIM and LPIPS are not measured by this script.",
              "Raw streams and all saliency audits remain local; the upload includes logs and frame timestamps."]
    (root / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def package(root):
    final = root / "UPLOAD_UNCAPPED_P4_RESULTS.zip"
    temp = root / "UPLOAD_UNCAPPED_P4_RESULTS.zip.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if not p.is_file() or p in (final, temp):
                continue
            if any(x in rel.parts for x in ("build", "trt_cache", "__pycache__", "map_audit")):
                continue
            if p.suffix in (".h264", ".hevc", ".ivf"):
                continue
            z.write(p, str(rel))
    os.replace(temp, final)
    return final


def self_test():
    assert metric(1200, 0, 20_000_000_000)["strictly_above_60"] is False
    assert metric(1200, 0, 19_000_000_000)["strictly_above_60"] is True
    assert metric(1200, 0, 21_000_000_000)["strictly_above_60"] is False
    assert distribution([1., 2., 3.])["p95_ms"] == 2.9
    assert INPUT.size == 20 and OUTPUT.size == 32
    binary = zlib.decompress(base64.b85decode(PAYLOAD.encode("ascii")))
    with zipfile.ZipFile(io.BytesIO(binary)) as z:
        for name in ("core.py", "live_model.py", "shared_frame.py", "map_projection.py", "audit_optimization.py"):
            compile(z.read(name), name, "exec")
        native = z.read("nvenc_uncapped.cpp").decode()
        assert "NV_ENC_PRESET_P4_GUID" in native
        assert "NV_ENC_PRESET_P6_GUID" not in native and "NV_ENC_PRESET_P7_GUID" not in native
        assert "sleep(" not in native
        assert "NV_ENC_MULTI_PASS_DISABLED" in native
    with tempfile.TemporaryDirectory(prefix="satc_uncapped_selftest_") as d:
        p = Path(d) / "test.ivf"
        s = Bitstream(p, "av1", 2)
        s.write(b"payload0", 0); s.write(b"payload1", 1); s.close()
        assert p.read_bytes()[:4] == b"DKIF" and p.stat().st_size == 72
    print("PASS: accounting, strict >60 decision, embedded Python syntax, IVF framing, frozen native settings")
    print("These checks do not exercise CUDA, NVENC, ONNX Runtime, or the user's GPU.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, action="append", help="prepared 4096x2048 MP4; repeat for several videos")
    p.add_argument("--model", type=Path)
    p.add_argument("--sdk", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--frames", type=int, default=1200, help="completed frames per measured batch (default: 1200)")
    p.add_argument("--warmup-frames", type=int, default=240)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(); return 0
    if os.geteuid() == 0:
        p.error("Run without sudo. This benchmark requires no network/admin changes.")
    if args.frames < 600 or args.warmup_frames < 160:
        p.error("Use at least 600 measured and 160 warmup frames; defaults are 1200 and 240")
    # Use the environment already used by the validated live pipeline. No installs.
    for key, value in {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}.items():
        os.environ[key] = value
    env_python = Path(os.environ.get("SATC_PYTHON", str(Path.home() / "venvs/satc/bin/python")))
    if env_python.is_file() and Path(sys.prefix).resolve() != env_python.parent.parent.resolve():
        os.execv(str(env_python), [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    videos = args.video or [Path.home() / "Downloads" / PREPARED_FOLDER / f"{n}_4096x2048_60fps.mp4"
                            for n in ("basketball", "rollercoaster", "ballet")]
    videos = [v.expanduser().resolve() for v in videos]
    if len(set(videos)) != len(videos):
        p.error("duplicate input paths are not independent test videos")
    for v in videos:
        if not v.is_file():
            p.error(f"Prepared input not found: {v}; use --video /exact/path.mp4")
    root = args.output.expanduser().resolve() if args.output else Path(tempfile.mkdtemp(
        prefix="SATC_UNCAPPED_P4_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_"),
        dir=Path.home() / "Downloads"))
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    print("RESULTS:", root, flush=True)
    print("Unpaced source + MoST-Sal + NVENC; no WebRTC or Mininet throughput claim.", flush=True)
    code = root / "code"; code.mkdir()
    plan, records = [], {}
    result = None
    try:
        materialize(code)
        sys.path.insert(0, str(code))
        # Retained maps, streams, logs and TensorRT engine cache are new files.
        # Refuse to start a long campaign with clearly insufficient disk space.
        required_free = max(8 * 1024**3, len(videos) * (args.frames + args.warmup_frames) * 2200000)
        free = shutil.disk_usage(root).free
        if free < required_free:
            raise RuntimeError(f"Need at least {required_free / 1024**3:.1f} GiB free in the results filesystem; available {free / 1024**3:.1f} GiB")
        for executable in ("ffmpeg", "ffprobe", "cmake", "c++", "nvidia-smi"):
            if not shutil.which(executable):
                raise RuntimeError(f"Required existing program is missing: {executable}")
        sdk, model_path = find_sdk(args.sdk), find_model(args.model)
        metadata = []
        for v in videos:
            metadata.append({"path": str(v), "sha256": digest(v), "probe": validate_video(v, args.frames + args.warmup_frames)})
        command(["nvidia-smi", "-q"], root / "gpu_before.txt")
        command(["ffmpeg", "-version"], root / "ffmpeg_version.txt")
        dump(root / "protocol.json", {"version": VERSION, "scope": SCOPE,
             "preset": "p4", "tuning": "low_latency", "rate_control": "CBR", "multi_pass": "disabled",
             "gop": 240, "b_frames": 0, "lookahead": 0, "adaptive_quantization": False,
             "vbv_seconds": .25, "roi_offsets_low_medium_high": [2, 0, -2],
             "media_timebase_fps": MEDIA_FPS, "wall_clock_pacing": None,
             "frames_per_batch": args.frames, "warmup_frames": args.warmup_frames,
             "source_guard_frames": GUARD_FRAMES, "queues": "4 frames; blocking; never discard",
             "model": str(model_path), "model_sha256": digest(model_path), "sdk": str(sdk),
             "video_inputs": metadata, "python": sys.executable,
             "timing_start": "after all warmup outputs complete, before opening source gate for measured frames",
             "timing_end": "after the final measured completed bitstream is read and written to buffered output",
             "includes": "source read/preprocessing, GPU model, QP map, transfer, NVENC, pipeline fill/drain",
             "excludes": "initialization, warmup, postrun decoder/map verification, network, display, durable disk fsync",
             "assertion": "greater than 60 only if exact unrounded FPS > 60 AND all integrity checks pass",
             "NVENC_reference": "https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html"})
        for vi, video in enumerate(videos):
            label = video.stem.replace("_4096x2048_60fps", "")
            for ci, codec in enumerate(CODECS):
                for ri, rate in enumerate((12, 35)):
                    methods = ("uniform", "roi") if (vi + ci + ri) % 2 == 0 else ("roi", "uniform")
                    for method in methods:
                        plan.append({"id": f"{len(plan)+1:02d}_{label}_{codec}_{rate}_{method}",
                                     "video_name": label, "video": str(video), "codec": codec,
                                     "target_mbps": rate, "method": method})
        dump(root / "frozen_run_order.json", plan)
        command(["cmake", "-S", code, "-B", root / "build", f"-DSDK_TOP={sdk}", "-DCMAKE_BUILD_TYPE=Release"], root / "build_configure.log")
        command(["cmake", "--build", root / "build", "-j", str(min(os.cpu_count() or 2, 4))], root / "build_compile.log")
        from live_model import LiveModel, preprocess_nv12
        model = LiveModel(model_path, root, provider="auto", map_backend="compact-cpu")
        # Existing model interface, checkpoint hash and bound inference equivalence are checked.
        a = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(videos[0]), "-map", "0:v:0", "-an",
             "-frames:v", "1", "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1"]
        one = subprocess.run(a, capture_output=True, timeout=60, check=True).stdout
        if len(one) != FRAME_BYTES:
            raise RuntimeError("first-frame preflight geometry mismatch")
        low = preprocess_nv12(one, W, H)
        model.validate_bound_inference([low] * 20)
        for job in plan:
            record = run_one(args, root, job, model)
            records[job["id"]] = record
            result = summarize(root, plan, records)
            if record.get("unsafe_to_continue") or record.get("interrupted"):
                break
        command(["nvidia-smi", "-q"], root / "gpu_after.txt")
    except BaseException:
        failure = traceback.format_exc()
        (root / "SETUP_OR_CAMPAIGN_ERROR.txt").write_text(failure, encoding="utf-8")
        print(failure, file=sys.stderr, flush=True)
    finally:
        if plan:
            result = summarize(root, plan, records)
        upload = package(root)
        print("\nUPLOAD THIS FILE:\n" + str(upload), flush=True)
        if result:
            print(f"Valid runs: {result['valid_runs']}/{result['planned_runs']}", flush=True)
            print("All proposed-method cases valid and strictly above 60 FPS:", result["all_roi_valid_and_above_60"], flush=True)
    return 0 if result and result["valid_runs"] == result["planned_runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
