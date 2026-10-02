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

VERSION = "SATC_Integrated_Real5G_60FPS_v14"
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
_PAYLOAD_B85 = 'c-mB&V{9f2({Qia*xI&j+qSvwRok|<wzsxzY;D`Nt@r-Ee@`;WWX|82OinVRBnt+P1^@t{0X)U*`l3^t&gmck01+4ffcPI3l{dCFmjgPxI5W6-xTL5|*ljQ&cU{n8h#<}XMU27IIUJU;A^MkqD#Njf+8=)r!Y*EcX8w=r{-ziKN_w+NNMTd|+VnS511j!ePD=+a@AchE8GnV1w6^M$HCLWo@3HjFmSQq-qi$1TWe$2yDii6`QIOs|WYiLkA|$-FqkIij9JRfUN?J$?(^n#;wJrKh=w^iXWR$qH&dH<%l+mM`GPBNcq=Cg#7eq%zW=r7X!c9rn{?+I4S=BFndk3z{)=*Wd1KH}B1pmk=<;cj(4>m;r4_??;&P@?Hs{(cwt5n=ZK5jUmdx1m90hs`~2|tJl5}mGD1S*^q`NP-?ktT{b_@T26yE=;KL5ynX?$;64znp$l`E4_Asb8~DV|WohK9?zw!mr-&db&V~(9evJ;XUvxgv!*S;u}KQpYYwU8S&X;vzi8wFC8hlI1Z-wX}I1AoOz5cgZYl##&cZKOQ^*|p3E`3=J|bA0!v{Vf7Nt&I~L!gEK<$FtK+u!JT~K-u4p*y(EafXdanqfpP)N%UyoZVG^!*@e}K~e^bqqZb;TME06-=H0HFS(ruI(e3=W>%+P3x^5-8s_hV^Ot3Ptp+i6$R5`ims;h74Rp@|i`3E<tr9n#%)dW0#2u$wWV2xf%K@MdqF2ft4T^=I(CWCmGu3V@Z^P*zL+9lr*%i?S#;Ub}qS*V%07svOY~Bzav+vfV71iM5yzpPTk7Iv4sXVt_;ibWV*oQ(Lb+m`am=;oWGpnb~3!bj&!ui<x*(sKvSks)XQSXO1zb6bjQsr3bvCd6}@Pa?Ejh6SI9u?(($2M#vc`%*?Rc_Nhe&t*2><K=;hJVo?2(h4VLp7tyC{Vn@DKAx{(MCzKBXp+QT0s%xabxP7VJYzP$|%EiGC3_%cY=C$E@N|8Ns9q@@j3!mS~B-)#53KJhEypbx#3%$CEZe*IjBiM}KCWX<Zn;u}a3`8~Z)L6Nx0h~K=@p9%`L+8cJChYx)Df~nfM@F8>5%2oZVBt!;pN7wMz0K0oe@IU>!-}?%l-0peWyf!Uv;8H}3l~xr9T@n8cwLb(k(!3&?Z5$(qVMW<L<MaqgqP`~Ij!4OGVx3$)KmQ?%L;4c9b8@AqV_V;@oV9s*CcBH+w&l~@5QT{JcHxGQw-i0ojYk>w&0<095+OyKlj12vXL0?{jDVxai##5Ugcr}Q+UtHABa4LH>2Ac(?|m+P6Wo_lR*8e+tL_E*tO6;{qMlA2H2hz?G{t^2>ci6+rp1V6nlin__UuCWXcuk5YQ{^M`wzxZ5eADq?hR!_^7hNf^wCr*Bk;}uipB7G`-Xce<Gp*#Uj11Dv%+mU6SlN3XrbObO*2|FN&eWCgx<DZi6b=>Hhy3`r0e`R_TXMSW0M0|59;ER2oygRS`!3XUrP1nL@|pd6nYDw&va-ADUiX5r0!~SChKdvTV`wp_LR8Zl$wa)a3!cZAk&0ahFuvsO@cYVZ`hSJ^p5^oAeF?+lJ$2)44I*MT6rZ(qSnio#RYUO*`4VF7GviS`*@dH;IR})T{6MX%&q3@?m`IKP!HiDz%?PCFg+;t`mF|<fm1#u(np^E;xR*tcx$=ZfC{{{mP66}i}7;(Q`OVyd?b{is9n>%#bgULCoek1kqd<wf<!boXgS*E<3mw8*CLb#w?d0>u2Z~<D@ul=uN<fTA1ym1W*s8e3wnu1n==Yu{R_NMhI>S=15_M$jrDQO2SzC+F*ZEuKZJWf4M-AKL2ksk`VP*}U6&37K~)-u$=dn|AK}nE05eJ@WJ0OvU_pz)_SbL^4w8K%He}2_{4%Gv<6u_D%M-K40-}$n4&6t@(`Vd_tH7%#d%+$*>QVl_-xY9?qX4X--om>hZe+%iyF$a0hV>N4srtwb{3GVp&+{ggyyP`iG#8RweZ-mVZjYb&p-ChyTKQmQ3WHhCJATL#v5W{v!e=3oQ06SoY28nl-Y;7k2Ff^00axTqV;&}PK8c*~bz$Tc>!w=bYvCB-M_>u1bK*ttPKsY;I|p!+E2wBBV;aV07!z|uKif0+{&+68<j>qzh7hV1he#|p!yL1{_siIA$D^O*a2|W;*S$YSaq1nEJa=Cd5(7HMUE2wnVuCxG%dfmBI5#0&x>Y#MuFJIFNps}9;mDV(sW>p!=W|D9J_3Sni|1IoRK)y<8rpC$)*1)SWV{u}TBl#>j%ve<{=U>4>ny7~*!aH8D_c)BI3b4$(xXTlwrJr5ec5(fyhR?!X<pmZhu*tJ0o6=sJ5h7}tEOO_=~AwNWY1|($5t>t_i`}UBjY;4Sc|=tEXGLW6v_L!?>JCh*w!wd@HJI>)yIGcHS;4EK0Kw6-T5<h=LMr&%Z^SX4@>v`Nz^FRzoMMjw?1M4*6Kmnw4KTzjR=EYIP}U1a4DPl*0!!HHBctik5f2Pqldb@-9-E<yxq?Yvbl}K9xmL-*C8J7$-q-z7i<w81N^%V`O8-nbVq@bFz3E6G4RPtxfqcOV@lALU371srfp|DLb$XUN!oEf%+Uch^WNPC`cxpn4y?=HEcmn(wviNM{X<B^%P<C763z${6CPp&q}Zo#Ia{f2*Wrq3VCorx_71;@L>uXJkmr`@ftpUG=11TDc@f<RM+w{)8xDdm_$CIPtV&hV!7>hKM{R<Bp}BG>RK&ikl)|C4{4e0d4858!Q-QboYdYytjS}<L`KEB+vJ+#?R}xglUlcsrn@;<G6RMeYiDUnWc2J&$5_I*?6=g2J$+s^jToL_mf198E4VQYfDu(ChrEc~{Lt$>*;E9HGFqSSn<IySRUaUYMKNSFGLySmZ8f%>pQ7wCBa>b!zalQr0`!0_v(iDR{UUFpTZ1Jq>k<u*2=2RvCp%|DD)|X~1_nQWC((Bi^G+f<8b?1Mz(bmbN1GC@D?m~$zp<vS1oD9pTIcU>8T+`tZ&f`!{8X1C6ZR}TWm4x0yt37_153s5_Kr(`$N$k!2@t0~920ID3Z*MD|U`cyrW|*o=Ue7oP>4-fO#f$2Vwq~Q=p@<A>DoSze%&yx@%7Js}r$)NHy8)XMCT8@9Lb@s+;t$c7Hs(FqK~Ql~wFPAmW#B?mYunZfvEA(-Pe~iiAJBV7a=L?l2H^H?+){V*U|_C!*eX-@?dBA`ITE=Exeh$`{1xr78vo*_eE$CT-GfRWbFZ}fg*83?Wy7+|5IlHfCy@tf!crU9JIkB$&HO@Ggr!cU4H>TJPZc3W#uo6`pD?shddrjok+oY!jVqeoel5SSpYxQ7LyFuA69V(TEc@5vGA$X3uvwuPJiPD(SPaN1%oREDl!n8;n7bT_1UNkSQnk?v4o4`=zl5VB&(m3PK3kSrKRzkN5JwPY@N^EQsCqgLFp0p)Kb%qT9dSxm8|bPoAl=+oozjh)`-^@25lC7{C_kaIV7%G|oK4I%n5!f@<#WTf|3+}**hAvbbtqAm%i>Zfk{@$ih|p|{y>q<+ujq)YIaJWfCo>TPFIVqLO=>&+Z;kC?$Oy;f&60Vi6d;MYVYi8gg56@KSMv`L7B3=4Nb@YFZ)AgkoRD7ylfia4Vbv_KQWs2@0ykgx&B7P(3y^l-PUWuxJaF$JfmmiE5u{;~DW_Hor6~Eez@DqlhQ552V&KjgjPAm7wMys$GA@V1J(Nkt3AJFuT>%}Ti*xbI-r^NL$>{mCl<2@=06z-&<tTP!2I)`t+_g8BcAp`%V{A*~*~<V_-U<c>>vx2Vo-(}HLyn%F;#uW2i)*<t?E{{DZt6|q2VGlSZa8>O_4@mtFs1}frmH_;Aa+S}@u&sN-OGDt_{6R$wXmH5N4UUekm2*HCa{;n{vbc5K6c0(sLoAyma-DhN2|H_JF0T{pLkM^6#5|*_C{4!!>@1_(wWm%V)#Z{TcI?@YL>jse!P8hepDO3MyCZ>1S+g<BJc9}hVe#H3SB+v2U{`&A)f>*uPDaX(voVnti}pt=AVI(V8_^U9XG%0ed!J3O4SlotZ-fV;WYA-&gp}UW1K?zxN@|+>&Jc{3wC^SW-ph%$~goE2U}v|-nN<wPCmvDKb;O-0B|q%o_wK0GT$|u>i6SBbR5+K&?M;mE`i?~QArf5+?qmf-nn6#v|{;k<_(iJ2a>(_nJDdeKc4QCjqOIo6!c>7$g{fpOd%R1XF#q23o-oZConx3ShHHDJ?pj6S8@mbLFE!;1Oq9zntI&RV{f+Sm?-gR<Tm<>DxhWYz3I8vpzmV%-iF{`mE7sS@f~urp|CCVsH~Jj&Q$iratfYDg*ovxpaLi75C#txtcJt(<OG6`iQ8$JxG5i*@4PGi0>uL6ux!3j^geJzh9g9+TAQR>;3l%er9V*H{R8s9KEAs@y$cNu0HmS<02u#Kpq-1kg_E(1xtXDpxv4qO&D`lfJ+Ia>bl%`V@n0zwnuk*1Pi)9;b5DqdimU9%b+(DSJU*s)av(xUR!RcE1*EzX_`h6(qKS_wFgWv@Cj#J%es1=m{9b-9V)19faqr&;E?Rf0-EZa3R*(R7*jncJ<Ri<his_A2XeO$Z=HWTWy2~;+_{Jwi&Hjlrur$a{X)_#>G!*RTYENA@pgkazTBqA-(fu<bMw$At1c&zCT@MIg;$r292y%2nhUbTCE!U9HDqg{ru)s7S=wI$9qRn778)jt)7%>ytO?v>HL{?d4LqK4RjzIW={f>`k8jt-dQ(NP|a#3V3!a>|%C#GLPsw#q_?++qvVMVs3#>rBH)hj9*?I_ePho3o{sbcFQzmrs=QsutEGOOY^lYz5tf7IorS2v~Zpf!VmzZfoQpY6k=kak5nKnCqAW`ES+<OAIvE;-+x;@}}ddhN9KRGvkSC=_ZHkcFF~u|t72XOdH!B8IIbO7u~O9RnhZ=>;wC;^DBhMEv{`dnlo5lNk3Pt<+LBiqB`(3$HHP%RN=^tkKvz2>{afwE{_pOlc6rx?6^k?(JqR(>(l+he{rrkGg@%4G;>AB~~)4UwyjKL7n_>$Hiw)hRqvYO&gaRwr-chA<U`jt~2-Y1;t?&GRM3{%juSq7eBul7bSuqkPT?YS`TkVOJ9@-loIjqCMMX8kGJE|^Q^P5a}q>=ES=r45&BLryw)uY=o1Kgajw)X#uKjdC(nr3w_PLsG@Ec^irF7VS80d0;V-10FP1L%Ksh2$&#;)$6fy1)c5V;Kn>+r_dZl>ul|xA$1;m{s|GT%#=fx*ft^v9+EwHDfSl;iVC&c<p>4QvgOOvV!ZyQa}SkdC)M8;(F-9S7ZnIY#no<cxgQdV9Kv0Y0!s)#~E%nAmHg4Q2Ihczy*nC7d8?vt)-?_ZCo%gAR1qFw_n<5TLS&Lt}&)8H;@7+IMX1<2ZQ&X^#x6QN?JL0FYp@ZdIlpF~bIB$kWY#5L^J1GH*Ef-Ei^fqbWTQ)=pIn?X1ILYDdQQK>NzN2`X~XD7RTL0itGhx~2(Ze#4>pS$=8bixHRunVT}@?(_;TItYRSII$DphFGbh&=7kemr7(b?PFD5D=`l(65$R2JvF40<NkeLFC{(>EMhazuJ*iZo=Ssy<*Pf1@Zc7X%DnSB<2j6w6~GDLN~MhjiXFYIIws*`mlfrj*(cwTj-5wx{8<5zi9WiD}}trGl|xflLXCvpxEPh*yFsc2)YZ7=YEPPys>`N>Fcv|68o$RN#S4A!RK?sthT$<W;ujQcOEQK+bY^Fq_{b;5v1ZjYeGUKI;uHK;oyVHp!sm9LEz+Y=ubCNGGZhY>6rCIeiaw^<?c@2fQqWnjhwWuDy1aPrkhmmi&&Z-=u~!%wUsE@;c;MXjAGHic`^A$qwewve0n<p{D+f}Mi1nXjn@YPKId-3lA`#BB|gJPzOR{Fd^~1J_tho5!MpH`44mA0OH_zQY}>h6;Y25Oem4oV33cJ$l^}PQe7qRYTU}qXb)oCo%qOz5=!TunUsHhna<TAl*u}8hKsyQ(HV*n;E^^hxY8nV0t!{4QN0%wKiYlXj8x83Sz$Pj7G&M+CO?P89_RRG{T|jTlcaXg?8*H{%4Qg#LE_a70V0)z0d!uFoB{Roz>u&1Qp*@)Ke6ItGt41K;EuqMiWS{UxR;frC@qI<w+}+ctpwb)h2Dzc=9w&7`s&a>U9lSyV=cY9KBjJ%o@~aw=GomKkz=6I#4r4cZ5VQO!j^j}+MZ{CsD_m3!1{h*G0G_6qwvaaa`xS>ooUXl)-2dN3yxf-mEEC1-+*Hd)=S4HvQW8G&7bnlPsU_!=rI+j{tPXgbWFJj7B>*Ky+U{|<`0;FL=}d58;~{PVt@<Hs_~2P+8-KL-P6&>NvvmDzC2m@Ou`X=TcmfIrG5f!K2;a3)lCH>&xny7x*tdOQTmTYkDeYzmsegaP$Ac{EM@{*%-+(&5sbVRxmp&pW*2jHKz|p5VpM%_UPqiLr^kWRZ*#jD^Y<Rk?*q%d17-iA7#g$ZOx6N~z-lTAv_^UgdegN}?ufpw1liW?;`8?=p&OKb0h-_i)(yGOpH_l0?86m;CnvJJn8LOM*l`kt-VMb8i;$Kb|sp97w=UZ~HmH1lv(X_1fM-wmKHb)ouc1sH;l%=EL+|Bv#hLMXcOpo?Y5VH&YTVI~+HUmnhQ`^;3HzeDcQ;BQJZ2OukSFD!CtcS*jiH#QaAiSn~xDUJ1#<=c3j418amynf%ps~<%x43xZE0~l&0=;~=UKU*I)J@wEdF}C(f_d47pFCL#5x+qzY#?8eoxIOUmRyDBodT^@lKGHYc)B7E^;=hn;oc7gAw070Hu)V*0xIeyS#i_xc=;}oY1pQ6;ISUP-<KawFRpZmk!_nUu1Z1pd_NHkIzFHF3%t!+OFL)aOGsTB*1guIiu(bm`9YnWHid=@Q<B+-VvIQ8Ml07_yHR~xT~)>5!g$`VXwf_o>-HRPC7=dtn|~{@=FrhtOZdbNaHrpE@tOHhdc_mrwzh|<bZJ0pao*Piu}F|cNufM$3^Z3=4?Gsu_s1VyAZO}JnjUXY`@GwqZjTF9wOaYXxlC&9O&M=V(L_qwirJ)8UHmd3QS&A_t^XT&>VK4v%GikC;Q#9mijWEkX@mfPwIcw4_P_4nY;I@v|IJ~G-&%8%t6}F2eLDlGCDPV1kx3f$gl;g=X8O2U==Y34>cK(F(w4zy%K_Bv+2GezYX#W^V>@vMuOWpqq>VOBicGOrIho)0gn{~@cO%uB>DZst>Iw5qzO9UtGnt1jPjwCM2R?cobD)8hUt&?Qp#)Ina#8$^<l!MmTiDT5oP1(}?GM~VuXv`Gc$3uUDovfVvV<^5oF{s+hVoPcAA%ML`NaTD8u!B>mA007lZi&g2T0aiKf5I9qpV=WSAmiEpK!W;c9vef;kl~scWeXeM~t~?5+|qU<|rc~PC7<_r>MVfa!0XZw(r&<QJG3AhH}R71<vfo%iLs81uyT9m%H1!(_xPDyZLEtF8Ytp3nJk1Q+?kQP-jo)$ao)>!K+r}JH6nE*4X&y)=}8siy}2FV~^3#C3{@`Tbns&q}-4INPCPabgT51KrW0k&`R<?+_lzWHpC`Vri}GwpmoH?oLeP0^M(Ep)FdNgpd62PdO3w#r=4(Uz*T9u_fg>&+gMiq(tLkVrspjle;k9^@qx%D<kAt!%S)VP%~eX!`4Fff#C=>__3scZy*;8Q@8o-3YLfJ7Sic^9_KU)#br>F&#XXN^n<3Ca=b2{j>UO7jB9J@Ve0;I3eX>NeqzZY8-ALuitK!H{jp^!fA!~t8(b=>H3}k@Dh&vUSb?IlHw0R*vaT)0?&&UndqR3Xjy!)Tg-)EB(PjJ(43C%<&mOb(n8r^N)PeDr4Iryvy?khUt=yfKJ<utzB+NJNCB9}tIZE^bPF)9DM2GA+AbTZ0d2=DsIC3T1Z;OcbhH={)~YZEWXy5%43Zn9~O&W^~1?0E#cjisr1OK)bEtd1!Av~MFx=R+a7($U&T__Rp+=d(mx>22Qg8XNLw^of{not3?qP>|BsL(5rx=6ZMCA{S}oC&(SV^wVCgs@zf=v#6@0I3#X^A-L?&aX?|hLOh_+EBHqUiQ`vwi>1%kQJ)*WxMe_z?D{inw+OP1n=l}(CuU}OC+t7d8G+mf4e6<lss=%T>y;z(?x+x4CYg-V`e>&m;m>il$e`^<O=*`Ca){ax!QCOIPOaa15$!>F#^%u|V8!(Did|m$kdw=c0d~mq#sHS;ai($oIfCs5*i*D4SL^RBZT&$-usyOqO)4#VEkdT2N1cc@L^WoUxH|N@3zmOV(yWlMyBe)1Zn%;#&aPV|bFh_Q7^>W@|LyOp%o*)ia@2MOdwt8B&EvAe7-2(SHaqdD$K|%ggfgKEb&Um_WTfb?Dj~;dfLsLmXiB@ktcYRPcrnp9tUqD{;qtjHTIxhclL2V161}T=J*?MIvmH#TUxN{GatK|75+f*7c-0qT5C{bXSJulwaVZ+6+PG(J2{?Lj<Bv%19KxxtU=1?SDPWc$_wY<7<rl8~+qAZhvT?4hH3cM&S|w}|BFX%X(w+39>r&RJgc`CWHvRJE!qLb8$o8y%K)Y8Cf@jsnPyY9dbA}m42ptu~YK!3@-xMDU<+>i5sk7EF*zx_ztiS&HXa`Fi23#tt|6dQer1ys^SaQzi@+n&kh+??;Zj(g6<W&>$nQ%3`cUzif>4{!RuJkIJRC>_h2Gr#12ba_#<%wifgq7Zme7u}U;#M}T2MDFD`;>##C1)d$zmvi0mmNH6OSpsJKVR?|)U-xJOj9er^AiEq^iqw@snhJp95{v=>mxe~g1HW)(C|OVRhPS&DP5s?l1vEvbvMU2?V#<cb>cb_W^q<VuOsywFQm7y(4T^F%|D(m+R?Dp0qv1N(CNBh%R$#CTz~k>gaqyGfWNe`sk8v6w%~p;p@=Y1BYI%E770bCrk0*5F27WjJ0|pA9SM%a-yZu83hD4}BXBUXnybVhq`1YAg1@>=MCLk;0tIA|=Dfj^f8~Z}P-A(?u{$&NXY%<R)351gmCoVC&hq?4zkXLJELU<&x!B6hXy!ByZ*T>q6e3rPy|7S*je?krd<rc?MhhL0cRO`<qfQ(vtekivaP>1HP3@Uv4DYEEy1O6kU?Wg0v+H)&_hyFnq$)Z~?saBFs4a+?8VOieQZcVbyzEhhGcmP?7}F}Lx^B0+x2kO=%=Q<$))fS*i0X6zGuuyZc?dtLJdy0@WxyHZf2ElW&g>6?EVK6!P=nz<-GF|;*fNwnZA(S*?S>tU_x^^Ik)Q&KmFO3#6ENvNONtaNnG`Z2A?BT_zKa`1LfC^EJa#dVXom}W_J0*42Wt|Qz?58)0IIp{55ypp;k)8Gsvj+w8|&p-Ue{~BnoZ1B`ec4+tN}^-t+ZPW+6ItXu<Z755Xwf|GJ1a46BEsUB}4Ip#S3)JmI?l-XZ_`UQzF+Ape59xQ&_|Ga%0%`<-)f|E@^2ZbcxRHgWSt&?&TGF@9xd}b`x=g&{Cx8Tf;wXBcj2-dTovTtgDDp2_^_8ANvTotiqtH(Q50jRSqCB;ML)`U8MlQByfl=v~SBNLAe`|9=L8YWRHgfi*WR^F6E??tS|C?f121H!ql3D?vcB2N1-y8*s6NT9CIY7<UC`Al?#$<VObV!`rTcjmmQcNiQ-<fBStLya&2R2?Wq3y;mpy~6b`aOO5@ZgEk)g1hgK^L14hyPfcMeCo?g32D^;s6@0vbwzU{RR*=*x$`yXtoXUF}80@FL_f{8#Df!qr{GK|}3gA;IMY25t}tVZoyf%+uw1d=dLDL0cuMlE@f_S?j_Y&WY(JdDhJQR6uj5RBgT7w5a`Y^jkb&n@(lcFOL2Wucn7iW+v9Y9HO8`4BetS-I^ynq7EpnEs;K_9G9*e$!upo|n<MOg}0AUdbOUy(F#Jk9OzVVAe!&ce)neh8vrEzD*dkBn!+BkjBWI{0s5e_0KTZQ-#OxpKSFl9Hp2h`EZjN{|FOv@Cte9&au1@H~e{ao>$^4$@SF7B^W+#tdting0NlFM$|*w4A^NtnE{S6whT$Q&RDopigOxx*us9+tES7nS!>I+q>*QA2?%V23_~;{gOVkaw^u*)Kh*{f(Kd<GD&NiwrX)(IsaU56$Ho{CZ3yNMUzi?==_rW)s{=ocjPHoWKU6r0_)iwg54J2E4fb-^33+!OV<Vc~YtHyR&wzj#s{@YgDN#tkWvGj;M5U9W-%o#(Wu{FzChL#apB?XeeEJJJc@q%t38Uhyl~jvAr7H%Z^J~_3>JvNGzu>(q#;Kl>u?aGs`?xvCQZZz{Ah-mSGgnDsrwVWAH>?y4phj_KVX=}|^LyDW0x&#fFnW$-I0iJ71M8iAqwYVpBV}4>;GK76*woHxP%c9sAy=?v;OZR>*v=7rY7kmFHABn8Se20u_o%_#r!knpx9;&h@GqATaFs(!x8%CVN>35AYPPccDRF&gTpB`L&S+=~Awr3yF;voVr78sZ&bwr6ayyX0fVQwH$Nb^C%$I&}-We*X{XL>TXFE$6SN;V#pZmsh7+4YPOkrYaowpk$FyYiR3mY!>_D;8O#=IFlmyUv>Eu2;xQ&wv%8l<bQ2b!|aK!r|XvJoWFU{~TDZR?OP`tcaa0Z#gZ`Y$9*f^}1OL?UO~s;LAB45fnTu#~2$98v2Dziw^bFZE@SK~&y!5y;jSPgd+7lwrcKV=8KjTNc5P^3jb3NQZwI_#5@C(4SEuhY7mD1x|ji*oF+t*v@`m_s3Ib*pwof9-#cOX>a=R3jVxb9(!L`3%vrlhS@m;{-=XQf2fy&K)ExS<PytS(JwQ>Y&Pi_9d*mX$dj<oX>!Zv0lDg-8%Ly3b{Y`FRC}0z0vXY7Fu)XD+y9wDXvFs#$=BXCldCgbN5TV~Pl=;Byoa5f9o4mR<aY|pGS<M=9o-kB7sTz6gzPdg%t!Ax197fie@t5{$qwKDqQ4Hj0!T@TX;Q0ZrM`f>3gB3Xy4J}X;P6Fz3U-q+C4oGk!k{`;G6nO>lCNSN$r@?^oHG1-Jko^Ng?RxYXIOfgJ^vyUrCx|QrFwE2?~LBzE|64fClvOp*0k9*5>TDvS-8LXZ>*bZp66YwKjO4H0AOJPNmNt0%84j&#E|X49|_UmKBQFSLH5eQ@Z*CssRdvu1l|B~Uj8t#_g4d62v#W^G#LiVPohxK8x}dP7cCQ{S`MCup#x+zY@ZmJ8%kUn#vTWf?14=mB>zB!cAjnSGl-|E-JsA6lE%89x0g$xbJo&xP6C^j-Vz~Y7a|#F;WzwmG$=Th0R`4{oK%M@n3XblE-m@~#862bszjb4_-dPacFFU007K*oeUMRyLe19uTg7-bMw+Qt1W{9^d>FH8&kqO}+XqkOEW`wSD=c5Gp`~&MbTyo8t8K%c(%KwYYW_g%7-Sz5-Ck0TX@<u?m2<3#QxMrUWXP7_3L}jQ%tAX8^?FT}EIf`YBU9W-sP%?~243TE3gOYdxq&O@4GJ8M4Ee|UizW#0<&tzycBu+tpG^wF`8T-+5KG8vVHOvG?hl`+VkSIw<_Jt`5~bMur|ah{CpoZS#4MjWI4_o0E`9BhB-9X^nUtbNZon3-Z;MQl0VE<kh9&0;<4hc*LQNaAkbexp^%23wZ#dV|X*0iC-vCP4!7b#gtFf`|wa~A<hsa`IsTsygXJi98DNaJ)Bu_nb*}zD-=?hwc+@kPzjO;_t^#-`z&S-IwQl+C2%U8;4Gcu0F%GYsUnF45oR;nk4GcKb3X?EhbR9kuZ``e2KJ1wn!(v@RHr9H|X4*yI^ce>9r16eU^^7$8^lbx+TUQVrm9QX%=!Ak*}knj(NJ9Dk96UCOPNLCsrfiF>ZQL>X^%7`tthrvp>7;?9T5*p>qEiUp^Musabm0{8hR?3mQnNBq*dxLJ?Jy7U@=|RpxJT;XWObw0cWwni6&QdvR`xQK5f<LbeM%v{$^NhDfVO%Cfx4frcns#<ISP?r)(|{}0=`VnDE-<GxavF%4RN^ss!8Hh(6Y0@sTkX5ldV+Q>yfxBqjSJFfs}n6;Pyae0FLF>4>Y1%GF*=0*in}my^@&?o1l}|e+34Sl%la8c#aO6mmwoT*2;5wAw7SNgfcAIZ0q0N3qFpnkd`Q6Juzt!eZ4LEagZ7+MnPaiLvK@P5nxjZ}KhbiF1>*Zh2i?aU)-X$^L3B8N!smxcB<b1R(yevM>JzI!(JmzJq>Mpn1#@-Ou*-3V)8JEWH9dd?XncFNnsZVX24_l1+$0-Ic$Egs&;L3kI>V(dG!wwD+IFgr%)oWh7T&OhZ1I;H4?hey2%~ipizaLcp<N79WOSB<SI*gW3YF-sKu0m|lFn{RL;a}P%OnPntYd{#l3rz1Xp)!4fGAVK8@6z{xVtxYX7T1?f{J?Q7Q0WQovIQu+fx@^5dU>+JY?zcn*R!DTBS!Lw2bLw_N#w9)(aB&3KM_vk(R{Rn@)@xiRSg>`Q5m9!t`vZ!RS0`h*R}QIHn>ZMA|NG!bA!q6Qo*iF&WtQS&vru+Xp^pJ;)dRz|^psXRl09i-aWSsSn3a|0<%3z<uAPfIA4s5Xv#_SBwFwaGh}+L9N<hIxLNYc78B|;LQl1@F}G1N*;Qc)75%JPdjCRY0O+{vPgo0IypMzgR<=<uNbs<OtQBYjI|c(@7_z}Kr-)rl0Y|*IK;e-0(%oB<b6NA8e?^~Pa4`;B{TGI9shq#Q+T9*=QtSxF06dODe`i*GRsGWt6!;!(nZOEkRrbl5<Ul<Co`9b){2Ly(;-K^Z8VVX=YPFcic_;_Sv&>Cem@d_08WU&#)8cTBx$HAXF!1ypA7^HEK`N{z<Id4NGav$?vHGK=%L=+z-2%aY#=4u*}?$R?ZnYHI!E8gPSS~mPGc%!e>S(G3`w+=TzGD%Lti%CAJf9t#;SRb`#RKI3Q*2wY_~B*FEhG4=O=gy4;8E{57Z|yM5-BRh>NQgcT9^B%+htp6rWh>K8XW_ogd1%P*b~f(ty!bb{ySFB~#<H5;?e(kx?rpUj*K|9nT0@>IEJ>cDB4U<Lj20(x*8H!_fg&|0Wucqm<qVP-}ufO+<*)(JQxdnG_?ZS`eae0}-|p`Jesi$GB2VsP_mU--D>U!Ycjnc3DQtUiHIA;W4XZoW+_k(|q_7OEd~0p2;OBG~@-xAqc+yUYiodeQ;rEVCC?_6RqbY)Dt%H$<hLtmztSg9B<E)V6N#U%{mXcXsTw5iuwuO#0Zdpx)R#J2Yyu8gv|*=GL;qV=G4jS^>SRhfp@S=We{SW_)l(mc<n7U2vn!^H;&`e&9Ab~K*D)b$*ODi+=44R$Wz-p-U|ol4!>R%JivkvzhLYF00M2AfUrplf0zq^%*OEyfjz8U;!HFb3A@>UaTD!^N9pO}lx@UrGXx@j79Gn$5h^ap8m<ecCNMylsN~nL)`{YQhP?6&HkI_4sz9KkxtPeg|8}Ug3S%AL(f{rQ)nq3^-5J*2-P$FC;=(;vSD$Y$=fr}nxlJta6?*)2_*I(Z?~zV->HGP3m+aovi~f2_tZZ^Z#ny_4Ms7Qhuptm+N0sJ`8t8ai)oyo>ex7n!!&dhXrczx4E2~<FT0GR`hX4|n-}Bg5r_hdnQ@}#1dlpqPu;hxEzk)j_Hj~5K9BU8B^D$`|4%@b6lesURWmq}3*C?N=rcZXI)y)|x)-wXg4t0$BC753kP8?Bv<?R%Se84Nh2EvC(Yla*OHFQcC&xX=v6wFL8muo4(kN<2b{%Lhh0+pAS>+}BF+CA)#<yNgQVoM6PO8_SaE~@{UC;PANfrNz2uG@nU<c|4X(b3~IGP*(AH{dmr7dw~en$0qg#BQX8+}wep(;{Rn)!AHxQdit)$&^1KQtTsRa1Q}#^{>K^8%BJ!sfj){I_%?k=u#F}cgo-CUmH<`oWzf3SU7am2mpsuhl(_@epkxq%y0w$P;`D2KlJ*N>gVFFfq}oxU|||Yw+{;!1AE@=$Gdeujif!l@I#pk^({S_pzub^*mNnFcUX=P!5Tr0zJP)l7=fe+^Q188{Z$%tD`)~P2X5_pj*(b&(==Caw{?O7I7KAKdg!NKrIb}?=bjn=sfF9b<1iouv>7Fp6Q`_(HJ~2vj{#QSFIy)qCTnYPsJQbk;)g(4aJ2Zp5Y{LV0WNbs{5goi6N@007Iqt*p59Ick;lMmJFV#0YJMgN3Rq8RM3UDYp9%{k1@<k?{ZAb7U|_6_d)S)gVOUU{(vkRv*j_BE+h$JHW`}bj%4A>XJeJ2a_l~irq|(A?47$4Bz6{z&M3DRq^wpl{FV~00g$I9sz6_C4ZrWvOqx8WfG*+d|6@);l{unlu9pNf^v3jM@i?!|pldPtc46op5mc_bEw)DLz9I4G5$u;>M=k~DoVWjY?3r9iE&4s+PWFLUWfSceqFWE7Y^xz4Slqk03sRF<-dJ56SmK6pyX8^|92KOKbe_%YZ=P3x|J(XX6TiYASnawwnaZO*)&_%@TidMHz<W!*qe%G{eV(ihX?@x8JUpNfpF&go<ZS}H1D^S6rdfoD^GV~EFJ;pUSF<h-r|Ehbmi0SgROuOysdMP1Ln>0mqm__#zW4#YP-Hmv_rgNvH*=Sg2m=lFV1tI_P(B#1~{n+)^$1orR=j)s#@ZQO(*ou2?$6sw>Isj5B7c4%tDIeXYB_mqmUoB=3@KpL|8cm+?5K#Wptb8UnEbesyV&({hzR4DBeGz=MS|#|#i|cpRr_0Z*uKIKQ4b28ue)BGY2FHf64MK|C7*(9t%4s-ckuWfiFV+g=tz}BqX7sx#AaK;_zV0ZLuiN+zg&pZ?n|o(Mo@^dK)>qK^Qv#DzJIk1<$(*mN;)iK&_tZB>4&O-fr!uj*8YZgXu82NiI8E|SM^^Z-hH9tE`z{a*UjfCxDGZE0qwUO8bN2La;hA91L-Ui~cj0pnlgHgEy^aXP%edc+=jXEuIBs=IYqtUT`|;fH*|cSt<ctp36H`~>nN%Td$$J`iv%_Cr&GXKG9!C93%2N&+$hb@{3DF*ZV|r^))_)`IDZm=2hy{}I*qg=ZRhrXGlRV13<EsDZf70910ewqlH#GE{UA1b+CltKctejofDTjN1O0Z47EdKqo9p5TVCJ<xyrL)<_nd8m;O3Al31m$abeHZS<;1wbveW((wsXs<x|8%xE(zNtG99zvI<gGvQ4YlG04^Cdsg(VJ`3VuPr(%=JMc=gK^g$OdO)dMIOEbtQoH%QRgb@imQeDF6_Bk_g6%FV(s1)ZYi)xo>9<8Mi#Ow>%Jk^)V%bgTjX642`{M>C?SBbg~B`%{LF8rQvkNovW-e7BG7K0_9Vo2#2uttTBmr%a@=Mpm6B2(}uC2P5HETcZ)3x~=c`YT6}!b1a~)1pnjsYkcKUL&skcFA?lW^&uh)Jb5W_*yNlb;fYGjq8n2l;4f*WvTYz5@cy!>wCAC4ly&d`hZ~o3YcyVx>OzHCI^B<lbg(&!IgYU$gK#q7t=ed4jeGuAn&O+kJ=H(rW4)s#(9Y5sq2~T-Zz!gX@Y`u($iC|@`wA!M@}ejAh*sh@sh+;qf-k-*tvtfC=2RyUL-&3%3<E2@07R<7g)WxQK=&aWexNe5efL)jKeko8o7TteR9uMHInUK<QQ7J*J2bc*!l|Ub?#>(5sg`P7wo;uiHAar|*x$<?k8Vfae%bWCXaR~bk^HYl*EGZcBjoJsbP36;!8H)>ew835sNXWxlI^aem!g%We}+E$>qvUEt5=5t?DuI@mjs6AUbZu3RJol$bP4wdH}Nlh%dQo#PGQFL-wWlX*L2%@>GelsEPh;d8<yhBwbMa3uz@szswROw<DlKH2B!!ycpA99OD(gX*+6wr0qiO#wC&hu2B1HjHr^Cf6?~87uhLdqS5?9dt7$FkR=<BY9}h#t7>TxYelH8KF(9@r!@+u$XiC5j3PEc=eI(8%OtTwu>P1jCl~YO(fS-tzhx#(dMRqQ<()2&;%ymjpQRvUbfP|}EJ9b9&KNVFwMb3t|eYYw6fyNwlu0d#<Cg)Wxs&n8qyjD6?&`ee<BB;gTBpwmY&>aW+B{Ox-E01mBiO0kQ=g#6*PaLYx$|WjJnjKAW#!W#;1*in>N&%kqlxm|)OGG118p!|<groXmX4%nDNZx~I%a?o0Pvs=N$-g}dYb@BNTblnmU~vs<ZnytMZuoBnu975#Rys?(U5>x%mnOX7c=Hc9`}J9Pe0guY>~6_j)bi6h7X{a6Axl=gR8J&W$?u}IqPExVRHi}4=eS`@b6@tvAC(=1)TGD2?zbSkaDg$LVmruO*c!D_FFDl0j{ekqtpIBToG1~vne|ITZGz=v98AJLC~gjv`u3dHx^T1kI9fBkQiO#AvHgVjksaxUy;%6oZ*J!YLTPQkt^Ldi5N@6cY($$g<atmmFbM_nE^VolV|`%U7u-nO5XdaUZce#lRDhj~bDX5&tH?;@4@tJSX4O&RD<+{q%gW72pUFIe-u`aSXH}i48R#9W0fcg<$vXl5^nVGibp2$09_0#$HP^$oekj*Mq_%5%=L!1zT)8dvVPvLdyc-#}V;@T&fJo~Lzbt(YIM6q6J9D{xZJtF_9_9W09u4(UjU!qg-0gRKyUJcP=i#AvRBrYk|C`-?rne4-TiSn92%j|}&OS~8y@KB_opJW^3J5g3#1d$c)b_DE-9Jul71Q&_m?c)AVWWz3DdBGIazKi%?Sn)A)Pl7=w7NyTs2YD!Ei!Z3AAx~N2{qiIeS0<deL)nRq%W}1)ib@#(;~rLD|ZEEj(tDeW<fvCpKUc#=8l}NvxfusM#AtD4{-4yN!AsQ3h+&yF5iB4FvXh&Dot_}SL6Fih<KtB5}7&5VAx>7L6e9vc8V`Vsfu$)s`kh%vxD4;ov+o+iR}+1LYLa;6p%xD>eKP6+mZf?+J$Y{+8VY1?Y!zUXz6b#XQ`?c>bVTvA9L~(E<?38=Z}nOEI2I}QuNHew2n4GVa!&F)BBo`;{mYfoyEyo!(BpTUff#JQn)$?w|}}9u%VSe3C+q^P%zO}+b?O8NH~#!R0nJgv@}+ibDe&cKcHQEy7q?GL-r3M{DR=5jk~trE7Um9`{(-63>n=G0U>_qY4<V)<ZQ4`0ltCja3N(jk2XI?e;Yu04Hf40jegeP%;DZQ(1OpyKTK0>+xMt5l{br*X4eQw9Z&pfgws%w^aMT)*PL%M8HQe4IBxWoi?Obrd$K-iUd1)R0F1f(-W4`F4_3B$8?@-nl%)94q^40A=iBF|&f^)51_GKKiT!L@Pz#HH49l4R_2tM1^13#bo(nKb7#&PU(SO^8H^{o`wktjm1q|Be1ber;_Vr#0F1DF%c>6;G7{af7?mmf;T|>;1t0k~-?1^d{pu9C(k4MsM`EEb;lNi?LK;8%~M`q?Kabv4D>?bjx<d+aAq0YlTFVCM#&ILZ8^jpE!{CQ}qJ$7f0{mUihp3nZ8w)}kRQhd@))_@k{?h*P7(izM+>u){W8mP=9LA_6}^z~G#&YnVYjw1Y>mbWv{;{Ad8-$TB5|3UHNe<g<y0D$lxwE?=B8`|2NncMuo-#=Y%=M9O(pPf9D>2lft80ds#x(OA>3p~jz*J{&FGwx=08cZu_c=0M>K))!;)xG<U;S3lSAJy1cyUMS?jI-;r{{nMayv1C&uBNgBS-5ZCHWwZJ0GO%X71h|}Q(&0Vd3+L<mPKZeG2)-5ZRKY%RYbELMQIJ1_YhTryLxkHNm2a7vBPZ8z^3UuScQ!6D|ENNI--_S{S3r_1oW3a+$SQcLF%2?NxgbvXdc)boeYG@!~+pygMW1VDQDk0XMd`uI37y2NR>G8yb?G$F~#T_$BA)qULGI!&&tPxX4*+FIN|r5)7@n{ea3^$beuuRg@y5}$gO1$$_1$(A1dQOp@rn!=POlqbUdM;AfMh0xZxzC;qB7c91Sdz=IhAHrsw^kG}$uUR2049J|oHM?*yqJ*Z$Ull%Kq80pr(~$Fl`ZZ7<S@+HxZ&oalwFn)K}9iWqg={(tyVzXgB19P2!u;j_ktJEWH0-W18okhJwfMCfTF8|mMn`<Puie^Co>Fw>!jg5%wH9Kvk;GfzS1&C}mc1J{hObpsRXBlmzn{#(>Y7G`1H-a-cYG!AJ$1YUAg2Q%fdW25%ny(j$#C>s$xBWr->Q;JOze$bq|mxh}Q_iQM8OyFhX=MwVMttF?Wkq)EWtcnUKk6kzyb5xt3K2WKP<{^jSXN;<HSeYJC+dPU#vJtOu7k#7tup<3FW}iWs;gp=Ss8=Jw*yvhsf0l_es7;E|42SEccqx3Phz-fLm~n8-MvXIfZMG2>mGIOaI(wQgUr4MCl`w!VbFiArACj{tgZ69-%%;><?>N@SOBk&Lq0>e|#hvR^0vs_VT^M1I>=N^v^!3Xs_1hg-p_Zx$HMeaJ1CzJjkCg(8ybJIz69Z*q_TyvYGM$IhT}PucINM(UIXewVX$wd~1`&eiSogKbBy06URg}X42rzLg%!f-Wld2oKGCDsS`%<h}X06o6rsFI2_eVGl6y{(Xw})uZ>>ns63<(D2F!AlE?DGP;oqZHjG`Jx3t6n_?J9!Z>klJv@gp7W^KlN*sJp7(oyV7V$hr5q#lF2$a6pzmp%B|_P%t;oMFC2Bh0eGU#?sbjrm6w*ZYQ1;a)dy@f;8MNjR_T|kOYl6OV_=3!WS!*wi4sNA&O*^zWcOqnwvG<#`Q0`vjaV;GTw`OSwXZY^`&N|L?MJ*VLFK3`b)lRC7EGn)Lkm$1Fy><#pcIC6j0K~IvT=5WtMi3M9Hl3S>_IN-nVL#U|2&H)S9Kd9l~%;Co-zFYb|W*NVBTbOOr8D#hZ#Vf6?a!7X{jFb8>PtHs8A*VGSNNFaZvu*ryrXRq9d&zT_u@)^$4~^Onx0KQhC1myFf%iRy0}6V`jTZA)tuzaz`Ua0{IV^YNb85%9cxSV0Tds6?A<HHAcnUm#z~dy41BT>ldCDHvRi(UdTG~BFi!H&pik$L!!!M;O=>`jg^x8Du^=z#Fef?c@dcFM;}zl7H*x8HE~&R+J53laeFri#Jb_bQyHj?o9?@bRZHFxiEEC@k(?CR^40Ir`J=A*R9Fk;Hkh3n62I|f0lcQsoEZ$sOGyd;zK4_pxef}egB4G3%6@X990c3XKW)(_Z7xiIyr8(-8L_(>S)}Zb;V^Zs1DDIzJ#0vuOW7%IJ^G~e690wKY;Kye<7q9|RS@g%K~B>%yjDes{rxjG!-QSF3jrN%b;L11hR^WZJ-QG(P;RWZEx2&6@qxDw(eza}?fYR3j_MXeFR}zev3AMV_)X-KqLB6o-{a%leO4&S4uf#8?(=43r~0f;5!a(Go83Tz+FYumnVF%i!Jxl7s5(L`_f{>N8kdB)XnccyNn4r{9!dPq6o!6u0ssE(Z(>4snQ@r03WGsLRX)zxq+<aEtwzF9hLk?_881D28{Y@tzEzOvnTv?%Cq~^`S!Md0ky;_kw)0jE+;uFDGg^+02n;<u>ae)q$cyP+n|!iGWBT)5EB1^tj6s~WJGs4GtCi7C;TVhW$3GC4u@SEBb&wASFwZhIgk#U2p%NF&BJeJnr38cijf!mUjwXt!u&W9gaf-?-+pd2VU{{P^*SbN!^Hw(nJ;Sh0BnMhZbw1z59z@Hx*!l7C^CpiLrd|_(L^HXBIkRM&$4m-GPCPez@ThIluet%+4BJ6g`-Iv0(fld~iy~a$`X#0a9Iz?L(?5fHH(JUa5`UT1ee^Awdz1qEYxDcfb5jQNrM!+&8>Cs9%<LUxe<U0O8~Dx2f&ywvh#X`Ss`$~_!Odif2d@(tMeKa*T|D&7V<`{j(bMVYEzeH5oN<nuZ5h2QsQFm}UhYI3^F8IgL!7#*Thnk^28%n>T~??Dqs-SYThSDF6w25!YeX1#eOEWSHq%GhqNMj^&1*&xSL%BBg6gA+eS-wOB#7PV4<NC$99qI|#8c(Po2!W%edh^RaRZdCqa46~>H*d3qiHA$m2tgYe9#w$+dZND2IyV+f$2WoWC^eIMl>d=xwq^lTUl+RDrZ^){)M<;C#2o@T8bge<=dbP{XD%4r>#4J4AL5DZ&C9{t4i8?T`j*SkuD-W2oHHVQ8*o`QX99WbrOTkmsQ<yYDuBVF6lkD&D|;Rs~3uRv#&))X@AQa0t6|)uFQd6nq{n%D}Y*?w?OCq$_j_ObyLI|dlM~Wa_Kd9aHhf5ku40=-aW?w%}4+~+m()*@B;%=g<0*WQ%27GgJ7Rw-vRHF6W7$p*ilwMZp$4ekUU6iQ^<1Z-4OVi!~Pxy`Q?t8*8jGVCfLi3WYFWu3V^LtbpRmDd*Q0dAb>b9b10(?Mn3#DX%y}1em84ei5UeW({?mt+S7|_nv5#o=|we`*V?{_M0^7x+;mXp7YI8q7S|T%tzPjE8W{AzGSnYvG8%pz@Vy>ALBwE$EZKsCVMeH;r^t|VC5@RuZ|?ZT?g#1dTdK8p(OGd#sB8pXHu_vLSaDoGWV6k<-mGPe^dMg2OP;|GsGcNOwU9aF%*25upn%<KpfKxcrE#G)47!HqtJ4rsEQHasr<|8dQj!WmXGzuW@t=dsVqT1bv-IJ_`P3uP;C<|J&<dorQL3?kRvtH=gwMnsQ$`osC&i+>kAlXIid3+`dmkE4=Ei}Nm}-?)r)CCdk<p0x^j31*3fNj(Vc%+W_y>912n&~P@J{;r?d5TG*gK$PYu*SV&iS2dTbN5X>3i7TC$IOlg$#B`{Ut&xx8yfG@4Y%QWXLj1tyc{kZ3tG%>|z!TA5@2{*}W&$!`~SiJbAyq+VIcNw4qN7nOhQCVFKg|E2+^06?~JFpqjNZJ+}$N;nn$OwkVT$(Nx!tGc`9Q_OI7Y#5vPZWkHRAqm&To>$S8+@?|^=_qh#zh$RXt%V@^<#W3s2CvbOs#~Up1hl1bL>#+Xqo*E-pnpw5Ys_M!dObI1!Kv~G^P?fjRah^^I>^uC0SMS6k-yp`Va37>SsPu>^V{ew+*Dk)BH&~q?N1Dc$@(+Sq9W)gK>~yFPtp2x-8d53jzY*CsE8tBx1Df$<4Q_k%+T7T7S*5lQWfR&Zw7;&EQ`~RH*hqazO0YaR^aLi~qNc-G4~Hx{uY>Um)nbq|#DK(Wij-1F_CNo})jI_V!^K*nZQC}#wr$(CZQHhO+qP}nw%vXIGjm_&)Lu`kQu`%ICABJ*&=WK8N7CS&?XG+Pdu`0LGHD~<hn~)%RX0oV@R9)yD3@lgyuc6Xj6WBH!Zsj=byZg}<!pizGl&-_^~Nd@2!|_=Iz#qDeeKx1TEZqn>L7i7nX;S&xa5%B;?d(}&6=PMR3gFb2OtiADTpe87EBpKUf`j{1**PGp7?h6W*gx4NjGWa_IAcydsUv1YDGHbtN{l+)F%nu0d_1n1V5w8navu9^HXi}!aViG4%nN4jPwe%zP8+5;(xuNLo$!C37+d#cwl%Et?Il!GS0vD8u95Li>iOK<A7$gixFn{u_0|G{ByHt56}~<NZXA@b%1wpzVnS~;*Y!5Ao#1KTEEdj&4qI6&REO53OgyNd=hTgIZ<%i9lq*`aUTi7iQPWt-AoQ`|GB!DQ!q<ftJDqSy?UVtV7#+JC)qA2V!%WrD~3HBX`SsVAT@abempYS-OQQ$G!4h$^FX&nTsE<9w*NJlhphdQZPHt=kYZ!f{@5+Zy@d6?s#Q&9Y1o5cWK}7azvn<7yLqIkr8Y!KG3KJjy~w+A&p)c-`iCqkWV2Z1<sO@I-On8f7|36C7Q|1+-W(OHaB}qukPb;kLBBrw?aefHgUKz1P6?z@3;+@)iQ8=r9l}_#g2pLMRC)sD)~)h*KA;??5UL$o7CT*6`3NaU$0P|9v5>2`6=Y0JjuEW`GI=2XL)E;4T$i)ht^GX6A=Y|fEP-<yj#cH8e*tR{E{qY^>L9Y!<Nl_}L1MNFlV}Jfe2|@)n+{W!BL54G%&2H)2GTGa3E1?qoCd<BQ7qyw9ug}1@>=@%n3-v8dj%nDmm5=Xv+4o4&e=P(n|X<>69isWObbVmqO+2~PGFX7q7Mo_s>n)VQo`+f`P(t0-f;qP8ePF>l#<_!i=koLRK1!0=!B9+xXoRVS*ET)9%$&9*ldOPQV}DJA(hd0Ac4bjj6{aptN5XFikWgJ-Zf63HE}zA`#@RFCQej*$dC>yE2g|&K*QuKqf0IeSG$MOHHU;r((0oTA3;*BqPd5%;(-y_ZE5KvS@Zdf9aSS$Is>LXFW8Zyy`qyf7Ev+LY6r$NOTo9(3cp?hY!;;1a@~N0!j<A>zGNb6U993HXgc_BKi*}RvfMBy#@!m#O?cF(Pta{LJxBrpPwo{91JW`JP8k`On!F5WL|&FV=iwyrsj{qA&O0nVe}*FHeH1If1DR5QBOR9q9zh(;Pp(lbIC@})I~}}FFZ*uT1hA%P<`4h<4X_m2sGe1~?CEg2*07cM-vwQUNDHBq<GHcd4Em94{N;4F`LQ5>!Umy=`<;CpqY2K$`M)B=$m(S1g5-!^Lz6z8EuOI=aacF(*x74efGS`)Ecnabq^^QmT$|ZL!AcLJ7L@?W!xni+_wq-UG=}U3K=tlP%Wl2t(oF3P6!~0zzc1tEeiY8S)tadA5x|ju0@;{(x2erKL@`VB6~rs8M1)Ee)-tP@>K7QLid{^@q6FcG`Aj`vS~f_mR{g|EUZstRq$yYa8LYn=D9TA`ZMJc`Cqy8|nt?PsB$pywS6WY1Srw{keZ5mYmWnQ9zczrVA#vVn9nYh22(>A-52WSs;xh(PkHynJ4bGRAJUkc2OcnMWzLrRYW!F|oub-zDcsedZEbRN80#6lT{G5nWj%tGgU`9}v$`w;FaQs<N#4Zm1z?UCqZf62uh^+9_k<y^*VU<_7P~;P0Ek!!LE2>0UiQHK)U9vMDxE}e&uzO(q1uPnrx-U@*JmJbLLABsA9d-6C1|p5|^}C8Y!JXT>(g3`2*e67lUL8+L3)fQLI&-hzMnC>NEVf2d0+c$af}2}^vKd)j)57b<`X{=`pCeg&rbAe+s)&Me)g(n~x|&3S+ZI86rb~Lg1Is%%(S2D2R%Ni|cqR8tZ_C1PU{>Eb^Hp>(8bT(%H^u)D5F>sCgOnyTmLza{AWY-J)<Y~S&t>75Lv~y3N_!S>A}XjcIz!yJ0>Aw=urhAHfE?`*c*ENpngor6*>@%hB%V6}w%O{_&-nT5SgUAz>S{K$NSOP^tJJq>pS<BiJUhQ?1;t<7ja(mOKfE!YWO^Y^axdmpk$ew2XH{v*zFLdA2)re3gBxzQCtsLXkR3<2?%b=lZff*X9OhS_Yiyxf5vJA@HFomXG^vjCU3iuBc7|zIy^(a{B^h^fmaJ;67<@q>KSMknT|6*Dm&Xm<Y!n#?HJN4SmDXw4^&plaCYe@0?f+xE8UEtT$YNw{IW2C%fZ@G&;ExMWfCrU!beJ;(+7ypgdXbN-HhKD;P6K{3kwZQ5eyjyD3O;{S6%}S%Pu$7|d$huE)CdiE{ox&S&wCT9P?VY56@Cu_$?&!$xdl=YbN%8<vr+$oaHqcz&69Cv^0S}oT|1R*dt&IQ@2mCyk=C5Ub-O`18E2J#T-+QJ$90lVXq<C!-|~9HIUJh)3iwe9*XAVSw8$6IYGfe;*Jp%{zJ(BPg8HIVkbkX|Vnk}oPBMl5<V?}J$jiCR=)60A-6@li^%^#?HP48SU+;SRGS<NB&b6>Ejj!ojtW)!i`^)}Whx-}+7V=chD`n|YcHs1A`>bMLt2TG|O|a^stT|1ww9lsHb?NKl$k2A_EpvL&i_<WfXASwLw`v(%^#33(BZDuA`^Tx$iCWHF@hD=<*a!)@d7z4lFYjZ;A>NFp6e~x1SPpDeCJGkON63v7&bl5!X)q5VWf#AfA>jK~^D^#?CIiAK503@ZC7Wr2kjv}y36}{S7hK{0qdWqd=myT%%tIg0GS*09$W}57)V#Nv5o%{)OWJjls~6;4%<d~f^e#af_LJd~O-9p(bNW;FnUW~Cn-7s6{>~ae$ckoqoE6c(oiF9;vL}{UGbKnHsk>}-roixATt_2`G7x67`y5J9=OG?CR-SbtChpf4BDPYu`6BZp>OX+ZC|<Pd`;`%`JO&UNM_zjvV<wc#U05YXT46#PZr-q{DP88uwHM>cSDc_J?Vm|!&W}PW5Yeh~!3w^ch6uvHREygzsj}+!F94nT%?Fd!R+JL8n&#Xw9@3ofZQ6~%tpR5%2-JSFZGm-V@*^xq{S~g1p$e@yq6~DR)g_rz$2SE`pbGHiJ^)N4$LZgndu(Alo2iXaF`;}bH`RgKC`bGWh)|<b=VarsP41W{RLU;}q0Yz(jqwCneVN4K=JYsGM9Wz1>3nyte>DGCX~p)jJQBA>Hb_@M2nd!hkf61as((#Eu90wLvR#Zy{@SS&wws`<5SOas{CZiCtx1`Hy<em}Zr$1`_*)rx!JcFVg4#62di`wbPt457^SXE3+Wq4~X037;Vds|f-hJ+39cc*Jb#4#bX3OVjqnpo=@xQ2#W4`|-kETjaF8~7oKtlomApXbN7})FCJK9;A7&%+m+0xp3l&IQR9f>0R*6J~|<NqtcW|PgCZ^!UVlbRez2-@F8HyRel+S(HDGIUE3f4kzMnuxgGw8<w!@Ho54*f7KMV8+_HEL=|2r?Hysk5Ac?;7?Q&<&Tx^{71~65|G5f$DK3~ql`UI%Ew`I%?^R4WfAbVo!W0G5HBTSp5KpDd{~?gjCIip%8>z;Z#GTktbG>(?{Wh0q+O;hV*`|D?CId7qo*tFcVaJAnlIhRH4ena9Qvi*Rx5AG8pMi;Dc)AjYU~GG$>0)yYHky+@ULchQcywT>;Y5LN+=!yT>f(y!9GZ_A%VtPg>r<|1RYuv(5w=ps{*Bs11l!5WdU^fq0=#?fZbHMgcU}Q@&*H04eFCv1=R%DTp&d!S?=?4jyOV3+_rT#(zTJnB$^ij;}J`;=&x|OEDG9LwA6SAJ)`4h8(y@(H75HD@PYIqt({5MmbR;|$A>XTv<cvSDNQDiZmW)E&t8ZgU-|a&PEo983*J8L0u8D8*YoGPfS@@AAMVlKl6TeWT{E%s=e1}@rmCk=@$811+vU?(!cC!c@&oR#e^N8zk{CHgsP2-g-cf1)8z^_2Vv;A1I>nsc;Ez>Ak;(YhV7q}?`6L|+N)McXz)42JiC51n^4k#;a-HS%5;cN#<G>95Wn1mzyT08ky3gCU{p%j%1{k9dJ}I+G8#3L#1$v>a<DS3!YOF3UkA0@B>kHn`SL4|uGbzUTiEnpees^!_@4SR#Q)<YC45)T@M{3ScmX*Y1{Rq40B_=pXbhA8rHCE_-ol`n*kW@|gkXi#5Ok1v~iMOmR3cSV~H6|nM-!n+Ip0Jo0K=-LX(ZlDHPO5>>)W(M~>HD8?Yow}9@<YU%J;{qBnJfUNa!z)+r+Vw{)T)mLLp1B~zd~;Z;|T>>+kZM5%wSpb@Zop+<ulGyn-(Jl)*+HvKu)D9qlXGm?1VTyjP>O->h@AXbBWOXos&<#YZ}as|HNGe$+)#uCF|027OjFP>q^r`h_Q?`AmJopeSpCc;%=_`9=~)z{DAAn?0Ie1T1?)v&NjL}roPQ3)<aZ7ZmQg+v4cFc227P7mw!HP+TJ}q@o$3L3O@Lh-6%SDXg+7k3Y}%f<?V<V^bP8ZG7t%wXdab`eYf7%7(&F_clVxYsZkARGKmeqwQ+-c4m7b6YEKL7yfVT~U3Pg7u9S!_SLA*)dLn({hK_NHn(8^~ig*^?F8=FPzOjS#@X9c{y})0?zb(UE!B4rS=$*Pa@43F&_T1AExmqg%`~KICgc2Qo#(4w)02Ai_oBy`1CbmX;#vZl?HWo&-M)vkKT7OQ3<4Auzej_+_x==`l%P)>z65~`wrbar2xq6LvyDMlR^#4#n(7=%G*#5q*asfd>j(WXRZA8Lyb8r89a&Pr#XU9V#FV@RAZQ(7hF(o#+GB#)!(TTfCWotxB{jI0T)7}2-v_($d#tHsdvzSe3$@Ex^TRCd6vvU%Apo9JX3R|RRY{Nl&rESl+kxnp6#RlR;Yo^0d>LJhENSw)^1z2;9K=6M3ioFJorGNE0OA*wPwKL<4|Dv*%!w;r=`c0!uJm+CDKpk&kJU$S|a!*5n)ERh;z34?_V7$`B^<*xauJwH{BQvL=7m1}hp95VyL2A-eAgF=AnT!<eRvxJ>bKg{P*MjW^UbUjMge4K>(r<##9#DRXzqqwA09%~5UIxDJk1347nq-gyMrg4(iESyUj<*m4;3LsLmr1U-won;x=e$?Xk`L;)J<r>Gm+-%Oe%tr1T9qBAlc{*A@~NS=VTCeO)+OO9DzcX=C(Iu2`DL9*Xcx6{dq|OLxUqAtt&G8yb3Jo~O8>PdVlx^BB9R+o&I%|R&V_$3DHB>Fy3>JFyLn&=(?V4&5!?e|Ueu0<(wTu~Sza`aI~Z%92EYG<yy>I3tb>BOoN>O>$ikErM-Cb$X;nq7?4`C2L)z$dMUF&4ogPs!Zm71!BJ4U1+#Bz8w5(!T;b~I2t!g;AA8PTCZITkTk*P{NI#Q?qW(}%q)Xyg|J$uwOAxpy&HN4+C9+>_CbWhjSm!NdmZ4Jz^oHP8U>6z&dJY|DbpK#j=n}y7%FdRZ6CTEp>mTGl-1yYbU!H+4NMUin*v9RnCtsi9$D0Bdthl7~~_2x1}X@r9Cv*83gPJV!;OfryGJ*Qx2R*F;8hN|C|Jg{xZlM~;dThTjtU^p8H(N$4x73oRfaf3DMe!U>bCv@4!F`91KRLMM@g-!)qTBHRqxS&rbF%wV5{xBouLd|wEZ+Dv9P_5J7e#W7L+GJ)1-ZE=FL!+fG2Mg$72FeOGi8>qS_umu+5dXDh4|)c=wM>r&i%+(0N>+jN2rW^+UGV4r=+q+-K$i36;tV7aNDaN2d>?^0EuAbn2kEnW6}EnXeotg=;?C=AME!U)-@cU_<>bA7K~?e0AnTnOf|;0jy8{uw65qmvGty+?dg|<qs9T?g;da6<jKD5(&Rmr*p<q6JPHd;T%=GjwPqd5r6gm0|tew6at$Y>j?73~<@}9TrIx=wDUa$ivX6&?s<Ffgw&4Cj{6yo+D;Si$#$il+fpW961CfvMeS}p(eEZDVru{<i`d=k1T>?|1?{wjbAz{q>joZgDU$zZDSlc32nCr4W$Q5F*HvH{~SnFc5)4?x^DTk~=lDOc{N*ZmQVf(M(ncvx>~<)Buc0BhIBmVJa#5`-WWgc*C#MTpQ(zs8czMk0vn1%Y(FtPRe!4<2yS5F9RI%d_-sTD$uGa!mu|1Gn!z=zMmmoUr(jUuU-a&e<g5_<=#nRNgxIvskoA@-WuYTy6dX+AaaUU)1_>Ei2g@%Ad8MWAfnnnXnbf6*#LvWg6$@KXK?D4=Q!vzJpNh0>Jwi;axVkFI3oPP(W{$tquIGk?5+3Bdbw0V^$=6AV`@_%BYDDQEP={kFP~=ZYCo@EUK%|x<&<qN;Q9i%%ez(Gf32IUJotSAE7Bi^RgL;m?`hAoEodo)l>;E0_=k|0qnaT$PusvX$*+==Bs+M>_%)$>gs|k&iTD>;8^vNYvDAH=jk(=WY=;2TUr30F?ezirl;*q;aSP!9BLC3t?|99{fbC$`e>ULAAr#=5bOY>nakO5y1^(lvDkMIQ7@7)6Fsh1KLW%7o08`h#rx;!QTV98j_ES`$<N-RdAS6dmp<gac=gH~Fze>?Q}CzZ=l{H6^;3WXF;^8y2(cEGigi3twQZl(IkAwTw3*JST)qgiRvRH^e>tJQJv#?HrF@$E$G^j&W&Q#t-A~x7jGoP|jyaoeuBZ>Cr9$5okfX8B5m#TEiXm-%j=*~9+HTg_PzadK)y66`)6A};YZ8$<YL%OxePr3$8DnbS90$M{aSi!Ym-GZYYtCZoxGVqqbP-9iTN>N&i=Caium1|*uW6V)HZjx=O@<Zo?CQ4M?Q`DzCXJr8_N`N;i^_umjP?vPbO3Va;Bnt=W(0`f7nYI4Z6TD=zCLwHhvhB=Rh|Vl$lQWcSNB_VP}Xzs7Vx`{bAB^B)b7!uL~4xW+8Zsd`UY$V<^erKcQ6T`F;E{D4*u@u)e%Jafr0oG8&N@+MZ(Y@NyMdWukzsHR;uA}9?3Zo2%l3n8>Iev#(A#YP@@kpztH4<jv@_T4zFR#i~$QqY+D*x4x5P#GXhePr5#=-k+uaN`$;hUv4mY9Pi#^XLcS5F4lTFbXGf#nvaPA%)m`-m7gocgp8<UBQ2!W?_1k(25C^B?6Sn`I&HXdjB}<6XmRyf6=Uh2U&Do^GA51_*A(AGKBLM3`mdV~d1Ts%ujpATUG)mky43;a8pkZtZ4iEyfCYXMPEf5|<&HoHo4YeNP0WyD<0S=6DAwvDtmh|)h@?|Q;SaKhh=>L(u)hRei;E@R|&>TJ`RAP{iCPE#9Kyp~B^N58y>0@S@K9>SSHM$$%xicDajy6q{3cRd}VeUZNyr8b*c8cnerqOVXFd_PU)q`sOMWiZ^!pMLQXiijo)v9&Anol}NP?#`k^3$lv40$>H2DQztQi(Cc_*I~}#+|=9)2R1o*?O|DG0kdoKc+a_m0Tj-VcE?Q-}a>i$sEOy!Ef~Oq>`Lg$byVP_vt({U+vBRTb;d~=gnScb{6f9-7wW9T3A9s&QeSxIS#Lf;xa_tMnqTbjg5mSnRlK*d)E^Wdx2vO$q40(I;B6F8nl^t9+xxr^=r<3?R^<L3Xlo@bASQ=jUzZ$J2Jv(fe?RXMHCTCci{G$0*{3MMIrr!9V|l2-UH@PVI()svIau0wunwHWw{YbJ84k?((V{*FCGOHNjK6L2@XB4jGm+aAdTkg@18u&205ts2smcoEFt-U7^hO6xD=OG?2sGvVP>d{wD*p+%N)sM<eLH`_Z!ris>Dy6xxEH($O{xotmnh=iuLhCsrFf2n*BsK89}{wHs&~G>pB0t0zSGt>BUwIHk~QAj!*o}Ojz`4N5XvXYoKM7kk)_i^Nt;&uRa$d1hja$NRJ+``UE4QM}9083ZIcI=;_z65$K$EOV8|13zyh*l!#IMqxBZMUm#9zvqUhLKqr3mBven*i!5+T*7Awz$#h0@%LM|R4ZKgGKK7i#JvZmfiPJXA7xV5r5ZrE!3a<a!PHd>8a_9-8R(U;&_Zi%idC@IUg!&xKq&%3diim5gaRby!U4Z&IOi9w_v+h5VoAnQN%Ws(k)t@(Grk@d-H~R^!+cvST7D@rr7-D^{!{&G6y)lcx!^v*z;4IiuXhgN8FrJdzV>hSSfp@EOlXaZ_FJG!k&*uUC2ChFy)6?K3`Ih1v#K4yJ$KB_`_FY{p=fCEF1lH|WTrtv}WD#HJ;2RMp1#1IhWjvOg@if#;SQT5cW4X1&LgL%tD#py64ododlT}zDIEDc(B%LR&n<Dv@UW#O;S%zqgSA)8|)aQljU1W6EdcWgoo!aRVP!V?3OXI9%vKi4W*Ay#~=o!wI{MHs~JZ21eyk{ZbKi7-ikfpKn`;rQKK`ql&b<Fnys?K?MkMmUuxohY_fclcYNF3_<7WZXbPP0%>q5s;|PoJg~m>E31Wobcz7gEL|?7&~2;xx%p@OzFM?p)^jo|o?NTk-X-jT%mmp!K(&_?i#Xw$nY}ujbi6K+5(gnz!F7Ssuqz&~jY{#GQ|xcxfl-Vl0C6^&X%)IlEa?0TdBA@GGl%3ojL!kQY}msSZ~_9|znw;v=~1HVu{qyiky3%Iq2=KiSVw!!du%G(e+X#c4=fe~8_m!}BOYB6@2T@4f~4g-L83CBM1qK*m?#rGJ6fv;>)P$sU|nVHJ4Dk=`#12t{U!xsi@eE4L5NtFVstR9}xPW}C0dadG-hBY%3<*4-ArKEM448l-n-8q_Wpx_%G;m*LlFegW~h>fU>DNCzeAT?-}dX9EU@k^1CPbPA0ehdmTJ1VoI1OXgKM8m?1f3e^(jX=~P8LLSHl^cOI**ng2I$)X}dzC~Ko=m+l9Ewaq<ihl++32@djT#$9>210xIb-NG?E28()hMNWtR&Og?TCIbeTZ*hhYS+#BAq}?LZxvgGJ$F(TS+~S(YljSxCoTS&{Aj?qI_ejN?+YQ7ajt9FRRV_@AaH6uCVmh!-GPq4;m<XZ-+~DWsK50%qgEtb87d=xkSG(pH6rcMz!UlwFEj5xw`64?`#I+r-{hE&WP1@60X~k`uKnJ=UBWU#M-;=6C{=iWIQWtwyzY9SBpy^!kaGZ)q0HTkr6yGTyv+T1?h@SrRiQvCuLiI)S+`r9dG;2Kgpiu(u*>e#GJRU(Z1*BkZ+^eKqzCehoi#EQ{emaOFsAkj8w9-rU{o-Utr7_`9s-mifdEP6GJ@jJ2s2TCV@G=d4iS_Oyb4nxvWQG8BMnRz``VQ8c0;0sGYDh^ynhf;*TtzEMxoC?%sPOlk3PB{h{h;#tzulbnI&;3v@8Z{!1v-##t@6Qlw1lV<-oJn21^!G3=&!=!drZ;daE+cO*F|9O_^P4t;xW_|8bC7$tzVOf-Rm&q|$xioM*FGd4_W~=Cxqfh_H5?(W?%Mr+`9n@}uaugXJ|M7mZR0k^IGxdN2=N4@F$W(f2)#Zt~@2jOWpKmy=Q*f`#|m6jic2<$yo`=yz1>>YhsaqW{IIG?t5BM3J(rMW|%6H<Rp>f?}st<SEx&<2q%hoxjQb8}Mp+RUBMb7<i0m2QH7sAZYZ}2f|cG8~Xfvq$Kkh?$GOP@YEieUv-VmbiFm!d4Y4$XPotYQ=eJSUp=hKHT+&|q)6;9tbWS55gqH!1R(GOed8=?q>bQ?{R4=H<D~c_be9)=23M>}<dvFJ2Zu4mt?d340I{vrEsQnauGJqI{c%Rul0!xp+2dTSblE}j<}mEkyPxA_6feuB%c<UndinJl<F0cbTY8TF#rx;s^7&aK;w<5A^R>oEz1`sx25f-#V=T$+#D^|i4cxg_a*O8dPJ-ByJQci*m58idep+Oa4Ow}o)wv(4p^S_Rnu7Mo!z`*vNYPx9Jv%K;s(Tf`trU6vIoCKsoDHClJvjR5Ve<hd3WEs@9GtdG!uBg2O}}0#y28yAf+SVk(ZyLkkW!pBLQ$*MWZ&@(ivyg^tukIFt&5{#RV2-e$yDn?uUl3Qp8s-t%mJB6t04uZuN^oY>oD0evKPmyOzR<@O@&h6m0kP3RO^w0N9Hs2>z3$EO$w&{u2*%{fkuIH;=TLamq1Bd+p*GSa(0F50<0>8hX(|mTNFl=^G89KEjb@H0N4HM__KMRRaSeGvf>Tm=xZ9%ec?9dbapG)olv|G)+QjQ&Gpz0Z<NSj2=q=)dJNja+b<Et(!tdmI6CftwoI1*?au`?GSDdT0>$hgP#huWpq{Ur1SfTVa%xb&QPFV{JyJpHx37O|!S8>`9|?9EF71c_0Ii$=0MP%j_KtSOE=K=nyVDJ}mE9KmW0zlu>swH9s;YGx_BE$TK(k97V68=FF%4P=e>~~ts%YKi!7%o7-`;e*r>3joGhY7z)N(52tUY@+8+qTiv%K-ez;+y*aPw-SOmAe7S&#@;Sf&kKWVd*Tjg@cuuthD00?L`}?U({9McAUrVR43jW>!tcs~MV@bQDvm4~;NjQ3gbC))-~9@h_tz(EIGt;LJf+I=RJlB+Q=8KCxcO;KR2$WSlQVzsF^};cYuxV*c=iS^yti1xHX`P-1L3L{iMDv1ao4gl%@D<LTs_e{jr=S9>HrgqFKh!bC(sADSzM7ZF9-AFk-`5x`QSG6mL+IZrM<P2|!%<aZxRc#n9Fytwejq(lh0n4~f~^J^*@Y;V2W_00yTAMfW+yo~6apHW!Rz}C~z8`IYcep(as@5w|L`5YV^Sy0o)$3z(0>jfzOR?$NFx1T-0eo$PVX`U(T0ATIF#ly+OS%#0}B&ksi^e5vp`dm{SIXQW!hU~}vMWfy9SRC`k)Wk<MK4>=%40pr)y}Px;SwB@eMRvRhjNCt3M_+cvGGN$W582!M4>EMwOipZK#<=DdI$IA4L3UtoW_u_Ya)(hdu}pAvU_K-q1q}ljMl`Wrt8_V3TS2d@vntKZYBs|I6an7?Fc{=HJY6ZQd;ITu#fIP_bPEYvSiUcAuAb&OvoX7As6$uK!T{x-a7*TQI@E$AF@p5&Vwd%j7M4A|Wdgc)@hGMEtZa%z6pR%sr^=GpJjffxTbI<*`mB)HcOF<UX7aV$fKof~w*|JIo>|zmug2ot<A#9ip1_TzE1Jmoz%0j>jj*;8$@_LYSPb7uJFrApeK;c6Io2dIijhOs7bgQQ0r|#^T?w>Tyr2A2R=Cya*hyGK$xRuVm`x$&+XEG>z3}sCx!~={+#2~4h~@^!LjQLpnL5$|J!EW_Raap0DSf)jtd9J}S|V?};ak|gHMOLnvrlL5@#hrvuFvO<;EP$$YB~5kzb+zwt4u%gbuf{3W?C@wE|SAou?W;bi!c%M{-Pw#1x1X}w6f+=duu$L`J|8F-edZ7z*o>+#K;*qlsi1(hg#tTWK2x+lCDXAxk0Ae;T3n!nOpT!ufVH{KJ63EbCceU%+3TQwwsDzcKcJJafHE%;z=skOtp_w%SqWIaqXt>TPXbX=g&?`DUg#<g<HC)cKQ2e0q^$AdPHp5TFhvF_diqJi4p#G@aAyW+v7tP|FVWfZ*JW!Fxit<8~;Wc4BjtU{Z{IsK}{|JLUw(YNl39r&>XLm$m+f!xsu&*FG@$jEq*mkGDuFEHq|;sKr^WHnm2#9axE%SrxRM6VrUpmC~UL+6(p@{CwKcroIG|$4Z^u4nVv}?Yx&D<OW)w$)k(+oY!*5+=!8}qTn)o?ivZ&wXKc>wxVtN9a5)Tru2mTT*`HzmJCb{+qP2N~RLX^Gu}Hue%=cGI=b4V*%j3uIcgIp8^8<^3a2RhBM_vnL5l`uBf_aCrP8PJ<?-S@=-+{4~JL;NIv*T(;3@1J@-ogx#j#crhXPa&IiNPVQw!Mh_mW-&WqbDh!O)*Uu{>(hY9Cz6*CJD}|isnOLXkYd#jXD(Qh<$NL5*cK-T5-pll%xnu9zkj~;_^cpDv5xN{YxDqK(rXD1VoES%|$^y%R>?E;@b6+U@LvQX(ax;6~!))*2MlsEpphX;5c=-2a|p=9h09)%Is#|!?Fb!f5{q=;_l(<QIK3lntZTsjG}1C7BeeE77l5raEfnv`bg%Fl2MYBXAQuvg0Z^i5hCefq-GP0kz%9w_Q07fCx~2;NWLI<%<vY7{S1oXhOSEqPJXWNBM`q>$bp)pPLO@H+$D?#0mmewHgt@X)0nsXdjD~P)~tIjXM`>D&fM7h79nSo=MLTev&+>eyCMQtZZZc47`KNPq6v!SkTZ~3%yxA2Db!wxZ!Ryy!^#zHX5?9#n#iXC7P+xcm#hH{xhKxow%J3FXS{WA6fX=3*1eqUnRJxyP#DH~EzID30&?7C3cYMzss?_OLG~JColOPP^co3~<Gl%qcY!;qYm}2=O7_6kRvPcPI89=jqAfUT6$lEGl}Dzm=P>F#g&yL9L4QQ~zH+f?U#)=D5|qtS6T)I~9J_IQzDT@J1nho)n>Mmj+j26%8XB1_G&6jZoaJ9o--X!$qJW)QH9;;M*u5W6i+t$Ks*ZEo0B6JnuK7#=Tv^p%A|HT~_MmElre-N|7={rcu6T>uq^>H11oJ1NkhNpRDR(|E0HfYO1WgdxM1kd7R7xa@ryT%%_~xWFCg(p9!<&(3hyvWf4&Dl+;W5LjgxfoQ;adR5kgBF9cst&n+#EfnO>MjoaYHvyascdMWkJ*aH~-(PS<c@enm{70hpA=+)4N1n!sAu0YWGk276{3&>+3JX{J3QC3JURW--?*SEnhES4}3Z|2-?F}!jr&{cz)Ryj_|jc-HB#xG>8O@cnbYM4|=q|rUVu&nQ32DB~Cf(OYu>}(8%tLo3zlaMAl4}O0??@2!QOCBO!ada2>MehF5zwBQKe&)ONp?enKpSR(BZI0m%&g7gG(9Dh(C7{&`%Ueq&V53I1anL=YudT*EtW^&d)nc+Y+}Vci^x*11wv!Awrsq}fa$mZqbO`B&>wm;EJ3@)ElO^>Rccoy8<T>p<bKqA4Rp3Z^wCs5@uz8^W}>;$J+L1aTiLV@dryriGc5%b1pmjg#?G*tlhQ0mY1A0f1T*0BrteGIYvvVE;F}olhn(6Ax3kOu<zEa=LZ*8rcTK!ep1Mu3@<WXJ&mnZ<5RMsG7+1xynt$YRCFKzh95gPcmE-&DEv3qlx>Is=ge2Y2Tle%7pmOqH}$sGXTp^n&vgLJbDLpC@cPST6z{@7>-YN&l1vcXlHBVyr;4wQIDk3QO#~NN_U1EA8!YaZux^6e1TL7BDHua<@w|`L!W<L4@tq1t+hwB=YUQ!3J9|515Y~tFttwb+>Rj>hby1=znuKVM!$S8hk_Dv+ogiDqVVC~K4~x{a`J+l;-vwoLPJ7BU9h;#6HN^g?-=6C>5$kODx|~aq9D`~ARU|nKOm%yRV|@g_IzK_fU&@y44h^Xgu1rb^xq@W%9gvjLKqD<C$McG8GZF;e_6dLgHnEZ&Wd>TcDYyp`u4np^qpyuV794MA>61re4i6BaGtz!R2zKc@`$Kj65{~M<fuyjfvvMRT}i-(-Z?aqhF(frf~+8r_`aA5xf>lH8SpX^;&E(8bpeJhTU&15q#thev6r92;_OpkKECdu5hpLc@u5*HhrOrk3v>)7Ee>5*;*-MaO^DYSH~jV;(Zn(n4HQP7A({ZW{EgrB;r@)_NF*8k;khkp8nh{G<eKtgUG?n_<ZG4<#wgu|^+?EEym_LK!mH7rWP)=w^6yQ#A01-BF2R45Jr+)lVh7Prv?m)u-A7j?QvYQF@!%UttAAt;i1GE-LGJmby%z1tB!B?=2Et&(!izZ<G*cCf1hqsC5BdM;ropX)ev$peZ$GE#f;P8?b34<o4c(|M@*Pfr^Qhen9!E;Z`=2=_(%h9D?fdST{*q^JtqD^O{Pk?Td9=~bJ{p{Y#68-a!`)~0aTAdrm~=(EWYb0=;7nuwN}>wL8gk^Cab4yVz=u_)=0?_mo~AY9O|eDnum1*yq}gB~4ero)r2<ue_dj!?6Z6~+wpXo#4~^FyzjSnRrJ<j#aY4qXVG7Ot>0&Lz-pcQs!Ja{a;Da2AaXcAULEy_xcJ=3=$bZP0!I4&Li)|5K<50|hq2~h6sTL&d!chL+YPmck6jRT>@yq7kktG{fi-$@;U3S?yL=fnh;?&2V6XI1q9JH0t#`~YCx&!BdO%GVMA9Jk`UcZ+r)Pb&$@QVb<pp27{FX@{1yjA-wTN)&sY-*d+b+9V>Hb&Qd7SeO8@3!Bvg+sYKwsLx4%jVirOkLgoPWAeD?W^0~bOoiHcuV<WUapNy?8(Z~Y3v6(%`N1vFU{UiAT<HsI=cS)!tduBfJdxL8ln`$SZ(3DR_!qMi$l4xz@^g!t^PaxDV!)o$w&cqBi+s-0i)#dKL?*tvb3l>q9357e=M^%@%dL|g;{n_F#FgKM;0O8LWG<v$hkKU((U_jM6@)UYeoC6502qnJfcvqpi%{+`V7@63YdKhb$8BV1FEZumK;v4`*JPO5l<;7HmD&%zvhG};GJ>GDHIpY3qk}~+vSKa#73Dco(P?ykl2;GzmbhrVC-ZXJe$K{pl-`&>|hF*d@~9fiJ96@R>~=I@?`0!)+MvSG>&Z(C|cB9i)DRC%Eqb1F;5(Cw)V=ppXR+5$kA!BAXDjE#l(2BURR3#VfdakYg`5%TkcAb-rxi272-D9ocIV@YHW~US<;mfyF0m~4n5ax+AiV+$Y9MgX^$4a;x^QEc8LDE+Vk+D)f3Kd?Lqgj{T(s)^2FvB{jyuG!}r6@<pb*wdq-o!EwYfl+X~&K64MFnS&baKB|FQtzt{?2xf+wBdUjIYBwCV=yY15t<wwTJZfvU+IOVO>N{i9s%zLxb-H}AoTuSyZ*@=**Lx9wC{gE&BSmjLFvy|PkC752Q$r?PqG~gY@7IiAxX6Q{KwbK2MT>fzaYcyYpIX=7SY7Y-JD5QH3wSvY>-L}>Cp9YC-g@&YGbT5rJqB{nHN=T<`f~}rS>`iT_i<OPF8)A&hBH?OM^)o4jblPY?*bB^$x0g3<s;@9xW{rV#&80wdbpZd>8b860@=~mRbUH}@UH5FcFjUdH_vR=%yHv4GYli%r+2lcaHCu0%UqN0dH@|2YLPA>F86Xldpz+X!<m>7LGI12iUCn{Q%cMjcadvgTs-1`1zm;k%VKJgG`KnKv!6M~K6(ma%f0OL_yuTGxg2fPUNxnGB4}cn^m0K*iBOg06Jg5_3(n6kjK6g8Xg}-u3MLX1r95y?(*yL(QOBC(NTCNIOEw-yk>`sTqf<ee#kY<b8t%?+*Ct_8Wk0Qz*hSSfjr#EEl+ItLF86r4>*+_AobfPipk***ZveSm^!`p4X>h9z>uGLRD2x{kHe-7u=-YLU;02lyu{<s6^(&aQZfXMK0uw$*B!e9`c*N3tN;jwbc(MjxUtGusNQBB}Kpu5Kyv`>xj8LpjG7nrL@Tx`WIc~@h*Eg&2Z9i>aQP^AyAFqv$ku+Qs5l#Nw0>@rfFt|?icRi{edXRU<_tZU6BN?h$Yi5x6aOriY&l_`UX%Jnw91(i|e7`vEeEg*09K}&Q44i;Uwu^*ONz2>EM#_mb!@~?}OBmqZCqR6}EnDMeoA=D&@%UHFDaaLLKvf*G)xMpbPhfMs?!6yTNMkF&puOduk?ZspfzTf4&jTHy99vuOm8(1If^SC)zhO(1*&$Jy@{~H3wErNo>{XNJO)pKXpifVbKw=Gw2gJQBldQZhm0H(=$@zpc1pPs21+qj$6c)Zq>1gw|b{Fanlfgf8p4`ncitC!s18c>-ZoIYPUHwZx%O~qJC<`}Z1MUmtze~@yK!y(Dw)D}`V6Dt_a^-}$18YZ9BR!smLa|=i;R@E0S$D#;_bzO{_;XAQSq6v4p8E~kHeAZS>)!JJZ4fSl+LDwF^8YtbRUunU#e;PA`dHPSsg8M%MBVOiSCjd+6=GX_o)uy)K^`f@;vF695d^j3KHrUaAxu63*+l9O<B{Jvhz+hnJ0e&Ijn7}N-lEA`&(dg8&%@sB1h*fvGQ0iM6<v={Y#aOln9=!|WVuIn+Ot5q@4qio{i{4xd73}5*dv=1c#4T*-ZXE%{scOLzrsG8hzQMj}a?@p>e1)Ox<75jpAEs@f%htbu^XByT4~^(Wj$5bNZ8p>yTdpa6de)2(ZKiJd#Et2;aKDiMufX#vcxe&|1ONaF1^|HhAM5CBV{fGA;9_8H;q380lWwLK)+V$@POdh|Ted_D2*P^_@5ob*dfRDr0wmHz#jpbWc&>fEQ^AByn-Rf=bL&2(lPkS9bazWOZ;stMuWc4@501U5aTb0*ozh``d9r0|bHI4Iaek#cQ+Hd;Xn3(-I|tBFv3X@XE&rM0FWtU8T{%;oO`Tmi_u$Fhb+K9V>B*(-9F*)&h<&*5=GJY=)Y<$}8C_qnjtXXD_E7IZ`wA-r^^}B)Al7Yq5@XeA$ylrW;;|{RAgA1nqOnvfdbYK%L8HgK@&#$7+*Z;kHB^>*8Ba$Oc-(7B+5J1-ilMXAx1PrR(o!IuNUh@Z9BK=rM4gZvjj=H;w&cdl@npASYbZxd@q~gODLLZun*WPFWr<$TX+*b4X?aU`X@@%wG8nkrXv@-G?iWkYf`AV_VCJm5w1<#hAPZX9R&qBD51WO-m_+H^k=H^Q2v+La-eDBUm`9aVbwEa(q>w}{0(cdGOdu~2VzB}aNkgd(?0@bWfV=2TtwGDrPsLPHsJ#Ze!b8FNn(Kg8H7B9yEH+oM5a7U>3jG_c((H^)gLud<PT-ZpQ#`K#ox+X=HNiBHX3zM(#bAN5o>gI3<$88+Ds)6&@)@P5n89F`ao%zrnhH8dNHcu{4)$AJ`0oz^VR_&S`5NlgmN^We5KLh5j;VfXo}uC8WFL1?da%{^PK)61!3yflg^@|_F=`;6-(d)mNWi9Rbm{B`h9g(X;PNQsno39_W)w@5kSlx9vn^0J|MzHtD<|^u1d~~M2N(1&S9w?mO=zsc?R!9CC19h)j+t~}{~Pcjc3@%tRK$Lg`4w*3h`~i>jGeYp(FAfM)ba<YF@!lxp*QTy00Ab6u*|;yH-Kr27PUqI0RUov0RZ6tW1Y+m98HY%OdSnu{^z))jpB^$7ClVw6(ze~B0w!Zfj_*)VIk-$nN*W&KRrgJGz*aRME)@u-s^qR^>9VnBY`~rL|mT&N)1b(P0M=H!X-Q!|F=DHgc=@g3~<-k@j{TJm%Uf;*x9uMWUEKb1Vl5df%Tce9~D*r$SLjx|75|-?c^(oXc~?EMK_OqK3eHLXMvUY7F-qNUWx_v5z)M0QHs?57Z}tnyS)y--ty06D0{85oQv`*<<F`Xbh*k7>QOYk$zUOcVPz#ONBKY9JQpKOqrzj}Ii<BdoA6wwd&T@<ZZ%$?TsfLlMzAYzrmGqOpmY#w*g1?;5ld@D1JWpM1jn$Yh!fdu7^9rw-Gc+y6uQ2ero0VDRs?oSI*CAcv5WErqWSFequTTYEwcwo*cOX(M##BwzEo1brk)3bhKamev@L7YICb7M`bl2fCY`%Sn^yIo9&f$Yx2|hN7(CEbxL(I7jk145b(}~g4MJ>3qTy6g*6V@vE?H8HD;BW6!T7EDB+2DvfZvI5R2h26gG_YT)8Q~O<b2F{Rw%W6&r32P{PCf)W4^s}_FYMsU&gb2r+rHJe+8=rRe$g$@a!!<gk8DA|7#?-r-7!<=|7LJ0sY@ou>ZfuEo|*woN3Kd6m+Zx=}~;{>cXKGV`o3ARVmgP42%>*cv92jSmsMfJ93;ZeZLaVwO#1d@fLTJyyiIGc6Dp+ts)diA)h=26AKGqWS&*`Qb<#SygfrK`|U%3OzwhL=D{KaD36rph(TAeXq>7sa;7=~o1ZRCbs!iY<2me7^4Y=-`xC4q3&B`i+tKi;f!`uR#r~^6(2(Iv_gpqTprt_x&_71{&hr=M@~{lrqg@@YVGkc7_Qy_9U!m1?gy+IS&3jqZ0in~4E<|6{;-dAT5+fd9OG$$Er48VYDmK!DxG~Da0?E`yk}8vTmS@<M-5n=TY119TCG(_UiEcsoT(&T#_bq;zBIn72BA?xpizK->J9|=`dZY(R-Q%h74w=(}M!p!@t|#S0Sx)$K_(>TVV<*XfM(dh_I8lKLM(DWNix0onb|aAxx6Q!Htc$LOLG!NOwAO0zl9uMgKLmx<`u)B)-EozflrSlgztl`uKS2Mldzv63LyG?^CGZG<|0f}x9Sw|3^o$Lh4QMT$>}+px_MmLWY{KrIP{`H{m_F%VR3srS3^VyR(@W(g$Gg-vC8W&DMXkhmct&4C2C{>?Uo{g&v3#Dg_`Z*`?0i4B_I|$R{$953_RIX<X7<kf{GMj;cmHPjK3?1U-f!RbelO4Z{$A7izS8u5c>e}Ha`JxG@_rxD_kIr3`aatJ-aq*Mj@(|%eiPsDe;&-(sOk+t)vD(sQFR-Ls8pRtB5TwiP^lV@Tza0sppD|=W^UXWoNaOF_zp3unOVfTPF-=ZIp=1@VhPv}rj@*M;?CS&onk}B*K=F!V&rAjL8NDH+IfV(l%)^i`<y?O*FXF`nF2#NqJ4n70L0r&(G>x(&Yp6DW4SPKx#r~jz(DRHS}S^Tj$^qtaBKctf#8nMjrK0N@%{xO%&*&9(tWnoMS|1gT6a#~q0lxbJy}#ZZ#+0}PCPUp?f-{mGv}0~U!T%jaW2RlS18O~2h;zHTj;cExbYyO($n@jc1xW{`W#gVYM+sU0`tH_B`;46UF{f53+uYKH+tjQVvd=!>z_FhNNKO5mAEfOFbY0fRs5wo!ZgLv{>EvQOx|4nKPPxd%lf!HJ908(@_{Kuw|_n}{$eu0f@QDGf8h>?$6`3>4~9KFd}x)YOYul|O5vt`L>VcgvaLx!FV8(tWkq0d#5xCysQF_P_#8e~kv<d&C1K^U$R&z@-k0JKVr5btw;(~ZH!Ii5T$`P}CLD4B1PnB3FNdF);vLz+vT8;>$xp)3*>mi>*`3FK&vgXH7e^bUKV~}u9sY;UFB;gu-;vghkT%^b#>;+iV7wn3=9!c?ia07j9GR$V%ev?=?z-9%!SiKQK^A*5crNVJM?V-%zh8`ayyR<d6+K&%0K+g(hckw2xo=X!oRYOGP)1HZMf#cr40>ffdk@o#is{bU7hE@Bbx$RWV$+85V{~?A1@}22gInVF=oMY<L+>UYdFxbF*qqUMcZ=lVP55OOM<W-&Q#{ZK(bW|k=YBT91lZAOBkLB(hGjAkx9BTQ?D96>G(xr-G6Rf>r7up}ny1ryaObSPUuit@&Voq1MrKbtDlD`j$ft>!7LM8<gB(5@g4YO)da4J_5BYTRQ;guJ#4S#kp&`n507nh81;2nzs0=LJPvG6d;wgwN3M_tuQt>$T_v5{u*79wzn<F&|JWJfOevatm=NBv=I{@alYdZPMvw!Y|Uac?#sf62%!djv3Q}NcA|AnDdRT~smfrF-1*}%sa=>0qJ-!>Nb5E6KR6p9=d!0cZHB1F>z2CF%F{|OxyhWXJdz<r~Kuy;3W<JfC^-Z;3i2MD)c@02nj)tR>3tosw}=7T+15=`p#SNz%DJEItJP2l#aojoPme@F!zs%ORwEm@$ao(T=`j0=N66VBO6k^&?>Z|Pu?t#!^IGDb2~!%l+XPrHWK;L%WRIU>-*!W&wA#@KvtAj2JE-x$kv5OBjJslp-SFnR^vjBr#D&P~yb{u14vf?H7FtA<Z+pMnTz3an56)7JQ`?KB=sy>{Q5c5oT`-e=OaZ+~t?j`nL!`}+;c`vKjJ)&t+oZXuOwWlIei13Jr|qQn#CCTj)Y=A-CJ+w?CX<9C~0kHy?$Xj)mVb!m}uZS~Z={Nk-p$<#0ia`_$JRn*m!yh_7NPr4c)9qrI2C6TJ2-}KLk+AU{<b2QN@$eE-1LSx`M+A-ATY<~O+$i`3q_^qFTpT17Mk&5}Jv)w@?WWF2j2btL63CS+p2*li%H57D>*DJtk2AVlz^g>D%>{KE@OFC$@E#6#X)g8a_eD2P65y;&{VV>Cxd&<G}_YLQ59IgI)zkc>T@B+w0riO72F6TmsaY-?Xh;i4Ty>Z5z-sc44Z-4hzN|B3Z0Pf>!eXF`IZ~S$7)IfE|-n;E;)0}QCq9d5wgfqQu<J9>A9>&c_-=wrYE^9!H4!$;zj1!Kn?NqZVK+Z(?3AUdtqw-35k<Oairvqsw>-FBh8=?oy!D(BR%o?L}9!xDa?7jQc9PiW}Jj^IG&3BFNpM#&iQE}%&-m@jEW5neW^Fo`uBIN6VR|o5gJ+%3CKK{;}@NjbsfyEUa_m-jUS~=3#h1G_;y2fUYQ!PG#>WQhBppzgOZPv-t^BvHW17Tip;!Rz1G$eb(khnMNj2Vv@V<2+}ogE!k)WKMzKLx+LDO>R5Q<%)u%P_>SHi9ep(M`;F3Y-C5oV^lUnSaASS(YwX>D>iHspdEhCVEP)2W*EuRz7*zE<ojHc(;19>Mrfkt$>Iw2#AXp?4g}3xhTipS6j_L1+RWbYLH7~k}Cz_F78_N6q5{{@f{)1J@{Z-2oJM6%aZG+s<<PVgxJ*Z#PQ)mJLWUT!7mxhWz5*?5C3`KpxPHd5uZWwy^opabUAVSh*iwlan>nF=4V)+b3$as$&tZn6QWd`P$o=i^1c>A#V25z_FUC!+L2~ZYlfOAauEdIK78qoDBRH+xg$*+ud05ssV(Tn#V{wB#!(xR^PM+})a0#ekQ+1AjUH06U_1A3%~alI52EX?e=-lF8&^eu<94zah$@W;Gs}yXzCoBu?9?b*ZIMsfyaQv~^MPT_{QTnN8D?Tr$lSNy|DB9z07CYOpvPc(4pujkIMUJ(#|GSJR=PSY#JRg-#c0|^oAFEe6uEVfDeX{c`C^+R-WeO#C^hZUkV^eY>oMZm7*vc)+bMKH!71}>uEgOX#q76||8$Ubc7o9`cjRM{d>IlDu@twlLvWxnP9O@k`J)c@SFSa~H`(@Aq4NWrdgq%(f|ZXlD|PYYuSfW4O@xY%{s}ZV9pAVIm+m(~_<x`&ag1M*?5D3d2|xW^Uhw@_S<pQUKUNj16gC68lDauqa#SN#rsXk~)rWvH4c<6lvKJ6^0SYI`%xg`Q2E21XDld72d`3g^eMbBV#lZWtw>bFWJ$g&xgxZy*_-N#n7W4Q<`>egALJI{RX&V1`4KVU-hy>gJAe%cZ({d-g|A3nRF5uLgbmc<6b|qVTmtf#iK*;gnPTVCDDs{moxb!ksZZ-vzHmqG*@c;F7o<U7*?;1};dXXYTq}f81sx$#Xq=ydDdvB3IC_$Ql98eIcL23{Lm0m-YCeq2FlMp}%O);S!14z4`d*_b#aQ<`mp4n@!S@V1Lr~P5gv)<>O#l<orn%;&?9BMywh`E6wn*eFQCQ2sR3E?&@{%HAXv$200-|BH6coEh76H%33E}k~X6%nN692pg>I{S=$9BT(gT=MIU+prp!ixd~B6?)Y$=;PT3dnZ_%SNIhI_`EZyu3ekV^F=<$=z-YS4L8+=uICWbf$_aS3IY8-y63tpaL&jMxrk)G5qn*(r+zpEt?0L01%AEFpzV+~zXi6w9o8Q)SvZFnxroJXg%oB5N<-ph71=50U4|l?Nppl#NDa;OR5ZGqEt5)GwPhn0o00JGVbUd5O`Dz_3X4~V$0xX2r7e+XK6?_PmDhSb#T5F&Lhqh(gQn#cRBr~(zuKzq&7X&o3&~IG<uVHehull)>BHirkuA&M<z--tAJR?V-ScYzvgFr5oQ&x~4n1cnRUFk#g0KqR2Ly;+p4B&17EBLKZ{j#$Fb0GdbSGpAq`!C~VeR-rSqkwXMUF5jL}BqJaiV~_S3X{(-Y+~YOwKVE9s+B=-z{NZIMNJiB62^NzJPoLB5n8aUFoTV!V(@>L^D5mz7jx4gP0=)Yd-Oy%<~4WEm5a-WxnrhX~M2QFXq{Fnuo@l5z?!pdDizU>po?jtm$NuI%OHhCDrcfi!Ay*VKvMhezo^WgT&lqGuV@Y$S?is&kAN>%vhIw7FqQy9fh?N&JrV-Ai*Uov&<rGH68npzCY%$?0VTDGqoZRgxozF?>~s6Wr>7(^&1cqj}JR&m+a5hP!YEA62}$su=z{rhfi0BDGiSTxx8qE@8$or`w4W!h<!~AY!nat)B2lqFHG6KSH+@m<8Ft_K~TnJ*SsGN7iP^B-<9}ls)GQ_`=lX3@}foop62EE!?j>xTYe8mW2eANm7t9ujTwxt5&LeDD;)L@RM&y6qkPR3Tpvn(QbSJkhsX$7`g$dQG1Yt?q)6f_?<}u;f{foIecvudfBEmQ83x-smSdI}%uV{qZQ@d3j4P)0{t?~4D37u|{+fHu)qYOD`YsuFq0I1np5&Ti4LsQ0S!g4^aQS8~w9%$@j0LBVRHfn@x>|n4|6_RYP*arhj9mnBsUJQ<HR&2^3p0%+MRv0U#3Y&_nh$p<db#53b3yTL&I(&jl~Di50!Xg{MWYxlDK|tIYQ0jgkvtFlj1Ytb;Cb%KuGDP$;HBKF)hk;L<CpIhLGwiR-qy)|ws1^*Jm$B{$6?I^(>H(dDJ$nsgp_(F^zH5P4M16imRzL6*0?dA?pfaV)<4!?JXlUGfL6fOwfP26w98wwXl#~$twmHCIQj85?ojH6ntulOdK7vUbFF9NNR?AYi)1sEj@4S1J*yW*l=KW6#^VBds9N*1$6}|#BT``aROfjQI~{#V<d~=DrY>u<lLa|#8D|55n~+Z38j+7&$`|JX^_<rLlRI*j9=HY9`D7oQsuS@~QH-s&<WnA(uxo{%o9}p19KOjOhHlS&6K8r<IA-y#akcqQR<Y<Z0|C3(Tz4_zi#&9XR(}ZY-sW2RXPB3B<m0YQFUQ(aasE3pG+Vb*7xn|k+QdAJwvO8w5J}&gguJr0ZXYVmi-*G)*y@M(1;_#K#5^pw=u{ia$u4uVjped4HEB+(&^Du~+65aI3#17SRaw38P2kq`tp<;nAC*om5=z%Cj509l3zObHav<JKLNufwJAP~NMR`DkeYk;bpR3i-6Zc_Qdq==)KZVKKg^W+>U`XcSdc%Falss1N7{&tyb4FTl^&wP2l=3v{BM0@n7KcP#_1?z)dPM|OS5U>}Stk{9ep&XkHpRFfXo7W#Pt6zd%)HF5@9r);fr{m{YuYB^*^GuKzjE@ON-L{dijZ4<Eu{!1g9a@G`c}k^w2WbrZ3|O3GX}-Lm5gxfV3l;~mgJZiCm27eH`QeA<u<f=k0qW(5BEOO7_C*>TxR5)a@IGkpF^xESA4wBYw}rC#*W8{T~)I;!8hThyNMcZiEp}Q&gx);fOlHvgzhZ#<)OTZHGUZWs5?07!6dc72<AKT(wJ#>pWF>QzJmR?!IWH}umRdS8x0pqE`e~`75k{e3<f1-4EWP)+xBi7gGR~$Whsx3){x{$X+MRlQ_XGzbjGJWhT@nXJ<L9w-J%ccUE>=W29R$s6J=~Sq{BPU8sNUuxA~-C7FW!$+tjT9n9)`?S9hK3dw9SUi1K~vribCWQyig~Os8;lGV49<_%%hN`|UO$gEz$~JWUs3**Jm<vPJSM43|t-<=vE19qUwM*9|bTTAh3Ih4k$C;@CQem-2aWehP#3{>_q^U)uFriz;Eedn#zopZ2+^0}HgSg5iNC)zSAOtvvmM81w#c6LU?;Vwof}*iNF0blH6dv+A8}J<sZkii>R(Rb^S%=RzV%-lLv4+v<u7U&1Vmk%&AwJ6WhrkG=)d*U*Sd;V`@Uyh!(}wQ11j2W@E{)Qg7^b#$2S>Gfed1>n1JCQV@=rR|%M8gz@%qjYpN98bVNsTB<xCyq-U_dfU=6u6hN-)_37W^NczN~tEwEqDsz=Q#_)dErMZkgOE)J#rlv9)IYv@DfL6ClpVs{=E~(Dh>)Mv$Xr3kxBFk9IOy<7nT?#5_-8N4+O~Q@gvv%&gM~n84dJ<1ow(n9h%*zo8)#y5h9@0>J6inB0cB!-PZ{k<)eh3gWUq5=tD{iguGsgs1Z_ArqVamPLb<NK<hYf54&Bi9j-{&%{z&!6ESa^+n_jvkY&m2zL5DMKfwWAA?TJl_+{WHDtu1kM@4PhMqw`;HWxZkK}iEh-`S~bH;CA5fac^H4F+bo{BY%RK#<wGdMfwOt)$d1E72J}qX!n<720GQkBe*O{bW5U7lRz7eTqk)cp3UY9!ohARJXp3(>kCp1itpyPQ$NwU+6|*n{C`Zsl?m}T-OfL)>{g$Xbe-``?UopoCQu_hP8hlBX?4&Ovg^+3bz1@eAre~?QVh>oRYbnJMn26-@`=nl{W8*O&I?o^@R~o@~i5{WjTDqo<Lan5zadfdjES$aTBv$Eu2MCjMiC_hras9&v`y<UHit!Rpn=EsEN?@Pq+D!LpIy}mR9F)-VCx?X>4np^h6jMbG>qVvVey$8EjBi3O-r)*I}>`1Xpb@@L`wfOK)Dck>z~6y<nMi8LZL0d~@fCXKBsVrq`&wiDEI;O#aapQ(l~LhzrQSL4ueXlgL=v>UG9e%3j1}6a12o;XB6{aOF`ORx-5f@N<qRHm^fAl=EKT*ftf`cK!ZHtwMJ*&Cgi*(vv@0$^zA=kJv&pAK#N|mG87LsRFk)SN3feiNPg3n)2dq?S)K5QnZq2yr<*ZfkDbz6z!&>hpXr`!&?A8$hRgjd}}<DOC`iUMlyabdafbF+0Q?OH-;yIvnZpd8z?mkXfB(rs!Qi|tOxpcOt|w4)UzLM&H6ar!zK39^_@KbXbrFUso&Ss<+tu-77WwmAhrPg*VjeCi!sZe-({X53FPsNJTtjXYD+Ns{YSVwd+=K(nPJCPB>uFA*G@mUFthV=xgp+?ROewE9FBYsjdX?aRHd0HOp-%TJ<-S3)#^~^h8(AQH7E+SVJ+aoM1h_R0)M#oUHejJ9dhBc$3Z!(<i_1HttD{wQ5#^sOQ~I|luL&&tfI@?@3e>Rs_>Kksz(~~2FQ2l6lBT{O|jIDuTkSD!|;n>DJ^}f%@`Arli^u=yLFe7{-leDZy=S+sHT|6nPy+QOd4EQN6lLq_PejcsLzsgN4c}2JT%#kZ^q2-H(hz-D}-i0&Z)ult=<^^Dsm8f{ja`_KPYS1Pmi=y--v2#^@j2dfJMWS6v;8f#Jd<dc&_(YV;b=uC5>;zTk0?}SgQPK66@3~uw7a5{l}XWWqAQEw%6MefP*@s&ih2I*qJv<!?Tn?&)9=XMtP<kLp;l+zI!h>RxCBP>2_@sa;F~Xye=Ex1siiqZLjt5il;s9BL@GduGn5cVUKP!1-}s#Zob@e>{T2wO=_^@VYxI2YFCRZ76Ao;M6mnzT*o5BoOV{?5v>8**^U0pBHLeFuMzo&F(hXU(rRFxV-W5v$ZZM4KxDpFO$&)m5=Sv~Cwb)b&;EX>svDDdk)%!r^S7Ut!Kp(fhsVU(OovWo#O>BF(;0ZQqq@ZLz@IiRqAE-#nlgqmQ|0ZFBC>0Et_1HOKET?p8ghysePosEh!-m2>YZ^T>nt8eo9sWKjL<9CXH$S+7SP>UkJXuVeQbOhc>HV|y7(=bl9X>kA4x%q8APX~>eKO_AF|(o&q}~&r*<r}JA*;<AA+u!Uv7l}&Eqfk#Tr#%Pok9CM|_q<C0YxyCo6YuH!FxYhvLho=ZEvXkxMqX<3h!-?({z?z+d6GATg<asN#@smrtejfn04W_m+3PL(s>>7H@79vrgjiC!+nF4#<^a6!Yp*@)wz~Hbry+S%6$0895A=d(!*qgkusD*B4{uHCpq)>mB^mbH<=AyGZmy9!3q5-4U!fJ6^o{Ak!<{Ryg=CDM@lmVgvIk;|*(X2b)@$gn0?luB^USm2#0lm6FGYv+dG#jqS%2I8Pb7a^8F_ngx}fW-`yAAdQJgi@<<m?&|TsetTOvi}@PYbbFV_RrCpe{YXXC7;`NE(0kV3#tN(kWS)|Ci)Edq`2xv0a|aiOQK4#P=}jnNh**XmdJ>FX&Gwl7%WB2~SeKwS^2&n=uI<*8OR=IV;1e?xfhsL#xJYL2A`Koia9xBi4u~5S1olGKT^xoyy|sgRp5NgsY{(5NxlJQ?T5#YcPtujNOo%V*q<hB*PWYUy&18PDQMr1?e-?$c7S1NHf-I}ocA1?SQ#eL3glC4w%rDKWe!Uhmj`)Af;^XrKlB|SdD)i6xs0#8RELzev8zMS*6xKUIAl$RWh$;I^BmQ&j)Q`J?a(hvWb`T$TsDoRvo1}SycM56PRJiZ3OI<>#E{{@uO(-ai8s06VvCHFy?Ft$GF-k6?U7MaH`p_;nz+_6m7=~n?tcp$?j?U48$0OJ(s*W^o#wY&#3%lG=%!Sstx;l=2#K>-+N9+iOI^AT3P<TXFbdvI5Q+aoQ>BUBS9J@yA;D=!8dmz_umL#dy#VS|6C*2ArzYi?At1MMnFw0%l(ke9n6R68OM1{B9)1-2Q>>lxhmzx(;?~j(14Dmipj_F58VeiZ`i1S0-Q$nnunxnW&t5@@PzQffIf~fLh>D@8wEwzZ)G4zEU%jiR(G3g`=gqh`!c^$+|+CM#F358WZ7yFrJNiC_=Ov*<;6T}Y$@Nczrn?tTdQcj)rU)!%cVf8AXSyy-;G{0XS;S}LZP|vmg0wPXP(QXyYZsn*bZ3D-q65!rYQ&f3_jhP{7Q01MOUN*&HSTnzVZFO)IkKblJwH-Cp8bJic2382z%^CJ>ZI6dXIwc>W!tS0t<>iiBPT)0(VAB>?Y*B?)Jr7Wp3c!_@<}4+VBxX6?OAXU+98(sZq$!`SC<6^4zzdftep@;ZN9N$KfdIfYivMxk{`Yzc{QLP(&(Yge*ApH97YjfHoM%BK<cv2x0Km)(08syf?fFGJ`}+Uw(w>JF37AD18UO&v1^{sU12Sd&N3U^zS7%qxp#QE%Ij?cPL-}uwWcL5uul&5i`Dpat3YOC6g3|wmo{J6ojUsgZLx|9M{Cw)j|H1YDLk7ut{Cw5UZ#>xkA2mGZ!Skm<euEZ%|IgWw^D^f>`QI{}0snI7&%@{K-QO^M(0Q}>JbT`I{LPjI|I3Lqgiuoby~yzEUi<a2(3HPt{{@x*B4Y'


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
            deadline=time.monotonic()+25; rows=[]; controls=[]; gcc_events=[]
            while time.monotonic()<deadline:
                if sp.poll() is not None: raise RuntimeError(f"{tag} sender exited during transport preflight")
                if rp.poll() is not None: raise RuntimeError(f"{tag} receiver exited during transport preflight")
                rows=csv_rows(rout/"receiver_frames.csv"); controls=csv_rows(out/"rtmpc_control.csv")
                gcc_events=csv_rows(sout/"gcc_events.csv")
                if len(rows)>=300 and len(controls)>=20 and len(gcc_events)>=3: break
                time.sleep(.5)
            if len(rows)<300: raise RuntimeError(f"{tag} receiver produced only {len(rows)} complete AUs in preflight")
            if len(gcc_events)<3: raise RuntimeError(f"{tag} GCC callback produced only {len(gcc_events)} estimator events; adaptive controller is not live")
            pts=[int(x["pts_ns"]) for x in rows[-300:] if int(x.get("pts_ns","-1"))>=0]
            if len(pts)<290: raise RuntimeError(f"{tag} preflight has too few valid PTS values: {len(pts)}")
            bad=sum(1 for a,b in zip(pts,pts[1:]) if b<=a or b-a>1.5*(1_000_000_000/FPS))
            if bad: raise RuntimeError(f"{tag} preflight access-unit PTS continuity failures: {bad}")
            caps=(rout/"receiver_caps.txt").read_text(encoding="utf-8",errors="replace") if (rout/"receiver_caps.txt").is_file() else ""
            row={"codec":codec,"av1_input_mode":av1_mode,"pass":True,"received_access_units":len(rows),"control_samples":len(controls),"gcc_estimator_events":len(gcc_events),"receiver_caps":caps}
            print(f"  PASS {tag}: {len(rows)} complete received AUs; {len(controls)} controller samples; {len(gcc_events)} GCC estimator events; caps={caps[:140]}",flush=True)
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
    out=root/"UPLOAD_SATC_INTEGRATED_REAL5G_RESULTS_V14.zip"
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


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--continue-on-benchmark-fail",action="store_true",help="diagnostic only: continue network runs if an uncapped case is <60")
    ap.add_argument("--root-run",action="store_true",help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.self_test: return self_test()
    if not args.root_run and os.geteuid()!=0:
        ensure_root([x for x in sys.argv[1:] if x!="--root-run"]); return 0
    if os.geteuid()!=0: raise RuntimeError("Root re-exec failed; Mininet/tc cannot run.")

    user,home=real_user_home()
    timestamp=_dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root=home/"Downloads"/f"SATC_INTEGRATED_REAL5G_60FPS_{timestamp}"
    root.mkdir(parents=True,exist_ok=False); code=root/"code"; code.mkdir(); extract_payload(code)
    print(f"RESULTS: {root}",flush=True)
    print("One-pass experiment: verified-live GCC + fast-down/slow-up 100-ms RT-MPC, three-codec WebRTC canary, uncapped capacity proof, then 18 real-5G Mininet/WebRTC sessions.",flush=True)
    benchmarks=[]; sessions=[]; verdict=None; upload=None
    try:
        traces=json.loads((code/"trace_data.json").read_text())
        meta=preflight(root,code,user,home); write_protocol(root,meta,traces)
        encoder=build_encoder(root,code)
        producer_wire_preflight(root, code, encoder)
        print("\n[3/5] AV1 IVF-unwrapping + stable full-caps preflight (OBU first, Annex B fallback); H.264/HEVC follow only after AV1 passes",flush=True)
        _transport_rows,selected_av1_mode=transport_preflight(root,code,encoder)
        benchmarks=benchmark_matrix(root,code,encoder,args.continue_on_benchmark_fail,selected_av1_mode)
        benchmark_ok=(len(benchmarks)==9 and all(r.get("status")=="VALID" and r.get("at_least_60") for r in benchmarks))
        if not benchmark_ok and not args.continue_on_benchmark_fail:
            raise RuntimeError("Strict uncapped >=60 FPS precondition failed. Network campaign intentionally not started; benchmark rows are preserved in benchmark_summary.csv and benchmark_verdict.json.")
        print("\n[5/5] Integrated network campaign: 3 codecs x 6 real-5G traces",flush=True)
        # Alternate condition order across codecs/replicates to avoid one fixed
        # low-before-high ordering pattern. The frozen order is saved first.
        plan=[]
        for ci,codec in enumerate(CODECS):
            for rep in (1,2,3):
                conds=CONDITIONS if (ci+rep)%2 else tuple(reversed(CONDITIONS))
                for condition in conds: plan.append({"codec":codec,"condition":condition,"replicate":rep,"video":REP_VIDEO[rep]})
        dump(root/"frozen_network_run_order.json",plan)
        for job in plan:
            result=run_network_session(root,code,encoder,traces,job["codec"],job["condition"],job["replicate"],selected_av1_mode)
            sessions.append(result); summarize(root,benchmarks,sessions,len(plan))
        command(["nvidia-smi","-q"],root/"gpu_after.txt")
    except KeyboardInterrupt:
        (root/"CAMPAIGN_INTERRUPTED.txt").write_text("Interrupted by user. Completed data are preserved.\n",encoding="utf-8")
    except BaseException:
        (root/"SETUP_OR_CAMPAIGN_ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print(traceback.format_exc(),file=sys.stderr,flush=True)
    finally:
        verdict=summarize(root,benchmarks,sessions,18)
        upload=package(root)
        chown_tree(root,user)
        print("\nUPLOAD THIS FILE:",upload,flush=True)
        print("Overall strict >=60 FPS verdict:",verdict["all_codecs_all_real5g_traces_sustain_60fps"],flush=True)
        print("Completed network sessions:",len(sessions),"/ 18",flush=True)
    return 0 if verdict and verdict["all_codecs_all_real5g_traces_sustain_60fps"] else 2

if __name__=="__main__":
    raise SystemExit(main())
