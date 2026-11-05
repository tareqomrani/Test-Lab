"""UAV Compass Variance Simulator: single-file Streamlit deployment.

Upload this app.py to repository root and set Streamlit main file to app.py.
All ten original simulation/helper modules are embedded. requirements.txt is still
needed for Streamlit, NumPy, Pandas, Matplotlib, Plotly, PyGeomag and SciPy.
"""
from __future__ import annotations

from dataclasses import replace
import io
import json
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================================
# SELF-CONTAINED SIMULATION ENGINES (UAV Compass v6)
# ============================================================================
# The original v6 Python modules are embedded here so this ONE app.py can run
# directly from the root of a GitHub/Streamlit Cloud repository. No engine.py,
# advanced_engine.py, or other locally imported files are needed.
#
# Module sources are losslessly compressed to keep the deployable file compact.
# Every embedded source is SHA-256 checked and loaded in dependency order into
# Python's in-memory module table. Standard requirements.txt packages remain
# external pip dependencies.
import base64 as _bundle_b64
import hashlib as _bundle_hashlib
import sys as _bundle_sys
import types as _bundle_types
import zlib as _bundle_zlib

_BUNDLED_V6_MODULES = [
    ('engine', '034858ed5443fd6887b0230465a4c71c444443b5e6992fd7ba620a5631350fd8', (
        'c-p-^YjfK;lHdI+aGKgHF`;Nle#G{~sm&&ynbakot0Xg9S1uKc79oifiBw3+kL=F=ru_~3%dQ^)2#}!d*{izLOd^p$qtR&e8_-'
        '}d_$Vuxl=Ic7$k+E`yH4_?6u-RwRm_XcHmR!pJh{o1Nu3qBNV8>DCu>pPW_h%?cQLvY>tc=%l6)!i8u}Dvn&n9?t5B?D0-'
        'u*+4sE{NW|b6+ve<~vAAh+Jc~RD@P?XRj$t#iNwJh^wy}pn3KCh%$%3_l&b6IC|aU<t-QHrb*2czUJtG*XWRxZk9BSlizS-nf;{+BG58^'
        'Cd1i4s0!xtd6{cUg#ab^}ANv*ijA%<HmPi+B4!hd2tNk5dc9Uq5^Z1&lM7Wt}ZDehj!L*(TZ>3<i6Id>k)!^{$k0EV9iOCLxkMFKTLB?Q'
        'xrVv0ekt_(V5K<=36$w`o!*0JW-Q^Hx6sf=EDoF89<MnA3{8)%RPZrF#B0o7bWEyxXqjo^G4(HrsoVR3hK%N82P%;Q{=&P51VGc>T-'
        '!&!5D(814!9Gq@}^uvF_oC<YfF-ihm7S=IRE68PczUTkE|FaNthvLx&5x}<pyLhk?L<)=@g*wx8WVl(<$ZHr<7h~k4c#pW8;B?Y#rfM(Z'
        '8K8J>T!QO|T-(C<NgAaO<HLt9SbxOd1g;KG+FN=M{2#Z}sqw`RsBuk-5T_*D{+%4F9^V7R&JdICJ0r+V&os16-'
        '#|JMar!S5tC+6i5ygZ6dPbY`RhewAmCgT^U>g9xc9!JLqhbO1U$K%O(I+;%P_V!Y_5VvKrjZ?WCCTGA+MN$W1^ol>75kX1`v&_Yi9u1Q~'
        'JQtIbaWoFZpTxm2d=Mjkpa7~l!>h+Y8i)#zjS+02KY|f*RJ6?y)Nm@EHQz`2qXJ&9WKkmBlXaYy*`kg&@m7}cm(B1xODd#9&#16je#q`%'
        'ZFc#*C`;H1@L4Z#u~<~H{vJ?U$Wj7F5O*MJ;0{mLMZO%-'
        'nnt+i5L^OFl3Z6qLw`Md5ZIF>?6Z6!vITT`1q^@`munC^lkpfotI5J(L;9eE56~rmIse)0!Xa$9Z!$msT<&DBM~}p(Z1bkb7uk}hsAgjk'
        '0g!xg1S^AYQ#oIw5Cm>@Ta0`yHbpAeJ?*Nklxe)#R?WM~iFyYzIE`iNWpvWK%iVXQadg0g?|2`_@Dc<s!Bl9~vDIN59WfGf38FMB&O}w0'
        '@Z6FX^nQ`-'
        '*3hv|N~imgo&>*_fRT;qM8C%Q8_@opP3^P<lIfYaE<lE#iysnLUIs!!q5zo%ESeXfY~o#QPvwB;f)XWW2HHAP9Uof_7(lDV<d7TWRTWp+'
        'a+AcH=JkP|OtOqYSxR?3Cx_~FSzpKXs+1K@%kDp!nkAPwL0*@`DLu;a9)2<;UKM5b4OAQK)kP)&v^{wP8Z>!KoT1QnIcF-wcc3b&-'
        '3Ew{&2ayMG6Zc}--n>FLDzv&NGec&sklkjJ0|_S09KIkd?n{!Jaa#Wod<wH@unHh0IUUoEw{iAXfYV+PW|`P?pT1L-'
        '4<0HXL(k~@vxHX1&Nu<BA2`ci7*R7Yd97!1vZJ0fyGOhGN|QL9N|ZPZde{4O{U{<&_JPuNzgv|6(@C3mPI)ne9pkYsA{<tn_X3lYY8-'
        '|S72ET$Dx?cXaEfs^htz3MKuW8$W06qW;5fHlksB+yv|DMfhrcnCh+_&Pdx+R?9~X)PSiB9wx5R{kJ8~?4ojV-yo=dTg9k1eJUt~Q2q~b'
        'GGB?0>px3y~^(!&}9vk?V{7YWk=4xOA9pE7-'
        '4+35p08|(o9PGV|w6NYd+a5bTl`fUcD}X|MJ&Q9nG`B?qK0F;04>b+E8wuYYjGw$pTFqkjE;h}E@CXcicSIDyrlXUFty=QuY3yt0bDx7L'
        'fjj&H*m5OkIM3>Jg_FBNEzwCYpNd0Unjz4SpZ_8Zyg{I>9^NboSnoG-'
        'I6rIjekf|&+(8B8*jV$1|Ig1oHZ!>v?G1V^kU!`47IpAI7YqbmURMv{LT<nnViXGh-'
        '!n#)rv)<Eoh5*CEUBTPs_KAjYRwO2pzASsE&U*T1P;>4^);T*p~o3{1lyLY^6%r@a&oAL<{hod;^MkOA&KY>YcL|?Rcf1xp>P({;bG@bZ'
        '90@Wbk4V6vgT>AiGGy1gy|P0tv@=0S71MdO6UIFUURUb*=^>Hhvc{}()-b(C~uQ8-7nEC-cL!;0%hM{{Ce>|6!*z3#{+KUDx0rm#ESVRw'
        '9#QVTlR)9Jo>$D)$Us|LnNgXJ0QsjsDxzaDWjCp?)RuM2*3|#<D!bhCnGzo6&I)$7!7#`D@DN2-'
        'LKVxg6E6Pz{W4LIe6K)jDYtu81NaGFgB>FSl`Gp(v)IQRR&wc@eu4q4735?ZZwZtc7LFaw{FXpMCFz%XgBvsX11c~3+~>P_GGFoAHtY{^'
        '**=X&x@-4o)+3_g3b^i!%bkJ6MPGdmXu|3KfE%$sYka)20EcbsmHa(#5ZAISKR;{&Q@0qwA~eP=4Ug;Q*N@5MMfxKD+1%k@Sl-uw$wDTZ'
        '3WEEQ~M@MK;eKNSH+^{$MRn4hoF)OdZS*N-'
        'ik0_Qio#n0{;fve%7e72B2zrHHjuCp#TR1|C$`lLU9$1kKoJ6AwHS%gXs~y4#5x~znJNXt^pOC;RBeQAP$Vq<Csio$P;T^9QFl0IOGQt{'
        'Cq%ooFX2q8IoH7MG*h2*yK5(qUi-'
        'o(%^`#pL^K||4gQhFOEwdO;6x!jl2Y6F3O8?ldRFm2A?W$rQ>PS6WIuyx~KL$YK=L0;qN(ddMawF`M53TetR<%RrXDuD^6<SA4Cq^^Xp|'
        'qQfG+N3<>s#NEsj<=TYP4a_|Bg7!Q|%oyg}D&uIW6RzowEii;R^1cD2<d=qUN5gkD8lu}k43rRo_OdAka6JUns8?aTNrP)Hl(=<kr=t6b'
        'IdoEmYXo4JyvG2A7Ij#%TAt;dcn{_n;iT8i;1sX&zFk02>_ww8#%S&=%Pdcq^)8=nxX$LmCgENO(q0=hX(tmK~PTy*#CA9)X2z6dtlAT)'
        'eD$=Z1_uHZ#+61@BKLPT0pqWV$EO*MC#>JNb2Tg|hd(fJsM2i_02pXD5o4eX0;pjmNfNe&paX=uDtA=tEKcEz|yPy#gU*Fl3aQLm$^xof'
        'e;%WNL-*l>*{;mwRIVgfErZ`i)8QvLd>)w3o#`)U8&TU({TU{Kh`p!{+sQ&I9#iy-E3W5H=BgH7dw!L@JR7in67-'
        'K9XCW~uWl5=XpM<JL)Rfc@hePL`^pP5@HJ2&9?!g6R71p22jW?zHAyQio$NC$8~Mn##*5-'
        's)}8pNTRStW9LS0mm6aUvTKzsy&dsZ!?byBh66@W!s!(1N%D$iFUb(U1gt77U3EbcAmP#uglbQhf!h76jaOD_IMI8-'
        'q3}V73O6Qe0xZEXQ2cyi6ALC`0QugYCzNr1yEU!F&}-M%M!UiqTqH1lZP_$#9g`ARdWW0mjlMz&!8<pYOv{UQwJc(H}aumjl+-'
        'tQquezYbu{hZD+AeAVXdx=xhh=m2{L#CZZl!WaJiYBQX{GHk|BJQn{}RC2pQmIuw2r(`>Tf#-e&hdoPoV4K-'
        'b9o|~SK)O~Z`>Vxbb{qS%&2trMCj<$W&ldit&(UYJ9GLD|Wi+xz4w`keeS^AErC)qDLD-BHM4zeMU7Sm_zV2YiFYrQxG072Sysl+FybDm'
        'Ue9>sKFi`yyHGzbXCrI%B7d46&2-p}bKs2m|-19j@h&~4)PFAnh7S<aa&qtPv;%c@)c);~-'
        'WCiTW0{0c`l1CS;kq&PPY?kUY_QikwAg(S#aT%gl2Mn7AjFP|<0Dj-'
        'E;0c9?F4Vu5Gn09`Lucp?B;odc$ZJM4YROhTMWp=Q)xnI7uMADTJJ2~mKcXge#jV2|tm3Q#g}-'
        '>RZ0^miFtKS(KAW$#dViOL%0v@2t!#k~NT5i+yE>e~<J(-'
        'fowVAbs5fno+_o@*^Cr=#CS>NK%(R9HkJXba4wCyuQ4J=l!KjOK1LA;M5dBgmczU5xTVK-'
        '_*r^7qh(`k*(lql1+L86(JMlKF=CDJg9Yj2%KnkbmL^V%pa03)~sVvC`;~C@_@N)$Wc~5X8N8BdsFU)u&QUvyMQaFjIz~NZ4ni!FlhMS~'
        'NL5rpB(+<JKJW|qSiL*j8a*4AU`<DW~N8kW@mVn+UKvUhnZXi!I<jG9$2dgXi>z#oVEtgjZYT>!l*RmdJK3ZNKs<rXGJ5sCEefL$lSFj('
        'Kmb2;XQrpaKMf=I&1-gl6&&0HmNzvq(pSXa)Lq1SV(W!5Lvf~t}$Cn^u0lz<sC5c^wC58fsOF1f_K90?Q3d9z7{z6Ia1%Yt%>0NGwHM-'
        '>69K^Ji&?=vxZ9gm{i7^8*Tv3n=7?{dZLX}B2!k3fB57}+jpG}s>I-6vr!m;YXZ^?<jTS+_?*?{E;jJWJZrcQ29Co{6#7-'
        '`#WS=?o?AaZ?=Ipc(?dbqepS#`f)WlbW`jK**v)zi>UyiO_!+7>2U?XJ<lt`RueY|m8M^BJoEt~lZiMgd>7hplZ&8!@CE%I0RWSU@j=R>'
        'COBb+*nh(SU^%7&Mht?HYL?%28vhwFtyZF-'
        '1Q~eZLw_W+>A~=E0c0zjO)Sgj|>%kg}|(fQKafpO4XeF(CLIY+DKc=M(E4jMrFx@VrJNv)0uLV67n|O0qoBPGpuruEq8QYlMTTH@SXe;%'
        'UwHV`w<x15j!GQ&8WX2!lUtp&){jtOn%`P-'
        'UCl8aTztomgzSO{2E};0G)v3)@ljSz}au+~oilq&tuvsFlq_ENQD`Is!So)wYH)=%0(@5wI&L3p78_F)uUZKVw#uxsuu{nh&^A+!CIHY{'
        'LA%vgn}|C>oyWTxM$wF*k|qeGIYG;1UE%ue{5%Is@Yq-3zRx;X1YqtLg-&QdI`~ABUL5!*D<GrmBfUyQyJQo`E-'
        'rH;zL3zZKKAeN1i;ouFizM@^-'
        'QC0*@zdGk)oJ&;~EUa5WxfKS__A4Y*!R!_%msXq558o6u#D;qP8fhfu31i)MpsK_A^ajXh3)=5>z+FrDt6HIJKA9CvrhGvraTyAU9e9O3'
        '$zoDEt=Q<SgioTh5)@a-U3`%NITcBatD>>g`rYm+JcVGLUV?Nd-'
        '6zNOJ65KA!S!8#zQhJ75N%(izkS2CDQL6=wy2+{vNeX|>wTe<I1ezORp)gG5Uw7b$#vp;087d4LnY1jn#jZBe)1pNwf!$<zA3uPw8lsYz'
        'Vc6vsC3I)XOv9UO?GU;wZUN*Prh0`Q6WTCWGuBnO2Q$=fC59!2sKO^!uAdei%&LsG3U2mLN44w{4l9|i$}<{wA;Yw#s)IvN8cYqM8WAH!'
        'GpIpA#nr&d9&~dooIVXX`}uIPMG5_kQ*RolSM7v7$(YdWb>mXn&qH_9z%C69#L(tDm8qaj!+WJK8OxiLAaQCnuuR4!c4)hJDpGsphJ$B;'
        '+cC@Y;zo0pahWv85l<JRkfzsn;-@B&%K6zQo}E&8E=Xi!2U*TXBwn*iV8KesN!cd({GMF|;LLP~8PG*St|cnKlxi-qKw6e)v4C#id=8dB'
        'M$UYyv`vRiayOh}s>#vBJ(&=(|A^`6F(v`NHv<ZOh8!ISfkp8Wg_fJ&Qb?o7o>^30H7%8&1jf&C&}v{j)`{A-'
        '#G$GyGr*7`{aJA=2U*yzCv1W%%W~ILwBti9E<r>=jng0&`+f&93%1sX3V}qjgaxl^HbkFo3henz6xt*X+g7l^ynSQ=`?18ru@QR%Vovqy'
        'g7a9g_}5Z^jH7fDS_f!6d~fVyG-oYH#BjPekm}#Eip+AvCIi03vUw1UC2=iGY=uSW+U%|xq;6Lk^decyCNw33;hVB@Ux~@o6&3V0+7@89'
        'E$U&=f@JFw1JHih3)gPs0k4R5{(v*pMN`XY-tULAz<qFp4{R&`9Mi}KjFrG;pXXdqXwSwpfSDZNMECrA2+O6jT5d2q!a4@Nu$r-'
        'fyu^grQG2$S3ZgPE7}#Wj`XnIGnR{K?eBb?psZFLOYzJgOQsB5$YL={^P^{!h-'
        'a!c&4TTNj#D(6~0!u@|ws&w2dWd5FAZ6NJUSZbNS9|AfWG*EG1Sr1+q%f(q*#pDkfGxkpmH|P4dCR$59M}s?%M<9*99TdiB1K!Iax}5Qz'
        'h41!1%(d6J8zyS#~VrP&Jp>YBxR_?WEY24n6axVTek3X5Yx`PM~qo6rl{mu2cT;BRDhj!-'
        '+?0LclIHN#^%ucE^n0Ttyc+$2|y&4=&!ZFnV?6dGr*B}gUArT8;pBo`2zh~)AS}OGt9k4d)_VA>>gFy>G1AZXLV`OG3qy^IB_fbb#lZm6'
        '-5m1hK((zywOIFDT5h2S9S|<Xf`6-b*WCvv-'
        'T{yegG23tps<*blZYg<%)dA){oHoDJ_vtH8=htwcK)t^|%$Ne!3v`I*585mJYmER7Ah`=roNpWWmokCXjt=Ic7Ryz$^%UPLd*Fz7#o~*a'
        'LeI41*<RM1kLsT-tc7vFIJ!RJ#gIQJLDN231FbB>*7b|NQ9_2oQ`y%y;Xg1UO`1P*kQ)a#`%Eruv_^5X?4m&~y3RuHc=!TY~KXYu{n~!='
        '5V0VT{3uw_Kl|Y`5!s4u5Q9DajC;Cp*wV&om!C<LPXRD&y0A7O+y9O^*dXW?#dG0~=o6z)(9fuab0=;kgNNn)p!5Zxmg$Of=snFjkD$iG'
        'PI^PA(Tdv}sAp`!_8fPjh)jkJVupddt?V*3*0KUFzS%-'
        'yAcYgB_uYb<DaoSaYXNA7&J^B?3{iyZOJlRcDhIToyMu97k>Iq(Qj71G`{b0s~k_=IL;#Bih4nRIg&bOlG#5(lS&XP>reTJT}JaNqdodF'
        'zVW^?9~A4I&RO@b$d)Bc3UXVjo)-'
        'uXVMZnWdjgXeba!V63&?l<T01dXazu@Z^r*Uo6cPQub?smp8<|%lw+jPx<b!(rx>^=*N$C79y{=_yKP0`fzd9VpJ*HLC?S(BWUK?fTI#c'
        '1d|MCC9(Ee_1oY}o!^dKGUC-*@W9p|w-Tr{;^;>DyKz<YKp&5G{qs@jTYaqA5S!MKD$cyU|45<}`&u@}!O|Hq$m+${aB#Ro$j6ASXxq-'
        'n!uwz@L{Fcc!6glXDQguWnV^qez7Ly)mwhD{HM@f_ws4_Fy<YK$Judw)ukLvF#A~#Xf151@Oy*fV-'
        'm8c7tQUM+eSP7#l*&wvvcAG7X@|m&#92iLDLMF8bSaY;L70xLLO`2R8ce>vG^5@SgB;kRE8faN!hHP6FwNm}!eR3_=Du-'
        '6d8j~mMY@XG2K;v-<P6&|8?66yahSY&ypi(IrhZ-@O-4@4T`;pF;wg1R<vaM(XBPdSb5p11%^UOl`d3D~^>W-'
        'aB83KYkY2fllH{O6HunhD8V7Uf1ZGcVNz>b~-'
        '41@<*L^;T&tc<xH;@rvjp}L0=1F`=oEM48?1m0Se&W<t(`oj+TWO=;K&<Z;z+h;Nzqg4<cbN<PxwrN@3{g7eVhS9;e1pyR_hZIzoeC#%T'
        ')T2fgc1QtG-01{o4+1gtLFM~k%kKo<xS`EliXs;|$i^M+PUZC^;xEZPpktYQeo)F~rQQ6I_OJm=!e|8BNO`MO^dKzD9++9llt-Gg_3+f@'
        '+V0_R+U|MGrg_L;W*)Piv@()-'
        '_<>FF;6a#AR`p)>+Db%WwdUa?1aZ=Ne(mj1$gX?oHCjBqniQH>J^x=Uj6Y~?_>G~bwET{>ChIBsFvm({JP!;Kfa+85K-'
        '@aJmi+AzXs9hqiMO_(^H)*nur>$ht87$#-'
        'O(|+Z3@t$Cdi_bXS!k&zdh%tSPEnmQ?{&}dQ*4+4q&w!I(D~}UIB*PRdiWOQ*hN>p~8M7Ciz)HBYf0~ww0J>i^ap9pU)=%c@gJQHda1?='
        '^FVC-XRU<Btk)UN<yE)wJi-otx=}xX*NCYxb$9<CE%VAur=ej9oEN#2*OhWpx7!#%jmq<rEgNc<*vfVi=vESPhz}T_r{aptQhp#U<nO9a'
        'omT!NAwED54@tHK^O}mFkqdew6TXh(NHv5FsP_8d_`aTb-Tv<72=KOczdjN1-'
        'BOvomKZsJ64>&H?PFl3QnR1pYx>z=e_rinAo}{9eu%h(?pd(U^mPDdDa*!l!6+2fncVfVLnbvv=6=0^0^$+z8b)m<BmXv8N1WyL-'
        'D1UocKn1V4$urb}K@>i11q%Dr}#1jR#=|cs~F2$B*xRy?8rtgR{-Z5B7cY8Tz)nHXL}uJI7+(K4_P(e-'
        '*!6e!O`9?)6{Zw;|d)r4PW{iyuF}e(TPo>P7zY=MV2LU%wyh`B$zLd{zf9I%oX)YUM^+e1&3U1Xs`sfB(Ua;af;;2oie?RWXW7e)+HhH>'
        'N)xvgtpw7>y?w)@*hT9^wpw@X5soqXSf9@K-y+(MwxL%HvT-GuPmO6@D5xOKE_waE)`~4*e<EY|%ScYa#tttv$R5ef(tdbAg?yr7A)`NT'
        '`!}m^!xSJ#uDcnF`kVA)8+;A@U%iF0VwV<eAwB0k=<2V42}BM7l9^idz%>`kMn@*F>EOs-M8f*l~-i25iv88)&3$8-'
        'A8I&+=9u;83bxPF2T_QDL>nJ1v~!jtveL1L<*fnUmYPIp*s0V6JM;!_K<hV$VPwW^8p^>?&M%r_bX&)k`qv@-'
        '8}R+v$HW>+jx_p|*&6*dlvVliF%{)Pb&!2)5>EvAl<n_EB3!!RC>NgjnR~J)Y~QpE;+?C<Pu2n%ulb<ADkVPyak)^3$h~pXurac;S{ac>'
        '1fnv<YyiZcbfBe>a`Z#-eI{+_+u*;OT9j(Y|U<Y|na|x+knnANz5C=k{T5)7^dH13-'
        'tE?@TxAq;zlV`&R4w2U~w@wf@%ET6+|l%BGB@O<yPZr&~J@y*=9Sorm5Y%z65w9}6oeq1!qS{k=?3qUmEE`BvGwLg|TT-kZ^UomKE0ue-'
        'v)dPmUMo+xe>$1`<3>$V@-D-qj3EhEb9;v7nchVCKtKQy|I2P%=)C^(;s-'
        'rJcj!#h2q?%JyCYpY`cbbkYX>}!gF2WX~a5Pi+Sy^N(am|MZb;Gr&g-s(aT$95M@JB~kk<+F~*Pz)xt1CT-B0SAr+E8aEUxjblm3C(-'
        'W6RY>{ty*p{B6HWGy)I?)rLp03wYn}bcTTH}?AmbODRAiQxn;n;S|Bt>PC01yAxpPK=A0D6v=Wt{Vr-'
        '$tYScvEqli8Ws&<v|!W^FkivWR_QB}g+Ce0PE-}5mJ6$+oLBgS<hu9F-'
        'OX)f1R<<%zB7QW{ODTSpv5p1U3xu>t!t5OF=o?}SHNnO|{FbO2xz|vSeL35bcftB~cQYSZCuyPNq{sS?sNqgq2?0v%zPVo=dsEm>uS)yb'
        '#cR9d9++wj!RyFIL;`Ixdgx%5&S8CAjEqJ?%PW#a*E_FzXpxsi*PLigoe!kAWwXY;Yle1aj_932zgZKGj1a-'
        'bt8>+tg7U%q_OAb?wY*O+_y1I1^LSNV1v3|94lVa%AKH)Y*S4Z6pB=Kf6#NE0N2X7y2>jM>lT|~}{C|ehE3idj8$lN`(I|jOf2xXUZtxd'
        'V9dLV#iAaMu(Ze)^oA=W@my#0KrQY%3lvtFeGpxW=cm?3HY5QIJgDPvwsc-yEqz}g0XAGsR)jBnl8&e3w2jQO4`Qz^Sonef!MCv4lR+t)'
        '5t@pW#(zgVJRU1OvlcSiEGww`;M8(S^$_A+K!8!h8jbjC~h`c;nu5mmd*1FLP@8Qt|ZZ@nvSnn4Q+NP`q}NZkdf?Gxs6cfH|B+%{CY5b#'
        'A`Z#U<_rnG^86G+t+xw>I!S=}=Y)l^<6CUGkF?fs0g>oQv|Wl4s(8L6>0q9ng}@m$ccZ1zhjrs^RW!cgJT|FGhvfjjNr?L)-'
        'rBGgh(U)BjOn2hI1xh`6FtCqXmT#SCq$JlK-dbVd@?o5~DVh_4?`p7=c+_{bR0}d(P;E>!!Uo1TtM-lWL1m-wlN%w<X&CDUe-'
        '<m^$*wqU9$040}Gl!kkFG%2NJUmK#&L@iTVBvUtiaEmH=#Zg(l8Og3VXy&=03@e&AqdpX@6Mf3u2SK?`p$uy7=s}<utI0TJzlM-f7Ur?-'
        '9zBPVDK4uLGs~{JIWGa5dj|1sQ5N=FdFDbLq__JB{tUtd8i~hn%@x6oX=Gy4ZGW3?o5-'
        'P9M;%W_1o>;BHjNK|7dR+2Eu6K5*jKGF^@1Bk0*Em?GJQ#MIEwM3V#rqRHZrHk}j^Unkb_kT@2fi#7^Ka^tlqVnWZd)C|R#@!x)4vE*U2'
        '-O*KE@Vrn+*OLFyura01H%Y<&Fp^M^Fze>*tXw(iKM%S{wm7rYkl9%IgNO#<W=DU&S1FU5oSh#(*VB$8o8vpB`)azgW415S3m|r6~J^ho'
        'g&fC$Y*b1j7i9Dx76t{yMohrnLK*J8RM5em*Rnya&%WIv}=NpObdUclld`fdkckJU*(1in3QCNGae~6R)V4c6T0bnPy3PXDnMBOkT8yZx'
        '^)NW`_CV5EEY1}!W^psBK67pC2s>hxtR`l(O5Sx1;{}@QBdusRaSrAHziNPD*^;o~CGNVX1vmok+hPa*zZ~UAWm{wEQKlm2+h<|<uRr`&'
        '!wxP;g(w4OjMcBOEid9hd+oo5$NuIZxxN^v2^z_0rEi8~kpJoCefn<q{JyM#S54+;oRUaXakr40JAR+29VPa4<8K)#)qIIy*vsaqt%Hxv'
        'Y{VC8EaiQz^^p8EkDyYM=cq$qlHN>&x;#&$cdvJQu!D&IBB`3gMPLDh&c|8flYI@{D%KGW8N02!oPEp0dSW~-'
        'AR7Xr>fgr(ThQ%J7+=?*B2F?W*KIX|VXtO}mknCHmm7~%+lFIn|7}|mXYpSS=TvaB6EH7n({<aU*Q~Hgm0vH!BRUr+lj)u~1;Vs9%6{Qp'
        '+mL)qH+p`P9R>XQ{ZA)Sm^*asxYZ5#%-6Wqk`y7}c0do}uFO^4E9_%M$OMPRA3FNcH1NTN&SAIS2TIju7hbo^y2HM-'
        '^^5WtXMXLl}g_0M11q~LAVUgGl<UYl!Y;5p{+kDuQ@$pGtXixi8Xx9XWJstn4!?L|z%uN5t65{xi+?6zab;r!8#7lDwy5Dn-'
        'd`*!WWGN1}v6)eUER^;SMS!sY5@B1?IV$HysK&I@vzzsrrrK1T7wg?7ug<RqAFVwEIR6R~sQIBc{rfbuk&t}2yD<g@c<}GL;q@5*3qw_E'
        'Tm'
    )),
    ('advanced_engine', 'f35dc8843da409b896abfb8659b4ae7bb5790624a73d2d6972e63b7ae5e7cd51', (
        'c-pOf+j84Tvfw+vqJ|w43rHYDQl>0XhQjXlr4^^G?yxmIF{@QDAh1Yc3<78Xltj&${iuCDPv-'
        ';nn>I7+f+}1j&%`=xi$qmrWo6~P5~bmA_%1G@B3xCY@F6U?$F{qqibq9Qah8Vn@j9&HEM@#L=S95bX~mp^mpm+@4a?GG@A`wm7rx>JPa{'
        '5ONfw0(tHN}RT|ZoTY{SE4oUT~}P2V2klCxEjZCO?9V1lfuHXh5fGOpr#&Wd=wsRnmp`t6+EWy?Ku4Y!;<j#gO#*eyM_A5kBVJ&l&xL+Z'
        '14EXuM1W&nhjY!yH9<>&$DLnxyv8x5*mk+Oq=FLx2NL0?`Rux+;FiN~&fy=Et)_rHE*S;0cKN)Y%c%F?RH5*BasgyWit2C&fU4QF{0!pg'
        '#Q$v*tYM+PKgCC@{mSDCFUBqGbhYEydbZdb8#gAYPhBCvQB1Cju~1PWDyGUt4G1lTTFSeCnOPSdc<wBkhyWW4<R7nT=U9?DIKOEwvg+0R'
        'd~OW``@Rkr2O6&~WV!KbUQU)c;k5_W14Q*1nE6`;Nh%gtQ|bYw`oGDM7kQZdeV5Ay+hz=h`={!1%=FdPmCv{{SAY6oQGiv`jIHVtM-GeM'
        'SeAo@gElJJP0s7_1%$Bv8EWmtt#0;}Y8t9s<If&*0}K9DV2AfFgk?Q_CVJbxcY)j)MhfnIx{8%uNbD2H{!1NfUSC00CL$0=6;oUifrrz~'
        'B?Yma?=_m^L;zk2NRukSy=kDq?NoGvDdvorZ|iXR88<Bw{ddF-JG^F<LZ<)<ZIYkogI8GCFMf-'
        'o(j3}#P(qGdP0sL0B4Q59kIZL#%Om}Kd?{<`>k1`M`w2ZK*6U}yg6_~c|Z0M=Oi{6FCVFvQ^VyHDy1F@g9p8~k+r<rndJIvxz(YJxjJnk'
        'SwH*C2!LfF7}T9I(h?69j!~O1X1F8hGIy5V)w~XqSYADQO^EQvMmiI`BZSkwsXHd?c7YlxoIDrb2)ypNk~k#ud_S4zibp6`*=N_Q%u#m^'
        'WEe;cijZjbs0GRyU&1DG<5QYc?}Mlsql7Vo}EHZD>z5_NM|`yxj?4o2NO7gQP1q2|<ga1!;^qE2{!phhO(8NEAE=LhxiMmXv@%RJ$d&XP'
        'KV)Q_&3j+R;8bok(=UM2_c466a+WFBjXoA7CW=EqN4g!(_1sHEhf?o}T!puLiQq3ZxbkYqf||pwGf7d9spwiGRu+D0Ij5?M|+4k5{ybGx'
        'jE6IMlcI3m?yBbBmvg5Via7VY1^NiXtnV;g2we(HJl*WyS6|oB9512I#E|&dXtIK7XPotoNTLdugsO&>a*>38Drq71%1RX!Yb{e*)99j('
        'KPHv^Hm9+vIfoeHfr2Xe}8=<C>2;W**x!H@OvCk6)VkG6nS=gOF8%fkpxfz=xjplr)pU?}S5~G6&ep10EW?FHQD!m!mQTnFh@Y`Tz|FSd'
        'eugj1<Bx$u0uKqbb>|4is#;(pa<v%VWd8e*O%+JsY~;O9mP#OnjT(&giTrxhb*u5Sq|-'
        'j&Q$pA5&~4D||O9YYtf8YS1x4`m_<kwfXgQrf{=2(25H2rRgMz(57<DcXzyc;5?-'
        'XXjupg<{u1DI{^4<vmjDA(Y&euT{gO4)eiih8`3WxvwwgZyw!%`3r8QTM7@-'
        'tBZh21wlGc`4>nm5KY?Q+c_szFZ;$)_fB);+7m6$21Id6n8qP?v2Tcq^;2&Rt*C#b4bjiWVqQ30lTbu}w2uu(G1F}eF0L1{NSawXWeRjp'
        '~!E-9(L;x>$cO`;EFHUpdJyN*z)iMWi)1oHx1|0^W-m!wH1EUP`2v;HU4Fai0HHpOLEOtQd;n%-'
        'iUH|;y`tm1%MOEzU3CPi@=_XvAPdxh^uBi`yjCfwLOKKp7K^F<08aNV1ec7d`Hyb>af|FX3|ET9s+Y-^{72xK@1v$#4Up-bsSD*wl&;$?'
        'LEtn^PkaHK7oOnAx;6fN+bD#rs^C5UZ_oCuXWXUEH#*)KsPl~)LzsroWBUDEqNuJmUEn=?PRY1bufWVzVw<ck_L{0GkL@($MMd1#2@#qd'
        '5*>4%}B<vwS*U`@|EqD@#cZpKRk}F7@FrxC8u4jCUt#2FGz+dY_DaaToNja9Y8UBSQd?52dgALX5CP5plsfQc<xvS(;(iUm%gZo_+;a)3'
        '-52$0<)ra@B6lTdiS&k|r4-'
        'h`rfgZTgG4y`|RmAeRiZ)Uwe379BXbmvw_vllKXb7UxB#vT0gQ7x4?kL7K3<5?YtOODWDgJDRO@=a(_OkZWVN8X(ln_GR2mh6Dw6#HA#='
        'yB4ulNuK#9>;Rd*CW>t=Xrs1)^WRX~9x^Cpz3MICQY+&kn?ZGENbw>X-'
        'KQi?Y(}x9sZ&UXx>C(%=SHKv+E>1jNh018Cy1L&CzC6`u;m$DLXvpnem3tm?%^=2!*M6dK?(;xp$45b%uAo7g=t;U5|>unkt+qc}!(#||'
        'KR0swNWjA#|h_zw676A%lz`_A3mkcSt9Fv^NTM9xMR5-'
        'Ivrn^^!dswX4C841C1NL+%k>nR$6rUs7;5ROEuMIxrduNS$^;ETQLsb~P3b9Xbjg`Kj<MvtercHFdCt})1=j?(QHtOe0%I(5uxJQzN{b*'
        ';Uw!)eC?dH4(o)$Ww2HljsbWT{hW(m(6$n44CAmafhMc#Q+5n6b(V55F&9ZOtg(i9A+myFw#c-'
        'hfl{@XZ9m1J56yz`x1t*1Pe?GY<ke_&TL;(;2pS5GB68y>;6RT!WhV-'
        'srXOz48?iVSYsQh3Tg_*QuTlrhARwUeUJ+d^*O-&t91P)c0`ydh%LY`JOw+E>RSfdkJ6U7Ofz{s?9W*|M*VC5e}uNc-fF6m1$U1S=$1A#'
        '})$ne}vF1WXJEBo|o-PG#nwsZ7F))VY23c28a%Z#Tejla1McKxim<Z<aLst4QvFuY6)hh07n6!0za#ap~{5p!-o;(N`3YbT-uUtFa?KPN'
        'x&kG`8RUV%l#JEr68w5+AbgxB7vT+m|_{175H(KDSV%S=Y%r=^Jma3vIxapToqxlXP8~h)?_yem94`>r$J_LOd<&4(6!#%rVWK2H@6hqE'
        '4KqQSGCW1Ac(GbY!6h9x9m>=n<D(ZUv9#jqlSP-$M^``Y-'
        'jI+dv*M*b9Dv2Cc6lZ$Dp+0QTPZpUi(=R1L;3Kdh7}O1weA|UkhWzL(EZ@0`DkGcH49TDj@pi+&msUdQS&W-'
        'owFzH$8ZSPxwE6eiBdc6a0I))oTJ6s9usac>93NBLvLpH%Co^OF4UPC6{TrTdlx3!94h}Sc*w^gH0IiUAR`RVU&Sj1O@_0S(Sh2sfnvW)'
        '0B=ND(ZZ2i<ZF^Jaj>UO8r~O4U}SzP@&PIpU3j&tJ~XdgLq?go5-KD=exhs-Neu7?X9Gz6s&;Oc5^a6yER!K%tO6c@(|HNcyg40kY)u&5'
        'uk@a^$X8fj($sPeJg>9-'
        'nD3x<pS>+$BQOR{@#;MR!<5KH;x`u^8ii?f%zC&82b2Iuyigxd2a7NE0#4kyhcgI=Peo6`>%`{;Mmr9U}YV||1IJVmV1rLy!@l69Kg_Z;'
        'W@PW*Om^hL3)x-QvVtlF3Jba^Nw5q@K?GfJq#kS%(ng~o&vvu7V3+_{(q?HI-'
        't&0s}ZsSd&r7!<tEE5x<UX7+I(awUPeoi%K9Vho$v&JFyId*d~}x;cqmYy9D-'
        'hE&TpBm7F|!|Cq&c%J+6KgN=Eg;jP@F|ChR2W+O#AwM{d2_3yY5&^WpO)3(?qML|e$yB?g<`h$3*GU&(xPp2Zk`1W0oq`7cbIGJfKLlX}'
        'O}u*yD1CI@3zM>Ke-Y=VjR4QK#rxs-'
        'W+9QgzL8gD7UUb2H6wuw+WP(}rw3M~r(5y+n~i#Z1*(6P}m5W9;*nHIq#FHab>!1A|5sO^e;NeMg`7I8+Ik|wQ(AQ~V*PVFZcZh}C2%fo'
        'UfGYl|~gR=sk%k_dlv>DC?3GlB$S0HX}Acq!#3E6G<=%A~k(mNtbTu=@b0ea1CP$7J;3uDgM<+~^+h*IkaOC)8NdFjw2GhhJP({bTqt4}'
        '&QLWHjsxXU0=#+j(6)8HeO7%>lTXbA;R#1W_BNVzo2Y5&w494cg8OjDSMss)P7b_Lp{wJv%PZI^a*RP|&f%;O-'
        'BHIIIp1mTZigjUd;12{q>fsv-'
        's<f;Yo$J`k$t0S7t+G{;h*&y^ZVCx;4YNVPyHqy8qEKN+|o2#gzG6fJhfPgK5EfxsC;b=uHdh~=W#!?mIU^rqS=AVsWSsEl@*1>I2<8%j'
        'Ppx(p&RGT`*`xk)OqSGA|+RUZk;_eubfYUx{N-jW^E#s(i9_?@;x4*9UMYcepG<c@5x-'
        'RRteBOF>8LyL3N$H~xS4ZIdAECY!=loPNP*FxnpPuN0_@a(5J^`xiVS|o8DIf|D5X>kqvWS-@p2dbD*-'
        'eKn7!AWuvywm_H)DJt^IPwdBuq#5po&A#LnGsOwky^Bh<51_Jn4Kd4+cRcuES>9z@jQ<Zyv#K6e)x-'
        'S*eT&f&zyS1ZV8eU@Hj}NQ1B<=$4cUFs2jCp{|kQv7SpT!K+1m?ik#0P^O2E;E%d$2Iw_u_lipBHM4pFih(}YcM#=6g{sVX3C9sl)rov^'
        'Gxlhh$)3Qhr2-2MwCFIhdz5M2W9%JM51`J0jUx0h0a3SRFM?F}no3{>^jG)*$ItMCZ1{lzk5_TJ%XTG1vfv>W#>q?!u@pE45eO?-'
        'K0IBdij-yLvmaxuvjCPvrk0?DbI2aT<eOwka4O;yG%%e82xdlRF)gv+Zs|+VnP(@Dp>2+Q;xu88fz>>JJf28>B8H-'
        'TEQ*KbU|%0Jh;E3p=;lw3>G>aGPleOroOmY^>>of$+BnC+4LRZ9U@AKt3sPGMK&#q|KW<(L(*JQY6Qs9rFna#LKI381h*UM|8~7i5TOLL'
        'm87Y9bN@1k4{qO&CRJy*LVC;>>1QTz9AAl^U3TTcZyn36zUHR9Cf0(NG5kwA{$jQ^gJjG~1rp%;!`23MzzVzRNrv50B_E?s4Q&xpXO7d1'
        '}L3z3vmoa%C5rFiX5Su~N^*&cI)Q3l8N8q}~7#SmBP7{nvL}!eRLkUkD{VtO5j&jV_NoO;}T8Q+#qX2?}vQO#+EHF}|)(KvZVeC61ROdF'
        'r-|d-2`RSMS*{5{ZO3aM_nR&Pc?O$*_2O)u^vrC>+rIaYe+Jsp4w3eVuclL;i#Fj<)0J6O+RQZ=YydTl=O0lL|p1R~u+68c!7r-'
        '7uR%9p$JVV(}XhY44C?2w75X_0I$KZ+Ki(_V!*E3_xIxWkW2c1o(vtw5UX4v+^n8UNl+4T5K>TNn+O|vaVP_2`YquS^+PeR;!N|77#=(E'
        'mdI2}>7ClU0lS!H}wN0ra{S|YL0A1b-'
        'JmTB`u*2pOH5GPSoMsy^42!#lIL&0XYOF=k%U<rx!f#@ae{;tY*5aj@G?9JX)qhH5ySmd`3#Lzt?8#b1*DBR!1V-'
        'iR_@S+c?(eh{|wlH?Xe=s-'
        'w^iR<=l78iA8_fqPK9DJ?TtJ}&%FUVY!CwpUSE%H@h9#NO!$HTPa~j?Yr7rY48fEC&^IT=c!m?Uu_ldTQzL6OL3s-j%B7_M(b-'
        '1z+^(6uO7y?V^r)gY=0K+s?<tEq^r)hTI)fi>F0!%6xo@ZL`!uDhZ|2wAu3%`UBRT%^fMYnFEZ_T!nQyf2*V)(cS*SwxPPFF*dKscr{_F'
        'n>tc#KEM))d@El3IF5QlruyvW-hjeZZcfA~Xmu^X&UYm?YUlV>aD*w_8D|KNt0{=xyOo^kEX72Cv4h-'
        'FuT^kq5Y8iLpjh`DGJAEk274C2aOD%z=?Vk9L@3#0`{e)O;#+?YOb83#<j{!9mHiupM!=2V+}Bn+8FmgomlvKcb|nRs2sPs72T64t9`|p'
        '@H~P6moZKa4>07ZLs?C>etVgSMPoq%JcOyF6yIK5+g(B$U@KuSV;mzyTLdS-J}~4qd<5jR}otDC&Lk!HC8nqSX=Oo3SY3y-'
        '60)%I5C`=O6$mP6#bvpHD7!id*Td{ju1hy0_%Gvd|7p$Bv$p<Hcoe0mN)<slZ`s-'
        'h=8pkvu}3w7_ytxwfJMu4Ce`XGhgFbDjMTq&}(&drut&nOu80aq%u`T23;8g^g|U^hxngs<qTSoz*VYRA%dDh2XpEVg^a>7!T*OCI%yzQ'
        'M`>}IL82X#VlBd<X<u6NNqf}^Mu)H_n6wj&z*tA3h`L6*O%lz*Fm3nheqbNbjg;vG9qcV-'
        '5gv%;WIX=)sZLYMbc6|mqPIZr=3&EL9ZC;TrBz#Ya_lvOI7wYRy%3JX5n$ok>uZ@w{A~4j^w65uSs*X>u#Xpc7_|>-'
        '65tCQAU!)+xT;9Pvcj@16$Y^u@335bcz+Hd?w(P)grYAj1~P!Dl6DdomPHX5N>*(^^4Fqfh$@@Z$*V4k2Sd8aA|r>c#FYJ+DIy$+l71KC'
        'hDPxxDOHkvUG`<A(U7w|5>y={VTfcow>EuybNC?VZlmxNqh{>?me4rV$#r8LppWLQoyU=frrS1|)`I0vPY){hP{s;6DeoOWpL-'
        'O4&T7f>k7xfroSA{o<aH;UiF@0ohd4>FJN6tQh!cck8-'
        '@%;$=1~o#ajnpkLswRv11^#pjL?Qghqv6r~`}6igFti+WXpa_SQ3LEIngx&km#8fR)RswOAvmqo;O9m}#>Ch(HMmDGn`+%(i*vr7?56h8'
        's10#`w)X&#HPP9*fuea5K5>1eKv>bo|rlfgL!$30kOPk+?rOwL7-SQ!cymToZrhSrXmDO-I<-PJB#!(4F7G>qgwt$sz~-'
        'bk?B32A3DqNwNth$<FC1r9T7p5{s4%H9!?0h!tb>)7a=zRFgOkL3^9OrJw%wA%Soy!PO=nxb<?J=ClL-'
        '>8Rl~JZ$1b9D9khWi_JDu<`KsVlr+kENb`jEKm5Va+)Hm0;Zr9Ry6v6e;w4GqoF0Sc7x_{FM=uIBrqCH5G4z};5D`Xug-'
        'l%IA%oPUK~?zWRCEzbnfn5a#$DM-8~vNtM8lR_QI>N$H!he4ipOK)x_fyFP#J$!`2|If(ok1s2aDLmOS8-'
        '5g(5PsYF&D`=7IA8l;m^S~tyP)1V`zMj{~ucgekF>R!C+*tVMVL0`q880|hYly+9V(Pyfuk!XK*EbcfGz6Rx9M85n6AyV{hu_nj2kw3&~'
        '@+91q@m<0LN9!EiOoZ21j%0M#eG`y==my&OBN&5Z=S`sBc%;YZoYX65iyNgI0t-n!t~VMSwDwQ}YHu9+Y>Pm?>_{G5tvd8-'
        'Z(WacgC{gY8wePYz&gBf&yD27VK6!G2#EumX)(pt#;=Zxd!WGMF1AJbSXArg*=E@7hdE|{ovOXaAJ_+3_Re-'
        'G<&6`*ly$)3*dR;msyP9aUSDnk%Ejr%Ve`0Ta*SCncEBdm4ed~+O?;(RZ9YczHTB2OkyEa>Thso)e!`by)%$D|0W)H*x#TCYN}QcC+sc*'
        'U_jtG&@dmVITq0q^!ec8Cq$e(%CmG+C81ig*0nR(P*bx81M_C%0HUQB((-'
        '7)b;`JeP=IPDuZ621T&wj(iQUNi#z*rmxtI(hsm*ETvzO4v;IvvSU#gR^E=|yID2s3PNCQzap8M!hfIgtd7LzB{IZ`%9uTtWxShuQ%d3e'
        'G=TaE5h}cAbJ5&h090L!Dawqt^*;L2--)&VbP~Mi)E`=behY6*BcruHL~K&KpjiXEa-'
        '0nCe+Sd=@m92VKTU=`i*TqrmS`bQg7yh9B#Cb)S7_pUJKJ9N?$UK~n{o)UZAP!=31~N<A}x-4UAy9|SyUazI?-B}C-'
        '{Xl*)s;K&Xr%{VijQRz{jjs<8XK*Kqu$+Ye1Lrxh`fMgj|yZb)}R?Plq9kI(7@7*A1s5>iN9iTlBS-'
        '*BCg>i?+SZAyI8f$;J!yA=`$U<e_Wv)8X-*HKK+IhPvsM>M9odfqVO;9wHdbfe?og5J-'
        '1P0xurgla?aH>%o4k*2o&fy)4(n)Fa&yx|!h&-'
        'W4ucWskPk?aD$Xh18>BvWf{ZqGV$&~q>KOGw2C$DGz*c(k=js0faqhD3D*Zcxy<W@+J!l}eLz4JusLVd=dw>{a$i1?-=;-'
        'hdZ*if|r1buA|2;}MH)IG52po%8!w+umD1WyqD&!-(>MUcaD51I~(LFd-hTu+eRHtWbsPrAvy{8szIc;!boDcjB|LG?CX;r6qY7H2e|=f'
        '<Nh4-%V$ad(JUTlSzj7lG-&vzeXZLfYi~d&IV1bPWa7=OANqmxnAopm%kJi#U-'
        '(o%8E83mIu_Dpw}=F5H2CmV$dhCa4G6mSX`#TyA~#D^_=htBPLrUSqZ<$?(D%y5uM~QvyA(zQrluQti05H0S!lpQR|b5T&-'
        'op6zl(we$x)RUxg@L*=72$Pe0Tax(7?<U;Q*XpF!NR`XF;oNmSI__Ez7R4<Hl{ejpW-'
        '}8e`@Lw~2?;nMrr#NbQ`*LWnSCs=%fxL?_E`nCDfAs8FpsoMPs))lBoC#5+<m^84XA@=Qk_4S8N2#uQM9kW7fbk}vtlkL~C!E-'
        'IdyeI|yYBqO+#SsvJiXXYD9+npd=8&bgZ3yFdkpFJT<CUZ+R)!_mjdq&Drw2$qg_=op$ZE3MKFfHQ)^FAIMAm?{(Utkb!wl|dtM)RoZ|='
        '8;p}A;^^tdnM<;6{0*gHcUFJ)Ni7FXWQ>xR)YM#k!_paH{;oQZmXW~SW0BAzF2xk3}Xv<Z8F&gb4RkulMt;X`|Y|mX@lhsBC4HKCiZJmq'
        '|T)C!r&6QEBrxFk)L?i9=gQ&Qp+fvTOy<a0r39|4Gm`J>g6l3b42<crR@qUhNPyiVSfbO+x)JWAmE_l3d=`DB$rnKBRa<q%)p?R3KjCbY'
        'PEU1rU^;KsF60SYsQCw2S%PO$$FmsgBuYHn7iaR@s!M>m4e_)qJ#z)avXMR*L=&@y%{*OxKbl4_*3X5g7D;vPTmB@r;R}#$bD2W4(BZqC'
        'kYFqFY9`Nl<Suk(!16Fon<QpzA>gX2?x>ur=F1!fny;!+y%GkhHwFei+&uQxph)6F9vU<~@lH6vA0_>S1Cw~)6h>U9TH^(A-'
        'R9yrekOn<@Wt0VzB&`4x&#^M_D1ssG*C>pk6C;jc2z{ub7{x%=BT*7G=u2!jjI=P(7nA8&r_gJD#gsh4tTv!2X>D7vWF$<76*iQ0D;tJ('
        'zJYNUCT}LE^%V1ttnT#&Zzlq0ls$xddb4d*VD&e;nMt@lS8p!@e{<m;msOS@B>>bPykuR@4GQ)zLA1|Bp;a*aa{VLw>$|`G_Uq-'
        '<^{}CItyNni-u-la`TK{i{))Lz$Lhd~c>h=rzk2sO`^(jr55HW#`|+195GI4^QSU!|`tt7mhxfy_UC@Bdg0aqxn|B;sS#@QBmBZ{lS{k9'
        's^G1&y0T`kGJrba$@O1ny;biI7L;EGLLI0H}9joxilj%V%E8YZOF&`a|eGmS%*9By5qSY?I(`p_(-'
        '#Y*tI4X&J)1BkJWhT8oH`PATYw%2sbl+~#p^)1=tF6o1g1)!%bQ-y_*pXV2+pLzcz}g#P71WJ7-'
        'k7r=xUEwWRR83NfXYEy$RM}N6VVn@o2I5EEU;9m+F)#4`2brUhT!|5<%ll95jW*x@e)|gG~3eqLgA|E-'
        '=HqSoIG6y8<+IP%6$d_JQz<8ZPR)Ofis{1uFgtwKE9=l7t=IA{YbW$KSs%}ghdL2tc!lFcX>%_)G-'
        '$xV(K>*zQ=Lx@EyAg(ydn><q5ixhoQPfyYBMOX5Bn^mq>wcK|lqg_O#2qKfl<K8++CTyu-'
        '@VJ}1r@d*A08HdrnN*^xZF%v)JDInrC^jbT85;2%Tu_7~-FUU@^Yw#62{h*gRwseY1nl&=ic^(#v;)O0V6w$yCvJmz4-'
        '_E}P&qUbpkV#C$0tfl{X7(&@Rzap!GvT>k-Z&JQ6ye>qo*Z#_;OI>7IcbRp=RjhBVuA%UHXo5#V-'
        'KTC1f{c18YaLI>V;k!kEXxXd%|7Z`KV&+6QlC6Eo4n!&-'
        '*A+z6g)qJAG9t=%b5KYKvASEd<u!4IPO8=r)+;Gf^;5e4Q*6=ckY%MbPeZr3}$JOM&$K8OFawr`$D^}39NgqhjZ)ezl^Pl$R*2Z!Ed<Us'
        'F0o=PRjCZh5+c}vz_C#1efh>Cik^H*PDQVJU=C>(%~Hcf3_Rw(*@^hT_QCH{BO0VY2A)FyUvLfDRp&U>(o|R3e!G_<Q<rSKA7NF52b(Y4'
        'QoGI(x!gUuZGoqb1($mpdo!%8b}@l8*b7wOKTZ*iA8UFCsMZ0g~z}Rw@jdIy}KZA@3_3<r7IMM<~<oWm;TbJU#TuY<dk6h>epMny5OSma'
        ';mhaSpVGtP~^UbbMaFgnB#X*@FR5@K9I-S-cbHN0eUDiX8&y3dS9G?Xy3k(V9Fl!!BwV;(3h*ge4E(8PV${K@Eo9PQyobsWaV4W{W!@Ya'
        '$NLkhnTbazQLFraD8NdgWZS4w(K6S?puPN;E1Mecn~wtuP1OQ+3hxP>eF2SrrFfO2?_GAjPD_fwAV^L47;sCHBEXCk?Kkd!T)&#W^pneA'
        'Jh#;;uWc*qvqQUhZ6&kVZ<AdjFvViRd!*)sx!@a4Tf4&{nXXgzuCul+{Ddr6~eAD=ma!y(A6!?X_j;WdRHBbU+%X3J8qb(_MYp;u3gu?p'
        'o6+^t|oCCn6h<(!>t|1%zDF`wa4T0p0UbPThv~e9*>B_56h$TYWM9{fyDT#+0PGCkT?<ko0TwzL}SXYh}UagppDS8*49eEFovv(P~R&`X'
        '{RR9aJ(>sf2#9LZpYF=>_CJT>W&~|ddnL2%9$A6us${ntFZ@AzbP5$5+~!Hh~>)L%DW-9U;+3d3X3FjqIvV0oMM-jb1<~QbmopOOr37s2'
        'QhmSV9z(f>)E`0bKcDNU(XnJ7}jse7Rxe_Z);*DdtASIi5-'
        'n>?ZVpz`T`62J7t{6!x#Sp`Tjrb+u4B=gSIdtRXK^WP5&4ITriD?6Yt<)`%mk}BHLDpaq8}j`vv+c_plpt`+^K{r)N3Gm(x0FgI8s%nqC'
        'micqI!Q7#$9k=OX1xCE9pr&}z@hJt)HMkt&a9|9TZ_p(4%Y>PD&?B`EUpMS>mNXfim%ckh26jmMLrH~c6r9R;qcHX^58|1K!1W-'
        'W)Fa)RShTwe$azqzR+d?U2?8WDUmYWEu6@$Ic)LR~*h5@g}w)ou&KiS#d=0Ef<yZjJvG;HlqgCF*<e#l1LHi5<Qa_i@P;^hn?vAY8)%Je'
        '=^+NsZ1=^(AW~Vh*0(^CZi87gEQ7)HU6;M!zWvC<#>KrFm1^$hwQIrZ3DFC0WUP#Jr|d!_mXr4b<Ij2k-'
        '^2)}qTX<R#?IWg8@SkD%4`P|Zd!tg<8kTAU&kzSh6kO8v?$akrQF4O?_Xsy0JiIHi6SMw?8pfJ{2sixrqMvscQO6Y#nhu%j_>e;UP_3;('
        'xf_g=dy6*~x;{Owwl6O~ZqO!RvU%I5KF7oIcZF?IA#kc^{Y{3R1;J=xjWpV`&rSEi4d43>Mz(iuf*4ZeG?et8!%a<*nsn6eCi;fyFs=di'
        'u78oD2*?85y7aQ{u+Jt}V()+gNcX{Ui}BPhKXOJqnhv^u1z-N9B79sWv}gyT_o7xJ$G`Kt^KK|}{*^b6bYAPnDN#^Wb-cxIq)r5gNYyjp'
        '=>TozfLhVX%;{Q^k*T3PK2{|s>Xs|3W<DV#d+S847p^iC1IT2(4f3z17WL`C=;W^!KOOPvxyX&i3)J+#}4;B1VNrf)<Q`fLPdO3xrN^wJ'
        '*m2-'
        'g`htwKb+?S1+AFCKh^PvDb#Yz7~Yn+@U#NBbhc_p%$iE`r7Ynwl*#d<=qFy|?C#M&j}!`jvns*3+TzfuxpnWB|sclW`|w!2dEyb||03yq'
        'asprCD#l_t+w-'
        'k=e>klfa#^)9oAZS3sut7T*Aucgkvcm27z1>jDTP>WSN?!I%!e)V~Y`u>IXK<{7Y!Q0>Ft!m`BGpv2|?SnS!YJNSP}A^P6'
    )),
    ('validation_io', 'b0c336a3a8b3e29e3a504b4533ed5eeccb56d527e7e05edcc2ebe4606857cf79', (
        'c-oa#O>f&c5WVYH40NgFDozlfhdOmGsZ#{ob%CVK#V`a$8rf_qQYEQ0-'
        'lG3~hZH5t%?DZ|v1xkv=JCzrP;nez$r2l+H)%m*g&JjnU{PR=R_~xtb;C3_N@h?g4GbnVv&9z62~>h_wiZl*l0wupik51$0JwpBCU^-'
        'yxZSGOLRbk(^9`5Bm1Z?&z$Lew2{4vf`ik<NYb9%9ltuTr<%K{XNQ$+n4KP`P<uw99Z+zGkK5r=(uN$Q)1WJVlq_Tue8v?t<dK$%X993G'
        '?kmprvTa9@hRuCcuGs~4UQJCVYAIY|EcEAX*>8Bc^Koj)WlpQ8Y+7I@4KQwFd5;swFy|`RlFP0aJ{NmHcFaIuYU=B&_*ykqBAVy|vUb~T'
        '`h1+hpMOJdTaR~0~9h2YuIE|vH#40F^`B1LJ@VXTwUO8*Z>3e3`WvA>AaWJ34(K*<*5x6=CFL`0tvk2%#VqJ7u!0aRmXn6rvw{{Cwk*tG'
        'f#12f`G=fuA`|1u^$q)|gdaxB0>E`P5$Kzm}iAS9HQhjZWb(I1o?-'
        'bJ|d|2Mxz%eZTxdb$%7;W#&!<EV}sQCkBDwHlMWr!eulHppazH?k4+?!9ZI~1gA@I%<9|E#$&^virzDB)f1KDmU)<>QtAtuxnXr!?8t(y'
        'Tok1qH;JX51j$I#LU*l}_Tl8W%UWAQ4LgVf75+*RJBzuZqhgK%}Y1q*U{LI-'
        '!cyEPF(6O+1=ilJ+bF%7hlB?@mu%WYV!<q>(cuoSt+CtdrNo9o|S6!kp2+lWRRE4_TGdF+t6Xq@T{vB{B0?KJTl6l2cALQ*J7E?9s)C<w'
        'U%+y9Vc#P|T+3lnIffogcjdFKnPt(lRa$EZLH}i#zxZeit9)?UMP!H!XQpCF!)(s*x;7&rkj;p93*aJI``iXi|+1ilzP2f!OC9rK_o+>G'
        '^O@52hXAaVUF8(sz$T-SnTa^Slam-}OGIT0zaV#|x!pa)W223gCmKcBO>xe}TK5KT7{(je;{xE9`vSN|K*_kDJ+Fd-'
        'r_9U1kbrpj9%ozL-ua=19XdOZIKBo0fih*!1X??E7Mf)k=u2Pp(HYd2%bt{5qzFX?WQ(5wGXYf|BlvCb8LGdpC5k>-'
        '1<4U}cK+3%e&B1saFfO4s)j8fog^Od4;uQ8RzW=OY}~(j629YebVg!?Menj^$sCZ;kOrSN&y_?2aB%Uxr=w;A-'
        '$6Z;~NvHY&IibA*<z5Arn1aA%d=A8@LV7jwI7luf#rX;<2TMHZ$*ZHM`ezkAH!g_-'
        '{xYPL_IYvYVD^GZ22%y;cSYh<#|_B)};<dj6so`7rKzBxeq-ZE*qz~_*%>DxC4NWCham$TZ-%$_RABX-'
        '`#e+bxH{&}?FipUMC>4v8h*k1x!93C9X-'
        '+y!1PlGg?x`<>njA!we<?Z77>hfyweg>_f=F;E%XJjYSU}3|9oqLv3t#+LVEi0)j#2xyt;3RtdlEF|}sDJc75(brT'
    )),
    ('flight_dynamics_v4', 'b66d0ac79b0a9d616ec39c67c0a55695652a1ddf8809f4e6d5fa3b7f74ebbc54', (
        'c-oy=`EMIH7XDp-'
        '1(Bi~Y0Mc*a_q)RRiwzZ0ot@kvcUp|A<#3FG|gNlXKX8f{eF+f9Fm+CEBKh<JNWqcj*rxnCr{oNiL5046g9gDU;puzt<wjYg!gH&VS#Rx'
        'DmS8*0jn}m)SMM%%>?@>bSyG9n@w3tqUk!73CqhwX3=Q$R*76LSXL6xhc~ZZusgX;<4m63l*x|0{fLn_O(qrFiF+1<@9TZ4C0na9XSHf1'
        'W>NK)k5rjuOr@J`ZS6_Q`+`-aPV4kea#p2vyalh8VN<o^5qqb~rbuXJObqR5y&I*P(d1!WhBU6qTGVM-usY49W`RgmsAMKcL0~vrQLMbt'
        '4D%a}Sl*x8olaxDz|2LK7E&m-E>$c~^D-wl!r(LsjShL+m}xX;LMn_DB2Qx`vT7?%Z)7d_2pow5Q_S#hgGH&bYBCKWj;$!tTC%t-'
        'YPf<Wa-GT8>6_^~EL8$0I`T-'
        'wHD)UnO;&8040b^=R`)NqQQ{6uiP(pd=~}?6SVJw9)G4^@M%`dLidZsTVFK0Ud_;EK3%%tKHtv+`HE4Dr@RWwhP{~=T*qzAI1XCqUNs)z'
        'cW~{`zhGh@}8@!_7xQ}v;DQOy*9LuU+^n>l==_}Tg%qyY9O(t1dusqe|5QK`*n$_i~i@z6)R@0w|2_@ODpFaKPckHu(;d!HL6Sp^#iMm$'
        'lO+%*EB}>8m-`{_WMo*qR8JXx?t=0|PAy+Gw<`tq48VVOH+OlAw!WaV2I-'
        'YiD<OXt4Z~b%8<kgM|&5CL?ddHURGI~C}xHunu{P!PIyw0Ne{CqSTVRzfGTopItYMa78St;Q2dW%?T-zx1X6Bhmgy-'
        '|?}rNnMAV)$S&uTy0M1)k9eQ8GxS51jb~q}8|i$=O*r4`;M1ca+R!5s^bkm##RAD_BV@ilB<jnsECH-R9MD!mzmRa~h-'
        'jymr8HBXHu~K(8L^)lRRH*pGr1X_h#?GNbglit8YT9$f{Ro>&M|1EblGoEYNnnh%6@E2N19+;J<UNsXqFzEDW3qtV*%#kC_|Pu#{yDrc*'
        'ttiQ8uCK(sK?K--9aQH2F;Tpql#?mztH+sD1$O-$IO{3ZQqP0^ADx+US*2p&~+-'
        'm&v1FBPKYHbK7o1#D^4QY_pO(I#@)EX7smTo2sDteWCYF&>4k;9JLV6ge{m<kT}|6rdgFdDX6_*e)?W9Qbv=XR(QyP$^iJ|agdPxm}yw&'
        'i7;m1E@XH(4w{qqR&%wupT4o$c3Dtxdl&>I>7kbWQ$$`c4*7p)k+p?CYjis<`WaA?i<P)wPY1jX(VHnZ1Ag=ELXrZ{NRpJ$Y*M=3rWV-M'
        'Cnwte;0yLm`DLlq|8z=N)+(T~1vBiei;_gn4vs3AqFz-'
        'l7Hc$<yfU!iLPlLvONZW(Rk>{@_pNcJ$k~fqrN3!>ZmYl+IPrQP0nv)p0WIPda_}lN(;4Pd2ek2Gr+HC>sc!&E4qA^uvzyEPC!KGLv`I)'
        'B1ArmI@>L>v?8PkMlj--01!^p?3>NxBX$iH5fmfA5u=?Zgtb8H&LeEqadj!L2n+I?VWHaHhq5XY1M#m-'
        'pYs+Ib0Tfg|kabEcO(FGWQfUP&Yqs#}CY#O+D?%R8yzXLyT1|nw35>C54TXUnQ~^?{n`|;_zPGV>#=wSOfPMDEbV?OF}170oTzR6<{<uR'
        'i|RCWwxI5Dq>rel2mNQS457G8>r^yX}9xF-'
        '@Z}R@aoVlbEAvOzLdzZO1Fg^h!;@Pr_z%8xGCjzQ)TkXG4=mlcLt%c{w7gn!*xcumWc{Wc^IHWnwyPHu%e<ToOyTIH)Lg5*O{C|qgLyuG'
        'P{#3OV_nQJQf%`I4TN}?Euz*lXeV14v10bcnJX#0k5nbQLE~M0r&=mUzp=SAKC)XrdfKU(k3U|*J1w-'
        '^O5yYf4S{5x)`Cb8Mc}IyzKl+FK4r8x)>Dr!(P^wY%6MU9WWgfMrSjYpXwLDH21VVqHC6k_-hZFTHtFsoojHW?`RXh=YiYRptUOq=7EWg'
        'eu-'
        '}7kdunsp`dwbW5^Zx;N9u1ZutbLT~o9H=t~~=Z+?fbA8B&T$>jsNt~^~|cMG~1OxQCsbhAyld&;LhF{`+XHKPQ_8<A}yP2n{b{ASWyzF#'
        '#I63*(cE3m=aje4ztf9MAVRQ<xD=g$Mrjr^zRqF@QBO#tUug@kwJ=R_ko<aJGcG3_78k%~f(XP1+|i=@w#VRvbnnd6bkgRoqOKt+g{Q%a'
        '*#f9A@C+N1?e8OdK1ZYDT-(L-l8f=D_TTkDhQiF@#!Fo-imQYYf`7h)t0?D8<4$?$@^80z}&WIuIKh|l4KR>RLMy8K6LZ6!S3JC6V`e(b'
        '@PqX4j1K}{XAMQOtAOY494-WDc+uOHZ_n$)65UVY$i_-'
        'o$qJAThAe#;dHoZ|+malm|@2%bouDDD+pL;O{l=O|?mKP1yFghB~RzajT3a~j7%rYYz+XG&!E=W5Dl*OA5OsC?<2IpOQ^u8+ex9XIW4{X'
        'vwp#UV~S%uu=-'
        'SQUVc{rYd~{_$SnpeDzv5Cprc>Ei6#3>hDpp|iz>9kM5Q8oF3q+M$8>4C!()n_Jp}vkZMUU(B4oFXqhkFh<9A+t#Hyd39{(o=j{6*t~Pc'
        'G~nrCKECITr4j0ANrci{eQ}mn<Ga%@Je={FSLNH=u-'
        'bk>Zj9Q30++jbVxRWPi5qOINg|Et5A07R>EwzS=!HX;<|);E%I6m>MyIt+fbLpe{YF9LZ3=B`0F+O2-'
        '9pdx8Ws)L6ldI^Jnk%kC*A7e#G9YA=J!hgsl!r1LJ6A(<U7A|)3%r17P^WWbQ~u<>Xn<vSrdmpy)hP}n(4e%9-9V<t-'
        '_`rd{5`jIr%kBF0!)Nba!Kkzi4=2mjO^lw~1+NuAj^$L$RTwCPf%I^WI_ugDUi(Q-!p+?ht584U8te-AT|@phw!IS<Iuv-VB{895lNbx`'
        '|z9w02%iui1(9Xum!O!{G2K$EOtZMtm!>HJ#wxcG$z*>3AM|>FDMR=NS5jx*nUUnu8mGcxCQJea0N_GENdf6}0nr31z`~kO-'
        '&8VnwGYZ(}z<mU=iR1-'
        'C(GbUVDCk@1RdcAe7V&^jDI0(D?r(M4uAv}dpdqWXIQ+l5r;^T0an(a`wrXMbk`=e7S|TWb{D<RT5RLpwaY0PO0#WuklBmAS(+tkGxovX'
        '?>Ah~wv-Q-~a;3n{8$CWudM_!@``_jam6QW2|C>#+$i?pi?Gf1H{FJP3X6*v0*}9O&vXy)+{(GpbTMwY2b;uI0h5ZaJ>6Ud^t#f1F-'
        'ph0Zbq+q{y&{hRZeozBK%t-LeFuGfdYEl!MMh9}#-'
        '8!fu7hOz>`<11%5cmCzxL%cJH`}wYcjSiget_<mJKNZ=EI+Ahb;t*?eXHSbNwh|y0p-'
        '#tQJ0dcBlefz4z;ZSRN^}szVtPu>77r}R5PQ1yx+ku_8?P;G!lOGk3I#f&vuaoFYbE|dy)V^{-EU<<P-'
        'Hnhe*I^IG3@MnBwM=U)?NiB@jc?-'
        'HUK**e(1@aXa<Lao&&$5!*~q;Tf{lxgM}T^{r&ydXV_0!0bdT_Wu7edk>%e3OOGm#fXbw{d+m3bua*OB%P}7%o=+PG*Y?>?k~$El^W)s{'
        'G47kkxRYbt@o`>+Rg=|A!!i~pz(V+qPm;;$MUQ@UdA%Hu+vgLri&Z-'
        'L5na(gPeS+9mz&Lcf_V8yk|uVojN3`7jDUBWLIN(X)~U>r;ie`s?xkT`^xx1X(r-'
        '9GBD<U6UO{R(2uP>Kx~7oHvJ13<mM%LWG2usOlpKb3B&?6H)2;l#61Rkjhd$mby^a*PqsGKG9+^02`K+Sv$5!;!9m)q?xsU_L#*@Z@efQ'
        'DVB-UB>OA`A$aH?~eDM8EQjgd+nEa|4Os4Cek?FK`alG1&1VvjFI9rYCZFvU^l#7)~@Hgx=Wb}=3O57u4FQv'
    )),
    ('sixdof_engine_v4', '0d18ddf4aa95565a4cee8eacca82eca47578dcae16415237d5a825d675768448', (
        'c-pO8|8v{6lK<U*1+ST&L<^!x$?_N3^f+zOG;>M%E_VBF&Zpr}BqXt>NQR^wD{uP0-'
        '!1@>00}C7?=E$1i2xQ0V6pqz#S%aK@Wa!ok<)_ud9X2xXuFI3GR@~kk|uGKz+>ZV{N}e`jAEDP>Dn);F*2egq#1=@Ng3~rkmk`74UPAo?'
        'oQr)y1TQEj^1yxG%x+6oXd5MtzT{^{E7lUHnJGXjT3_wWwgaUjSPwl!_TSl>9>2i;n9;HM<KLJ6U(^YM1=uANm?4KI9hK?qexffS7=9!G'
        'T%|-'
        'vtM8b5jKrh&>1!)2K3+`1vJNJz;5C{MQc`M6#Js2TO>^lzbvD27gA#t#U;(LC*KItZ5Gol(DU=X@%q=#cjO5B_l?yqjtyrvE=rsl3Uic;'
        '&oFt=74RW-^!7h~v5tQD;fJGDo^B1#TkXnSPCd_v*n}B=0#sSmf>(h-'
        '`$6m%1ud#N^&^r~n8Sebnn4=JG+<5Zn)J(#9*HkF_tocQx6Ss34|AARk1{_A;Q{<);Sr}ulQoQ?A~yV*`<WNgHDMoeKO`IKhf%Vgp4sG='
        '*S~+bzat-id-L{={QT?tiRXBim;BSjyPT0DL-TLt@8Z(K$xRcO+kymnS`=QH`@y5PC4QVH>-wAbXIju~V{+#F@X1dC8hR?1uPIQ-'
        'U{ZcglU1}PAiw3tgY}L>g3A&GUX+6{fP6&DoXxaXe5EwwRJlBP;Xd)VQBZiWH>%;kBX`vfcpc@d+}~1?i^cMiC84{R=3bbVBuXM&caMpI'
        'e&F)I{P7g3(u}Zu1oWP0L6&K__qP5zp~y1l|GSjLdU!Y#Hmfu#fQ1kVnMX&sH@#o~+nrq*A78)2PmX<M{CxlU1Ad;s=l36f=a20pwsKQg'
        'L)USv>FMdk+1dHjc1};H=TlyI#tP4@3&***oL)>W&(H14$@$UIZ7rcg;N%-'
        'k+<VxS<`H{j+(plzPWbW+Tgzq6*%)Di++{T2+qBV5wF26vA&m)B0M#CcielG0jtzT(u3+CsC{q@0H#X07MRN)=<M~k@<o*hjQ&HxIYYgP'
        '3jL8$&2*qv-RAo{FMkkBPwDgK-'
        'z4g6qR?I<YQojT;wl$ft8h#MaxT!IbtJoK;$~pl#f0An{bZk}$1Sl^(KjsR_+cs642~av>l|s4#MUtgO>7n3x-'
        'mswYO3Wmqvdd^bG%dA~S(isEHj~cOxONR}YZVzqp~(t-GIi#P;N^Z)P~#6j-'
        'qE+H?S=!`Bt`VxG7``$?wbPWTGrGtwkL%F7G*zBI%eI0%>i{%Qg8F^-%$2m&|d)kaO2bTW822B-'
        '!y&T;9$0#YnS!bUYu)pf|dbmHI=%{1J%CA*Bxv(X2no9py^0=WdQ=(h%ilnW}4A1gXgyBxxMDsI3wSklGmdC!7h)En5;bd@zzhgT?X5f4'
        'xdQ|3M)yU$cE%XY6^FAo84}#`%%umKe|a~0T}S(wN<3?6CD-'
        '=YM9nND}HvLM~}CR2>Z_VBsQUYI*e&Dd^XMT<L!p;9+wpZefAgnc)ZPT9|ftLf;~)(XEMx2nRRcTOlqz{$us2Oal6b^CgYwjXhK60mL%X'
        '`sZi~^1ns>aCFPziwrg9bvqsCD2~#bAP5`zOLCaXCA)9Xj3BE?<26ZKv_-'
        't2Nd>{Tq1AkXY_yK@}fHt5;mSAo{+EF&3KL#McgoW@h%hP}s#W+cEQDK9pe9~74R|Kt-'
        'f{PYYe4ybCX<S!EDFGti_%ns4Fq9`hkI+I|)nGZ*YF*CEtJQ-|rVDrV;E+@NGa<8u8<woz66-'
        'y|Yy^+P2WxgLT20O4u!J8Xe_wEEPUzQXcXj(bTHe-'
        '6!o`X0A>Tc?GKTTW@Ym~{GVLK(J*j6@YC|do&3gFaE^mhoOQOT$<B55@6q5DjVLm1E>B1PfMpM(W?TceE<S!4?IhoG8HBPG<Sq9_FZdV|'
        '*Uu1E`k%Pp7$SN``z%(5ObJ^&b<C}5^X!0Qotv6sPe_>D}jDm8(1yiVmsEC+220(x)1R8C?HNpKps}#Z8XE2;7aB>5oZhW_cV1yY!)C|x'
        'r08J2(&5ji5E)S?Hfu~_0Xk=Lo#uh8P!>Sp01c3#AQAM!DWpavGajZUvCtVN&!wsw&>aeD+JNWy@j}v<`8yGZ(5%!ShJmmm+V5(7Kb3n1'
        '_7AkXO2_?h8D!`hJCZI)~ratudo%(0CJ#FfvLHJICOWVF|8ZeY_xW#ljJr!N-Nts6svw><3T?Ii7eG)V#89rAq53zXAF&La+MM)-'
        'W1!=agR~(*an~3)tQvSEx0HQ5p>iQObA4>i5W?#W6bKx=m^{C*7eMy>1Vd*;73CEHG$Iz^B^K_Sh99iM6!rjNb4sQjP3zVrxpJ={s>Xxl'
        'qux=i}nJ5ii{`lmCW7#WavRg$0@m>XG9=FQ`?iGNo+`)8%rF11oKRn()T+W@z!o1ysZa7<vU^)Ff_xHmG9P@*-'
        'AndQbAhsp6;B$hD4weMGPMYBIJ?ugCEJ)LbsF|^vVdSre50@6<KZZ3U>_=4cH*Vdj;6^N?5FG!j2b$hR?+!!_s3pbl)CASXml;MA^BwGN'
        'b|<V>kQPIwI&6ADt@4V}oTRH&K}%Tcwxa~f=%7?zUci4Qk$N`!92d6ps}4YCv8~?daqQ<CItcJtFbLqhWPA4$IE+^)3@kF>{^CPZ_8l!p'
        'R=W5xr+<PjbnOOki$eCfD!mfl0*XNlK}o7zJ!_~WQQ<NmJ{%-|GB8=c+J_7iDUX{`tfFTR&;fN3$us-'
        '218@M&OhV=Ha5mAapEGyfWxt0eXAy|dRsmngm;%HD0|Htcr%IW;dSG5zcV3reF`$@e8B?hc6i!@j+`*^cKED6-'
        '`orKVFPA_BJymILLRv&Q5Z#1pF=Rzm3DuZ>E!>BN&{0X5r=aD64GqD2uAX!BfF5uo`wzASdyI_ViG8u&=cy~nN39&VtV_~Ymjqzc3JMbn'
        '0J*y+QjMiAFFBhT@EDmUtOwT8i?57NHA);a0FH{G#($Ra3H2J_g(sLw0Bn>;zXk<P0$AuBOdM+>sFkUKy6dDeiw0-'
        'SXyRb#v#fXUrG|J+4R3r<q4mPmj}mpE(c>z(haGfg5rO9;KwAJ8$r_+OD({gafsRtB=SKvB7tQXb+lYYU<HsaRQG@JCe@#gNUsi5A(gRw'
        'gRGgqk)x`l5Ai^TUJrVYc_)Ogb%vUvvN2;-'
        'Sh)7&1j?!tYw(L%ZLP<85)xTH9Pav3M2IEl(PXts=xyy&a%WShRFv5wU(Fi0U2}&bf7BqkIVP6%NLXQdZ{^4wX$;?ZolN-SYr*uNb@Oui'
        'T7HfOU$S=GwD;B2NAvJO(A_5p$Nt$o{cnJGPP+-'
        '0Wk|0I&jk?noA!!3D^D$DmnlY{>#Ih&$qL0AH)iKfb!7Db7TGqi|3R8tP%in;(HO*^lGMjfS3J1cUcL}Q0$|<h=h9<&XxyNrYSO+p*(zW'
        'pNG_E<fzazJf>>&$a>lq^iC(^7ugQoh!WZ_l;(joiWmY;VysETqw4nbQo&2Ge?7MNQX@fZoB>^yyDssp~UO^+KgPQSKji9n^|FNq-'
        'L6$_%nvK44r0d!#p0c^|=VKh8K*QF(lxDS7n!IEnmuy@!xu(6xh3UatGS@Hj94Lmk#;s&_<!$vO5@jF;a`*e2v?f?m{ZCgax#AD+NjToe'
        'HjOLBwA#kpe$#C5Cit8qSZ7+@p?**GS4XF(X04vz+Gn5F0wo1B7o6X?}b+L1CTuVvC^q5bvu9eemRr5LK8$-'
        '8uIKdlb&CYF;orEl_9i<kZ+UCo4SM5t0^f>@y2ojZnZ4L5h2_Fl8C51wIHkSae=x{4Mpj;q?h1~_NWvj^HMZ>T}|6U!nIB{VBrv>Hxjd?'
        '4nr8cm5*V(4Opv4-'
        '4c@85a0%KA(nG}nd8>^f)=RGYtu#+=i#E)3usIuu}0geDTff5o{JxpoAHZD#6O!tTsTI34w+W9e)vD}4gTH=unVui6E;Tk0A7Wax!0?lc'
        'epR@?kXYk~M2xF+P@9uv8`1Xwv01AngQ4HFN2ckML_Kly1c+h72r>Z!Tl;PO|_RKS=c$<`C1>~xe%jb^8;&6SVWawDuunPA1?3fcE6~VD'
        '%*=9$!C8*pLRzV#J@sGr@rZbhsb3(xV>I*7jAyI{i#}+v~^GE?MKT1BS+lth3G^#nw8KTFp1YJ}H@kEp0Y`fB_)ev|vTan<eE>^u|L`T@'
        '5Jy(FrbQ}8XNCMZ+<e2R;a2STx<l?wA$>nivcwzSN$xN!_oWm21s$z6Se5JPdWC8-Av+(-'
        'bRWOJnDuGpYrIy0dZCuvdA<!Fhzg4Z(_yt;nvNXHZ(TO>9oHkq%84O5au`Vs{1y;wWLqUGbW)t^5rG-EW-L?hNTr**ItQk?-'
        '^i0$VR^IGD>P?eaEx|8bp>#)zT5Y<jLkBV!=-hVdvteQNlR|0mbrZ$ZXxvj~-'
        '^+3H0HzIoZXDR~UCC^!Sy`IJbX5+!v(oFzUG=(g(vH9*iKZL#X2Lj>_!~ROV3%X}{5x=`y`(nc)zj`EOPf%eB(|$`aaB5mIwWyim6uNQB'
        'J&fsbjGFKDSE|iL6wrRW}aH*Ju>qHPu3fDY=#oc{nVUpQ>cEWA*UWLG}#m25!VJxI5#XkLqp={G)JV{VrExh)8qPx<e0I-'
        'Ee@zgn8*&Pkh;_vQ+wRFy2m&d-HM5uIO9aBnCL5lG)Q1Bq$Ettn^TQgY9;URmX_cnCv2MEBa0%-'
        '<w)(xya_DvJ=oCT(L>zTiKf?~8+T}3Opc`($hBLmz-'
        '!mB07goMDARorx{4XT*RNfsa_Zg>wG?XO+O3m8*xtd_RR`QL?)5ObtDucil^ylKV}f;^x?>489Cf5XG}CA#0C`LSGoKgHiZz`Op+-'
        '*%J<RWTfulbCiR8N^s<`gZ8M6w<;=z$_o_RcP<U0KmCWZXe`NEj%TuYTl9(7d30o8$pQfXlH(gG7nlW<t-'
        'H$i2U+!oruU<QVV_%&8W?L$}^^&u+m$)OQ4?)0|imGQv`FAF-O2~(^LNrSdaO4e5$o}*6)k^^e2{3tF!pJo(wTD-q%(MwXp-'
        '+{$OOlOEwT~LlGo_ZK81qoaY^#6U4-'
        '>^<o^rkTVtmCV<K(;VsXx`M(nK!Ox2IP1%Cg>dL1jqeHr`@1NvQl(+FmIB<YIP{1BBG_k!b&O~(YU8R^L%*4!`c`8af}xAkYS1suoCaCC'
        '%sClxk>ZrTbiKG#6~rY<aT3;Kydl+-TtUJoMkj54cya+xed=!huDuvnzFr8<|t*~lrq4UJ)JNuRG-fB`QZ5=1EIqCGx*R$K{ix)!c~)z3'
        'l6spqr%5!!vxF0e3_>4uw^8A+E;2#YQ(g-'
        'n2E|z4d$xhQmOr=x^wbDV%win&x%k9@+fz=l6rV_r9C)m?`MU2cf<Z)jwdy~+P5qPZvf9GN7V_p?1V99M2J@Fdodcw6a3##K}n+OrGu$p'
        'cut_VB*>}ku8y5k+ak{S%xcbE6~D3)MI0eE9zQ2I&Q3+pyI3VlL~_We$@@?zpzRbfxRn;og34XqwzH*pH$<+ZK@HqZY#$!WjR;ORj%7Mu'
        '=!Ts)2g)7x7su1lgKkZDS(h44%P7zF4q7vC#`SGCKa8Ur?Mm~B6UcB{A;XHUc#hendTFR)H)<N2h}odxsxzEOp6k(uv_k>kv`%}`7G9}G'
        'v}HdP?y3rKGb{<86%;zAkfpEa4C>i=W&9nzu)t4@B^BNjxLE8cAliZ|T7lV!1wN?w?4cx_ER44X5#~v4sb8?;MV66bu!BoB;Z1THfG?|u'
        '+%KS&zfpE#S=HxgsD?iL5CubqfQ^bB9vGE+YBJ2$Ve*4V4d>>(H~;yGdN651AYpr)F3hoVY}-'
        'VxcpY7|MIo4|1gCm}7#w>UK@pGVjGEWZtR5@x3Lni6=66e{yVm9Tar1R#+0$cA_+<nVl|aG?Bw7M`??AGIIu}9TNZKu!osbXp=C2a(9&b'
        '@Ve4QhB?dal3m}Al;L!gXeKEzy3kSKMH+HzTrr6xD-MIY4eKn5M$?uMoGbEZw>=sBoUTZ6&@tAm#9!6wdP@h~U+gBm_XUBO+h;!9o*>(@'
        '1S7@oAlr8HW@?kzS$PTs`mR9mT`(gZ~0ww#kAHIRcR9xxF@d78l$5{++IN;c!}W;(;q(vml>>`m+Zr`Bj5dym9b-'
        'X<B=ySdTJ@C6aICQqe~)ZzIl19}zDjj>Whl`(rgwU4@~jcG_-cOae7Fpi8(8i#{g%$t;&12M){;2Kd4sGS>U(j3#a<^Ur{NV?nrkiZ_jE'
        'f;4Tm*!lcN$lykgfN&ElX`Zpor@QCp^eVA`skq|CmSiAr#=8TX3I{-e8F21OXC-'
        '=6JO&C`^D@if;FmlQxHelP<2;(A}I9$Fw^3=wYJ0of72LG0Sm+b-BIZ@ip;3Oow|p?#4T~xU^U@hE5iJ-vi?;Fm`@T*;M1IID8zvly75i'
        ')RN`qY_uSfve!ixqhgl11kVu_Y>*82+K*kfBOzif&yfQvhv?C&@!XnO5u@~@2pe1xfp@1cAA@-'
        '@>7~+v>gXQH~EwcFXAg75|W>>J6hms;HN*>8s?hVWa&;7Vg+#&*wiA*@sqKGn=X5z#dnpMZ>_?qA@2_j9zUCmbGhp7(H9(!@8(KL<GcU~'
        '+L9!{0iEW3EEccI#IE!HgCL~=5Piln4<CGUEhtBc$ekg*~QfwlCLM=y@HQOT6~_g|WmwpuThfAR!c;=4+_tM{MZ8j1zRRvM4U0U2ATXUC'
        'Z6g3kNlA*d7JTd?LZxtNib@EQ0{c{}$@B$B)1+C;(z7UxQs)LmOy+&Fwch_`6KzT{=FQ{fa0a=T=&cS#gyEtSA?S@<lpms%vM14^|okB8'
        'lS_i^<RCJhq^nhsyO_N^&&?ADw)=j};?>Ctc*(y?Qm!g9*P5!pE-Yc}Hx>ijDUN@MzjXWYFjs#Iumo42?H>AK*|Ocq4fUA_xLn!jWN9d6'
        'uK2i5BZn#rrmV4Y#6<P2NY^D>Ezc2AWtp5pm1U$_sUPw=!YCl$d(tE!rFTzT&Q7$8XXiov`jgKB#Sx>ZC|GHap#Qq3}`SLOgU`eJHgau3'
        'kTUrei-p_-cQIWT_l^skQ28#7&StvepAGQ?khxPO2D`<u7M+fRSI{qWo0-'
        'VVA*OIglqXNvG%d3Mu&<y7h6{pY*Cy?y)UP!APU@2B$S?Yqyf-@JXJY24t?{^p<We==VGeE<HBPVdEmR#p1E>cuJVy|^~c#-'
        'r=9tZS82nDJ<>E|>}a*aV$D*izs@4VkOQ9X|8_&*PhD{k<@oxM7CKU(E;uVHEouYIFjGNRB{3i`XCTL8zkQNQxP5l~rLGmsbWFZnt$1bO'
        '{AaCASM4u(VeK!S;F);vh*)TuQ&{4#p%E!p^n8I#ow7dwx8_9w+i+TYECoAbp93a9yRsL@Csh>YlKPFJ!Pnkb2-'
        '0ubZEAkX0o)z2b|XbR2Bgyi8U#YNA`paGMv+Iu#RL7nNR9Mksk3nkFzMc)FrLW^B*Lxc`t*Im!{IQ_3?Km=Z*g>r~Hr3A0nV>JzCt6K@S'
        'h-;f&ft?QzK;>#C1JYBQ<i1Jr7xat$!@WU-sN$hk+oT_Om(rtvb*xMiY0bc;ou816w8O(J7n!OaC*=xcx2Sw=x^uHef()6I*L^7BwU;lN'
        '!K?$<lVj*r@>0nvbeoRf0PW5ScpoMv`KV}6kXV{*7Bu!D2;0qF$GqJE5R<2#v=3ob?{ReaWEs+k=Cy&yKWw2Hb+NVQgutkzcEo`D!a@9n'
        '0Pqd4Q8iU8poJcn@%^qT4ubD1{zhuxL0x>DggI<Npa{3t`Xpw*he(y(ZjRR;hRB+=;!Cf9!!K~{3f!C)nUW^$lwLX#jR2{FXpNq%j>XW|'
        'Vp|{TO>|m;18B6oJ3_qX-B*NcsGJDbxD%{5v`yXJJShfUz34Ji<e@T;e(t4vD>}{1v-'
        'KC8$cWu56=KQT+uD|d*&nOc55g&+Gg~_1u5b&XRN4Rem8YdKm1<{CXt>jY0neHRU-'
        '9|Pd_E9F*$de|=^q@+VsJ5Fr{k)(OD8M=E+e?0Pm{z%Q`-'
        'juT$sc6wpns;_SbnkwZzElKeu)=L^wMTgqW!}R1!t<auZ&5S=#ptf5ag{!TddL-ZKd<YVbZy_)rGbd13x-mL=*?@Jy!YBSH<ZkoA<Tp*-'
        'z{>m(??SwXRa0${S3;jHtBltK=La{jIN6-'
        'v+P4O`X}q{1q~g!x!BuQ!Ujd&d_af%9dYvvJwWLX(@?Yzu2`BgZY5pclZ0JpI?7?|I_E!_uy>a{r>s$Z|~sapf7buxCNHTN2pG#I(|c)D'
        '*&H6w*aC0R;&=m(c#{-'
        '{5T%6m$mfn(9PLQcQ26on${VPg<rK7;`K9lUd0AlMV6Cw9ap|GO3)jO6Lv^1GL*uVD_>QDoL4tJtYEX2dGRo4$5p(E4GMIjHf&cCZ(d8T'
        ')BpoWEf6Q1$d#o$xA3sb#DLRKRuBW2+7}Xt;xu3mt<*Z-7W&q(0t`OD-'
        '!1i%4mEriP2mFv(ueP$al1%Akbj1NZaBz<@MtQaAbB=5t+vzCH@D5{E4Tew0RDty+s94834dSw<fM5iYUIc?BnPKOL@KGNNV#|_8THh?8'
        'cXxktX-dW9qDn0FUg1UMI+lHYL;W09?V})CZkpCwg=W!0^!|NtqZm5T33S_EO!{Zv}Pi^)GAQVWmjEO%rI!-_AUW@-'
        'q*N3k8fPlngH@1itE&>dZh7T7ZpuZfbDvgpX@>XG4QL8B`{=O=Fxghb40v#w+%q&MC3NB+z%?~$fAVpi9-'
        'S$`u|%Kvom$0$CH$0;W5ap?kL3v=-*DfeG5#hM3lllN?O3&?MQ-'
        'xYA1C;FRM}FsW81JAV0S7?TGFeU(!(9qlBPUhMSU}F01mE;?xirYpzf_v|=hUR@suvdTFpdBi@YWGSbn>9T6JCgRV1B^C^qwU-'
        'Hdl;fid<ZDd<8zSIl+JWg8|l97Cu6m#$(U8j2aTi}uDwTorf?!=yJ?_uOn=_8Uhen!&GeT-Afn%D+57cc@}bI8N`_7}WnBx<f1!HQCfpG'
        'UTCeSey`=XMwnykW={7Y#Y=Krk9Pazt8m(j4mm-N76xk>lg-'
        'U#2`aDyHG>s!I}|JoM9_U>)YlEp1y0>Lw`Wcw3v`mc5V#qpDdL4({In*Vwk50U7*_0WT@rlD+P+^&`AX5MkbcRAH7H_=T&7hbDe3#f(WS'
        'EZmL+6V;O@9Vd0Mu*%e9m;ZveaCixZ^3!e$7B=u>VifzN+@O;!tJ!Km6CioN0pm9IpQFM+z|U?$EGb@%T;dBX2gfA2NQyH=WG{!b+?Tee'
        'LRvmZ&zK}WrIe21ie0^k`5DTM&brY~p{v(;XBVy+5Lkf(8i9{2vf*(RRy#I)E5c5hSa{K#!0x#6SW)(G>Y7ts&Bm*#>&>KkFtm>hz55kw'
        'j3o{Kr^tVmw!&o&-(3Q<V4T|9`kY@HO-o#E+*7;h^?e-giZW-'
        '{>ag1#%Dr^0L$~`G@1EzkaQq?N4lC~Mt_cVx+GgR#b*AROx1V0SQ%4qG_GYTM840hRD52y8v!*dm1(QAx@p*=PR`*ok3$m~^xqsAVPH)z'
        ')-E+IWhB}*it<2`<_Vujd-r6pI=L50g<E4)&MVGT5jS80W(>&wGy|jNc=r)VkE*AU>RrSyh78lqMmlm{lj`>CP?l-'
        'D>SNsA9=?eGe1}Yu57r-qT74Rx|>5i+2F}r9|*@gT?WYFKgev|LuktUjl?OeS3wmrE#f`>)z5)sRD-'
        'I=smb|BCEwkG<7{09I@AL|XX=rssjt&<>)ciRNiJ1p6kUBm|fgZu+DW_7c?QC|{*8s9HaiLp7&fG{7~!aVw4nr<lx'
    )),
    ('reference_validation_v4', '19c9f51f0c996e3a2ed7a8578ae7cc13497c2c021ef59dc7e7fec269e6d7efcf', (
        'c-o~}`*YJs68_zP#T@s8B#@000=p3?bu2MUu0pbvgng9DrJ6|Ncs8=6)ksdfDgO8S=Aow_aCa#P(mc9*db+=Uh-'
        'c59T^BhI3&N9#WemTPB3=`bmU+k`GLQM)qIh|Ic5_aOyet+ZOqUtWxkwXYUtHc0nh+X>Ig5Ca=FVs^cz-'
        'L>xGWgf(?3~=(KVrWIb%zZCR~suqX|#$=4DJSFV6oU4;*Bdv<Meq$P-'
        '?08V|xW0Y#oB!UfBu$kP~eSy|xy0kXQJ;7jI`H0O6baUllwO5u4(V^B+2xnj8(4c>=EiIIO_|8&ilFMs*{h(xrYWLb)W+%h7{EQ>jdM&x'
        '>(6bpv^q+pnv^T(phnIJT0Bw-*n$ntcV77(&X+v*~0B{_#|M1INB6-'
        'y|d<v~lqazX`uW0Omgv>+5J)IwrtG<f#x*<cQ;BnalEWETX4FS9hq$^`4F(!jtBEvQ(;{I+`i6L(NOLw$?tIVqPJR4z!8RYMung8}@>BF'
        '&H`P#de5J>)bCB6c?z++2OR`4oJ(Jpc0f;@Y<@spUXeuEnS*f+ZdjY*?r#8W+4MBc@&!j7GRsNd<4XuGl<g95-'
        'v8DbGd5SQIQXVL5}JCzq#_o3jtw_{G!^*E{1B0=M4Vya#0C=YJCC=F`>W`qSn4>9zl)Xj)k+lo1B`Qb<<b!Qp7^_U5wV*K+ExnW~tpU6P'
        'e|RL|Yg<i0iWx}m28+qR`XYbC`G%~W*)J%wX<N{p|A!60ID!a_E;p-M^8g$~ufxIM>H`(b~-'
        '{e1o75t&e2>M(?9GPtsNg$qc#U9v)YlC)L1CtGkgrb|D`Mmdg26Hzg6-'
        '1eAe=BPc?fAi{x9}iwZ@VD&He=~Y>eDvyv<KyG;!4Vt2Qfm^QzQlg<S6<i$Lw5LFzVO5bKj-'
        '=LLro}W@REecse*?0_S8;Z$m7VxVJh@u^9&)KZMK1*9lIpEy}O<($((v~CEv`Yk6ie}1IOt7j-'
        '`>rl0boGP^3Ye0+{TaK6qI)It2uM%qaj&<5vI;dh<A?MLm!vW?;K4=Etmtg#=5z-'
        'Wi^>8PxZDljBzADeQ#YC+P!F8_=HMuC)O`fS2I;asgxmyrsYfB{7*?7$&s=EV;nB@BC><K#zpjLHeFd_#6`w@i2))&(ux~n$2ELxnSg18'
        'kcO6=V@-wt<QR@)?FQN;&Pb?k35;ZUoEGiCK=en<MBJc(c?S+c&x&~@ZiVRM4G(wF<=Z|*-~Ew>nvG8E0H-'
        'cIHx)M7*H(?%V_T7Fe7MY>bd<h&tOa0+SV&;%|!EtV>F+JGauIjdjLu^1YTQ>xl9Tmd`rpHJs-'
        '@Bi@A)842#WH@QI3b#(aI}&WOgbz4vSvX`+cV`pwxK-'
        'Y$cc42ca_$)*jzBYG=r=j3<{HVF<0{pdAqOg4+8iuI~}!d#ZcFr5!!l6T*d=W+eFuTZQnRILcd-'
        'H@d_*Tjm}L`{9H$wSRF&*|E3EJ@*6cQ8#h2bJ>~dXW!`LrK+OrGiQsooirh)89G)9%{3UxPHxSI#i2mh*EDg%#BAg=dFs&tsV6RG+VQU^'
        'Srxr%iJapOwF`VX<7J6-'
        'O>Z5O}PF)bAegMdB9{^B^)z9`j>oss|Ks&LyoHFc2vL$BUmSknHfxmr$h#iL$Oj&e7lBkZ$(U%B=NLLkbaU2&n_zuiCH4|J12*)gYkGQ|'
        'L!yQtJ<*VJZ6MFhB&t%hp)-!pR7(`gCIGSrvx-'
        'S6{9MyAV&`G1EjZnjwIEt9=r%CyxTsJ9H_DXgWLrcf0bNDcO>Q^qqjWf#oFYjGR;(=dHw0V&Qx^;&?Kx_TStJ367n^@4A-e20-1D!n7-'
        'dIr*qsP+&jhq>I@nOJi?>&Z^4QHrolGyH><(jU@mW@Wl<;vR#F_rK^azOM44j^BikseN+M9PPu3&YQ+shUBnynrfLPGO*Kf>K)$#99q5P'
        'u?ag}ZjXjd0Ax?_d?=zKf;wQQu?TQ4To3(F<e=+6|6X$GAdt*hnJO})+j;ae(-2a^jpjl}I{^%j-'
        'N^>7dtXg3S7_x^#KgtWW6uY0*Qm$hP`5?q}euS_cYU9Qm(6C_HF76BK?Kwe6rh8HgtgP~+VP>V@lvbLz9V_dT;39+q38b^Q#TqiWbZ*x^'
        '#S$pHusU}4vsk&q|bfi8ygmeJlcM!rirP@dvj&kAAf<5Y4yB1o%1GFAkZae?*MUmUhL5$VSoNB@MDH1L`#6A$8V@PNytEj%Mbc^B?Xd(k'
        '}L|U3OIP_#iER0e5MuG8guO*irnYosB?bcM!>1SOJY#^Yur5>C0aKYPkr9GlrO%i$d9F@G~-'
        'N0H<us)(}NkG~lQ>XN{ypJ>%tVhYfgMj5zz6?xHqAkcKfb^v?aI%xa2Hf-sDDc_2y$4ORs?gJI8D`?!>c|#+*{w8cSnLcVR1{?vvuS6oj'
        '2<&@XSUx7uiaF?V)7k`SLOS83j$24zZ>w(TY%$rL~3;o6oTnQM1xuL*vV4VNGDa<TWS%#p_R?3_uc)3j$$BhUnTMkThy{TA_a>Xs_uMU$'
        'Ca%rkp!H|ZBdz2dfb`>Z7oi)#lzptszeWNElW@$3CU}FCDO@|*xY%CBnRC3g^VQf9Ysq6YDN))ocN$9tvzOi%%@aIlinhUS?O+BuQ~)*3'
        '52$iSeb}(GXAev%Uz<NxKkYhcoHH<!*ZcIr}5fwEPULD;jv#ijVvQB?`k|dDe&vX0UI7mZT07e2eWsD{5_N__DTJ*jALnQWvrB;WLN#t^'
        'g*d3OjokO^A<TMvcN1g%U}?<Jc?Kq#r{cfd>J7S0#KnvTu9*KNr|A2J17>kAP;F7M>1+7myAi7L@XZ)D(@vC(K?|^9*R+Y71?M>AGckGb'
        '0Q}WMq~GIG=8r1B$q9jEQ;z4C~=*=*LAK0y`#9KHVhFhk*LESmmMy3ZdOZKhd$xAQP6&>FvTmr>LoEx?uGwU)h1pWYpT-'
        '3YoksJaJ)?jYU52)CV3k`(_oCZ0XiKT9RTvWuU~7|hj#AS+J%|mAE_>NtPvb(DLPKU=7H9iEGqSrUAWEV$;YXtX39gOM>F`sLNYFRR8ph'
        '@enV~u)>F-Xw#QTC28qv-JI<OjChQ%8V7N?V7u#;NAmw}u-'
        '=zsv7d>s)sBIyqRV&=4!&^tGi@LW`Nq_Uqt|W4mr}yHA$?`RIg!#YWGDtrZhN_YkqAk{0T7ZlgX%nhhyK>B!P(5fu9+NmDOiy}yU%K`i)'
        'jn1JLlebkQxe%sby_{njZSt=?5~o->tnWqkUKhl-'
        '66Eu_ueDd7X5EX)@zpUrdB!2Zc_dB`fggabL}R!(UIRxt?~Cg<TktbdkD6idN;`$0Rw}8mRE&j)IT~|jQVq;O<^-'
        '_mQ{{t$&Y@&POi>Ao=r|Y^^I$YVoZJl(}jd;Bh)o)njk)Es$bjIJhk)}C}bNjf;-'
        'h?o%v5d1n<ex;an!ZS`F16g)^=d^Cd4JGhcoqV{0K}Mg@*}o2GGx+B+XqoUh+OMk0YkO{`ZbaQ&^Q!p+|C)RBPpaiCbwe3@~qrdp7hBDa'
        '!+_UU7^k`Yc+&B}Ud>*HgfI@aoLWH~<PIU*;xsq&^>(cyMoL%@^_jfvm@nT&`{W2tT>xm9`QTUV1`&fkBS1izgjt(^RRGr78We;(Xio#F'
        'S{X;ny91!UilU^%UCu=bbh$rbth<=WWNAC9$YdQ~sUt5|P#l`utx{7i9hFgD+KUF0@2EQneli{UznZnJ1A(dwyReK!4q!S2qK*Y2-'
        'uZBu!>J1T8=?X^?9oz0YuCYNUkCcl;o<bAq*q`#!Ab0J5idG84Q3f8kuFF$<wJh`~3zqAWUp)|g124e<`t-'
        '5=Ie!aVuocXlW{f?w)R;VaD{bf57>KC{zYU)2B&|;a2Y=N^YI&w9^ogp{96CD%My%X;njhG6V{sXvQ-=_'
    )),
    ('rascal_reference_v5', '9b20910ed312d77ee6a246172d060863dbca868b42651dac2131909132621253', (
        'c-oa$ZByGg5dQ98p>iJ-GsaL-'
        '+L9y8v?Xxd@zN>j&2gQMN3rEZ#g=@M3>b$0e!H?AJG4i+%;d$=?z5|BU$j|UTf4AY#5N$AidnXyWLA;iFWz5>ocwiqLQ)o6!f&mHZ>2+W'
        'ptvDYS)xnX$|+4^uB}Kb!br|+ya+vSy5Pj9Qpeok<fXA>21{j8WP&GTRuKy#s^FSA<lmTC05-Fhqi6+RfwNR-b<HJ{F%P|qV-hQQ&9xCq'
        'g0SY`9010`hGdc#wgLx?Awm)+$>G5%5w*K%nPnu+Dk0}iBJ|eQ*1S}!oJ3Jt+EVi<A|fx8wx9|s%%X`}M5uP_J^FQg?lq4Ka5@vS_7M)h'
        'Xn)`4S;(#CJe*`a=h9B$H(`cMrk-'
        '~=9dCuhtygd1*Y@~uy1u^cU7UPAitzE}t8iy`xVyEzJAAXf^?LYv9ag=Q_ZOh^YG=5!yFGm4d0xU(64h+jNS67G>w#1W4@fD6HKU{sE1U'
        '6bWUTf_9)Xug2?`-'
        'EXG(Gm9zKT_g4{0FlK9e07*76VS;;3_D?Lc*sW1jvym!bC)OW&Ya_r295J4?w;hft6#bxTF5la0KVh=w0q2@)#Vm_dR2DAYHzfjl(3!+y'
        'a;VC~pONcy=E#y+;ic=bqd;EH#ThBCy;7W+SVs@}RY6y&oV&MDh&e9kB{MwLOFl~7B&CHC*BPMjLS!yGQ=RmWYk;_bfV;|}5Q3@5oZt$Q'
        'Pmni4&52LNEEqp(54EnBtUnvp^F>txeWJ31F!_)U2Ni(|l;^6$l^z!3`B2i$f<rS^t_T+4OJiUCp`;-'
        '@xTg$a%8G$qvR$v)|<%8*gt1vtqi(_5@*Nr1qS3NwX?07N_9dCoCLL3J^W&v&wYBhd~C{O|6N0fEeu_R#qDff=$HOoXoSdT2oDp#7I+ER'
        'mtLd)ha#xzZGER9sm>C@+KGH9<;*l*h!!ZpXuIW|c%b{0}0lL2j(sQ;_$Cwz4iR-zfpuxWLnYgGEBSF1jw?v`cOf;7sHtY~h^%#QDuKy;'
        'bElI5I^oRj?ub;|)1ssSoQDlo)Br(U=4$Sm0Q&TBeq`dwH*yBLQHew&CnH0?oO@dliYAbVJpZ`0A^KGmF}p$lbE6fECU=yh}i9|1juA1}'
        'j)pugYqtY9+Ap977O4KyMHa1XcH>~M1qxbZ^i<mc#4?1#|pPKPUQZ=<XEkJ)fp&|rd`L&WP~-'
        'I38=3<3S)XnjdaE!V@Zs^{TX)${P3To3>5?s@okchAFjavi?wc#+HLtNBlCIN2NP?7+Q9NAQ8@4Rp8teImoXUK%>0pAFaDGy)AJXs(`w7'
        'aTsMg+YjwFrz?Aj2X7R9$HdC-9R36ct?}-)5-'
        'MH<>v9(<kR%{9XWjWVR}K%&rU8!4%rPxV2W|A0vu|JGQ%kbCoh)32A(h=AC3a7ZOF3RG!`G04Mf1z;wt>h)mM^&*cCSU!D_JuZD|Cnxg7'
        'LTFzBUA&<kzwC=9;8H-6szHRlRzquO-yS9_Bc%D{bc-'
        'L|J+$aDd{&Xk}$2Tq}16>uP+7;YSr2sT%E3kiwiBqNqfgQMXm{teo6Z=7tvZiHPRuUrez0!ouI1|*d(5lRSN<_1kS0M>xmZed8Pr!t;|4'
        'h`9iwW?=Ftis9>cIT*qOzlX(8DiW|N2lg5_j?l-#(lN1<IBlMMvQ+rz5TRu9+_)QvY8pQ!2bSAzVUX|=zmO6;N2Z0B)NCG4-k-'
        'lN}+K8s2N)g1&)?jGce!QBmDJF7tEC_=nHMb5veq4P}*wwI;MYv(aqkagRJ+f@z*XJx@-Z?OGr;B8rr54n`L|><JUYR3A4<F7J@2Kh-Fa'
        '1ZurmeUxjp5tiq%58~h&u?m*^+4OVmQ!@6Yx{#;~AU6lo~Xrdm7VpT%cB&2N-F-}S|UMbka99w!Ct}uZW-'
        '!zQKp+VdsF@Xj*E4E_;xI3WX=4MaaY;HOQsKf6N;a`i4@tFSyt<~uO-EdsvdjA5mK8`B'
    )),
    ('identification_v5', '6f574462c51b44cfdfd328d728bf40d1ac1249fbe20e6645ac1aa6a510db9d0a', (
        'c-pO7dvn`1w*TLs0_)DMNQa^&rB1g>nAx;-'
        'THl+*S<B7sj;@9SlaR!iuMm`M+3WAVzXO0TNp><j)p#U6&N(<ZIB&p*hllGEmgOSt$61y$E{ZJNq&$fNmh*xqq7(&-'
        'LXnoyJPLRjWhuLf$|Wm{h{ydflAp6EE!i@Q!)#T$2M1?qzADEo;Bj<S@KS^hOL!S9>4HZ^Q1E%#FT`9FA`Jw4`}@0h?DFFG%iowR0e;Mb'
        'a>dK6V5Nu!#xFL`L7J89^nc$T|N3Dv>1S!YVM3M>$O`GQ4;hm@$z#DH$yQQ?Y!2+J3`|x`5Z62n*&6qux^)lUMrl+E%wS%z!h--qYH-'
        'PTAy}G4ax-Ry__B&{yNnjgKFx^*vEV8DpUF>?D1j(e<?;YyirYMnf~W+4!4lF7AiU@f49O$|J5jngP6U^$0`r2*h`L}`0_+k+9wm}xbBq'
        'QqWdc!NF7Z4I1)E2uWWX!3g&I8*bF4;0L_!R)2Fq(99roe;67vKf7`oyv;dzdc-GetQFGLsx1dGL$lH~RA;{4Ou<m~*zn|Ex5MUDZKtwq'
        'X}Kp?UpOT(y<2w&k&gzmxN;o-'
        'r&$P(uJ^Hqr{`96!1JS$3|VZF2zH3&q`F+VAVtYIW^;uIo}c_1`c&dVj%s{;9m+k*<cSU?@6sP30s5@>b%S!QW<owU%>Rg!No7EAN$QI1'
        'LB0sit(4=aNx-'
        '?%bbq&%+1mVCX)Vt*dRSV9dpkFoltZ>&T5>+dVzUmbSw+xfs@lWK+DM8%wp$6<w$gr8nH?X@~M`1$<Z@9#fMJgcWn$&bRG(<`!A;0L-'
        'be2I$`^p}2uE9gzBF2d|4rHccGPjBTXIRUQC?~VD*{ep*z#FzT2sID|B#)|0)K}Rr9bG_j==88wK2C%O0U>a+(OMPodtU}Dm0F+Biwv2^'
        '8@P|G7;NzQ%H}5Cjy@rgVdD$CVy&xU{G>Cm3=S!|HE6P3FkPQpIP{5%A4*e(%SAhs?011zz05<-'
        'WC>5EFubHV~XnK0Y8YcF0Gq|2v4Xfs_fD<)h(y*8KU-Zuomk^XfNo)ELmHE-EsO#2P4+jSap_o^0NcgoIw8Hr~58Yp2C~rw|?EdTa?O2V'
        'NL{h?%y=kC4T)?7~87*Sb9QWiV@hqIXK^Cu)RL&G0-Wk+!!O7)(;_*tHLRSl`_g=qYjYl@PW7d1tb6C$M2X7eywp~$-'
        'MGB_v*DnSGe=r#Aqs(s^-'
        'r{5Yv+QZ?&;y5Y7EH&EVSfgO^Nm##58X(n+_GJsZmRDmcwf^qsivQcB8ke9bVZLfZOm)tb?#hP)Fu~}EQ=ae*`I>bS8W&<Uts%26b++Nl'
        'vd~xtlhuwwZaY7-h{;7gI+!Q>`vjFf>{gi*kCcDDR-'
        'u$WV0x;RerT;ISyN~;HIlI`mz#`Si)~D`!C+`siEKK7>?1&Uo(UW!3)x#y`i(3P0a`4JUV5ppIlv$bXpR!pW^1HCkEAw>Hw6G|ArZE3@6'
        ')`FddC&<8T_xJQz@!`3M2<x`9QH96@7AucwCZP9@F3^IYWwfJFc<trkUlW@wR?LLuYWdU7@&vQ*P**;UsFh;KD5XQ6C%&#F2S-'
        'rswRnTyt7#`YHWtCQg#n>gixNzPa*L@3#7?_{_&wo-Nxm)ygwDFLz;t`0fvS+++692ucPdgTrF7~`Dg9HjxW7iE(~PtwdZFZb%_p|?$y6'
        'jf2RUk}_NQqe*^MV`Xb=EdNong4!)e?-'
        '=Tpn%zq2EW}gxk&?rqf}*I%@U|gP35noRWY&(!4^E1Pq1}NcTvsD&>cMg8ZQ$PHpw7a$rAY;7J?Trb66)<J$VU+Ou{N|kzOF}z)O`>hgq'
        '<q)DTP~^Ic})7Guy;i`J?YX<uVC{<ih(l{ef~(*3+%8Z-uhP*i0D*BzQs3xT{2s%fH@uNAWp3-Axsl-'
        '8mXvm8F&O@M9?y0aT!j^}a4OV1q*4NGC^sYh0`+}p}%8yy4<j~3*_L8W2AOw!P*=pKiqZU1{%n=+oi_OyMB2~S`4hmG0%b>w*iCSX+NK?'
        '_yABfP#hY(0K;vd7TQQn{+LaARMr%<9Udo$VQcuGijTI(;N(o3&kx0?!h$;0oiZY;G5irwj0+5WkY+?c$=ym?I(A*iR*ZPvm~PRKH^rzK'
        '+Vf>rMaHEan@A=ntomXYnS@67ml%VZ9`mkuZ^lN}i}$kozGhVcB`HVOJ5Cu2!Kn`7fQW_$VHOYirs`+^zK*IYu-iEp(ncY7B4aXx9hDQB'
        'F7_CsjaZ!g_AcOeS#iHOgvj{yXBIJndkjWR+O%(7_$N1&<dlvi$@+J9sC_g!O7tCQ2y02uhxg^rS{JNBs^AQA$A#5Zo`BQV>VEh8@n1KR'
        'NxOQ&Idf@!XRmDp`&`9UYC1xQi()d_-67CnE)Np=lL-V@+EnjCEA#d8s@>>zQF-QE;^!B7$8ch+@%a=X3vc8TDv_C1fD;ObYA5u?EhFcy'
        'XE`2Cjp@UuSk58w25S2uVu+dU0fh>pszcM$;-XDYf_^0fPYbgR4~-'
        'x4ypd)1yk4A%f>R6Q=ftf9omwN_c$@tX>A(kzB94w3O5pizH5cr=X~i@hr)YG_8KXk1BGbnSK0aq7?`gjAzkH117h@`tE3SM5~4s(3S`9'
        'TXTjbGnBq#KW9ZDg0e0pVJ%6f8VCrt@hXJdQQ}~?>`Yx~>o|irqEr-'
        'R#I=mF0=c1Qg*X2=b1&^}S*maKM*Ds^4o{;fK^<FPCQuZK^4U)`h;k?m%&>w<1Ym+?R)iHK+)V>F+D9woFdQc=yw;ixW=`vF*tr|&^@N{'
        'AZU=95Uxm4<(A+U!JjhZZE$#DdM=dD^o!Jhdn4+##eSVcfWn+Zy^xYp$9F7M*RM5tnf-Vti(KSWu4qZ|Lzyg6FS_r0s-'
        'H%W?puwyb<*_w7`h;OQND*#}R8`=9qgX29!`aq)Rzr~36wTHcx*mR-'
        '+tDt!g=QGohSr18`|}jpb9O^D5X(?Y6l<maFIgDLfEQsE5+b46W=#ZPIyrESiWCdv0eEzLJZh&k_t#_VkEwn=?(5%~bGQ9O`@3t$`op=i'
        '$?H`*=OVjLq-'
        '9q{y=jted=w~tRpDE8?zxpJMa6Ei^lxO)!lN&g|I{8cnUc4sUr=9}AmM=`CC@uVO$5%fEcUwP+W7vVsn~GyP}e0?t<h8osYxKWE0(D)>p'
        'YLXXWv}En6#=+{@Stc8?&P-K&r&tMj_OtzCY7FOO+UrdffD7j?&0f12ir0K(%kH``s=HD)~*-'
        'M5k(gGjC5i?3fraS0%f5|3JFR7ior+Fo2V4B&>wxU?g25>0Vi-QO#`rpm4M{hLleg)@(YMDH?gTMAOc)&RpFNv5;@PtSQ~;z?KG3jg%C#'
        'AvgRl$bH8y#Agaht5n<u)DdQlAY3*{P<$||16L@rs{Tw4=~5~u^HmXLE7|U2o0N_|6R*(<xej4?L6+nzsQ+qhw5e5jr`IiTWt+D?Jq20`'
        'm8ZI4Vjm4f|A$T~MIf)p>8UlamFZ0Th+z|m_Z7guR&ma|1pqvb$dR(rwUUT#VjzF5`~tgI@_y*dx}{||t$D<<lwZi>I0g0JJ;gp4Yiv<z'
        'n7hiAlet3_QuA;Uo?FVYoEy7U!MC)~ta=kMwev=T79r73r0~fSp#Z3yd*j+l<(I;gYADpKKmKvkzoRIJL7aVrkd#CVM2%|mprS<5JywF='
        '#?uuPFjgtN5v|N0+aN7p)x3Gw<DZfYS-5QUxk8-nT9~pj2>Q;`NJc!J_q)<-k*wUo{f-'
        'OjeX}%`IQPp_NADj`e10Y2YmgzV;B@v9{j%b;fg|V|&^1}d{qbZk+0@8+ThLVi{*W=gDl-^W6fm<>qw-Ef+j2?!Wh&ZMn|e%1&}DCxZZh'
        '5fGo-'
        '3cNFmmdxWQZ1?#P!L8Brm)N~2Obu#7drdA;=nRU#jgq3AQ>HeRBN;5FB6kL6eKM`C)^#@pl5S`)5T@)5Lpo!cmN@b!_OR4Plo3Dp#D%ev'
        'Pw_OUkG4khI@iS3q_^~SB67IZ{gtG-<F(aDS6xcBp$ckfOw{K@YhKfXIV{l)+3Uzev7|F<`j-'
        '~5ZyfBt@Uar#TIg<Un9{Ki)WWN_&`DiYEp<y%__jr{#)<J2c-9E(!y!lQl+5lAKY@2QA(dU<-mPA8XV@84XWoqu5GAKv||-'
        'c6O&ws33vIpL{Tt&#63ZQCd<XzfShO}z`<tCB^XeL=E*AlLrYrbMhN_{~^dbrLDl0>VrC_21R5#Uncx7j?rBu3R-'
        'y|2}yCQ`Illn|@JdR8g+t(%YZ>A>|{P5Mx%9xjzv{XXlCm+isyxyDK~M%=V_MA1DNODQ?4PA)uL^?wqDzq>yk*i1ci8ui7+q-'
        'xLegR{ThJg$3sv6j4r6^m?#Xw*J1gx!lU5+se!IPHwf|-'
        '|neWZ{7$8o+k`_ZpO1(mI4{{9NJ6qXkT|fbiWT%gCJUfC^hDsBuqC}uX_@~U!TAy(sH=81Kg`RCdEQ}xvS<*)TPxHw%1h^dhd4F)N7{Sw'
        'kW!qH}|GM5b;Y`lSPBNA#ze^oTBYSDg*FF_Ossqr<F_!at-'
        's||Dk{v?7l?{`C)g5W6bW2MjY<ArNbF@&!NM<PqE(jJ+J5cwBo+s8|#%fYM`yh?~ss>l_Fh{j@7pOPONA?U#VPF!`By-vzwOhz<r9G0#7'
        '^O9h*HoJOtT3hmTC67%Zk#d?2UbabKMm(Z@krp~AQhBwtGrhy6yl?t!BCUZ21SS$)vtm(#cJPJbp3HBnnkWZ`s-'
        'giF|ewh$@hs??J1V<AP6XE7aX(awPEmO7RQhbsK_WJ5RR3+2J+@CMv$Ni{iq8x27RHLir&yHw61)S<0$Jt|~^ZlTYIm|jm4Gfg)H_BxL-'
        'bvkX~s2u`0Xxb!`6+wNV0qK}Z(sc$Q6=N6gC#Nr2D#+tv(vYLfRs}_06h@>KfjKK<R^pq>QL{7JRjuhGQ5q%o-'
        '}Zv<M&{X*n{H>_V@f1)ScPPWA4RNB!R*~??Rc<iQ@5^eTiIj9i2WDu&)}Di_k<!qg(P;iI`y$6?Q{c}8>L-'
        '8Pd@dka^Iw5*5hTrb9A&L|8_EB`nK8=b&AuyMSe>R-'
        '&*M}yajwIB&9NrTPrNG+`BARt@>>P5IPAka5_7#&$~OWp$Z4eKiEhnmEBJV<Kc@L@hGVuQ<`H75C#HRO`pyjLHjA}Z*0#SzNp`*{Yh_k-'
        '&2oLgZ<h2AzS2?uZWEh#!m`i3$r@Qg&02FnBx<(*6uFUI7VV+-'
        '8rRG#!gu{oI6)nTcwYLJ&0x_PW3abvR}sbu~ioI>kR6BgSJdO&YxLVSAEXyBSoYGap&C`mR{aXha*}k)!FR;%^lxsZTo;z0&%!wH8I3#;'
        '=>jbbD}##xEOTnbe$E=J5^z87iQ>~PG?sR<-jS2p3;eFgzD5Wm)$}z^-<KQmcmeNMUuos6hmUeemSs@-'
        'NDe_Vlyng>aD`=yi_aFw7k{8&0TlwO(jRpyK=z%J~#7IgV?+Mu+l3B&psc8<<6Vyp{F%vLz+TzFq*OpIjSI^h2_!p@Ob2uxT2pY(VlnEM'
        '&F9|eFyEiLV6ly<0)9{diZRFf#~b`v+M8e;}^T2jW}yP)@ESQIBLPn!tIW;=g7#$?6;FMrs8`u{c%R81)et#h66lte;Du0hW4Bcs;i><N'
        'RZBKVlgk9xS@|FGLfzYNo%`pS$heW&iACDRr!jpCu#K=8<Yk_6op66``TaC&zD&d-'
        'BFm0iNJD`XQc&Ly?LWAhO?G4rM9cL)$=8~v_m(3F{}Ez{VfnZ>K0JrLunh*i+OSta&|kmZlBQ+P&be4zgnZC>bh?(YBUr0_V;gk&CGODE'
        'P1SZ(*W*mhqdTX7bn@*uW4ZRwQAZVXnV6wf?QGadIyuoWmF<*{K^#cLXBp7Cx8S)Et{c{%g3lFsJ$fgAp}AX#iE5CQu70jf16UXFRYf<G'
        'JEOjY@_D&$Jxwu%eKcdryTyw!C<$KjkZ)BkHw!n-'
        'tf`!!j;LM>s8Qk3W1}quLfUftX5qr1>L6q@JMIPB*$Z8F0FPC)jPd$@BHKG2lnp#{G-E?ozkjPoLr~bO<GlNI*X%<g(XpFyqL-s<-'
        'Vp>T=$^%b*h>kkxz&C^SwZ{ZAgdgMFFmV$;@e4D(En4#%W`SDk7#13F_ickKOvWf%@!NyWijRLR1OZ>R8qP`oI<<)BnqWVR%UAyO5eX1e'
        '&j6>i;?F)Qh7EV%_}|sQ+XhN3}p64*mnTW$Ri'
    )),
    ('reference_campaign_v5', '13391ae74cb4dc420389dc8283bbe877d8e52fb0572f70d8ec92417c916ca7be', (
        'c-oy<ZExE)5dQ98!DYZ?W2#&X?LN5m0MB+14|(Zqr`uo{1T9fEJBrdsDzQ8Czwbzil5E-'
        '8)}Ux2^6q%A5AS$&bv37d<xEmU_Zep)(<n(J%26iMCn|^tsgJfaO{kU!A8E-5M+zu4A|gU*MkUcqY*Cnsh-sDz6?8g_o6)Nu-'
        'u{T5sAMst3T+OWs&|N>C&F2T5+Ybkl}1qtn<52ax!zHsSx7VmvMm)xa4JzMJ607qKuQ_{C6l2fu|_;45uDtI?JVaCj_wnkq$=A{Ndt6~!'
        'zoD8Cq|#)S3;OjMqd$huCA^+u}l+$ah&U1Qj8HxvQ!$QKpSH1C`clrNyv#(plC@WX~s!NJLTt&s2yh;_xVYs!u>7sBs(AiCT4Ca1Gm8f{'
        'L7-'
        '^1W8m#IF>X9y}{r3=`E^=rK0$g+9;M_XE@#`yx?S!aanAf1`0R$te3%)wa|hg=4`vuI64TDuu$QrH;!YGOPOwgWyEAGNkTiFPDEp*X#&Q'
        'Aq>v;TVO!CK)3FWb!2a)t>Z$qNd-YSqLOtjp_=!oV)$pPfwXB{cmlT2_(BJk0ems@t5*(6C?hU-'
        '1)|ndg`&*`W`6dX{r2pafH~pL0ZGXelO+O(@Q`vV1^yk*E<*xva^88c5AR}QKWY4D`Zb#!O{%y9r!5^pd+4XEX!Q+R!yV<hAcIv+>O+my'
        'ji=P)PL47DBu-'
        'c@)Nl7%H{`xqZPbUp26&Egv&M2^$J=`}~Y)#^raf;QBKxugc>)VyHq+DIsEShp^tG41O6E!HlyH((N!H(`DwuOT1_G$qp^A**&Mgq|DaW'
        'S3a$#i@>0-'
        'Nyt!xE1kmp9Y<<!n4!9$6Wtsf7A8xnmOwnD2olOO=HQd_AAwO&T5WVs^wOv`jffhUbY;z;rhqEgsEbqw(@_w0tnfU*As0OYjh$j|wGBq('
        '=9XMMIKi2^a)fi=h~{B7ssl_4DG7`!NPK&}DQt1-XmyeD+~^?D{h1HNJU+DnTuQ!66!9NI!-'
        'D+<JriR8aqjA|ZRMcQRKR3#?qFouhnaBEl3W8JzMuII}0t?-8z<&Z9=bpV~SZEg$b{v%WN34*uE50zxdlV-~>?t41#EQXZwbzO-'
        '%evBDvfCHM2`$Jz9^=_Ejt)B-'
        '%5!Vy?N0f7x{IgBzjV&b&nOzmh?YWSsupKwRRKWpr9_r}0LjQ{UQr1Ol^l_hQgoS#L(1XBFke5}6`uk6(s4ZBUG@=wske*rf9GhpESdZy'
        '_8c2)cBLi1ZyREa!S6#khM_b+6He<l?QUnC4tKx~JrJ&NJl-'
        'b1V_{~#9I9B|<awp4fby|rb8y^?nz2~!k(Bs`~6h?eYncg0TKFo41m8j1n(zW03O1)ov|X;D!1dQPCyvdWd!NjTQM06ZYIdWL(X>x#tdL'
        '8o!T*AC4k-'
        'a%E2Z4(yubX1&fo2I(CahF5_(111+MHEU29OhM)oJaX@hR2$=GX*r|=`)qx7j+xCUh4O|TieH&P$Wc$yIO<)bZeX{)JoI;rq9x`q4DD5S'
        '%souySM2=B}+;o47znOE;<^FyB<ruUhuO~cJ|Ovyv~za;Z_lWk1yX>CRdug$JJ(7|KF|kl22v=QYV8#+;EdcPEDf%<VJ4u2sC&yt)MC03'
        'q^3Q^1mjXmmmjKBI?Xd!`v)XxvPOy7(jp)0@TR5ppM`0*;#m5m!T;^<^pdUg4T6N-'
        'MQ#HC*=&REbSNrH;z`d1xy(*(WU;@w;Ugt5v?H6Qcj1%rvBbKaGD&kHjQX-mdh=!hVI`CPfj<%F8QiGXWa|2x^t*mjBT@t>Ki2#M61MOC'
        '@eeSltyR6+RN=rp}=c{?l9Bw$RK?8wEOWdH9LN@xAB|JjDN0sY2=_mL3^H28P3d%H?oyoTO+2BX`|Xfj;qU~u%w<)(chN$%W<h>t8;pMO'
        '^J+3F{lz_e4Hnji^X_v7=tUdQRBp=>)tz5IJ73Y;N*27jy~6v4X=YBak?YQ9p?$-'
        'W03_+#gKTKc2`CVzc|XL&%SsDk3cKewfCi=?!I|HG(_)*Kfbot+uQfUg8u#R?d#Vsdg@VRyMuzhksE2bP<dvsn?}|n{X$INDB5^mD1G)V'
        'TJ9*wv$2GMTAC6zF9iHh%*_$9Gp~`!8GMxk8eDbnP@F?I>RP)}oH^1OP(COaj4uOL$xswTgQxkyM}orj4G9S`mhLc^mi56(3JNG-'
        'a#%hWbXy&0J#3Yswtn=Ru`NC6=)e<O43-j{{4gwAgXS8{e<~voQ@2@s)nj2o;XB;HB~sHg*J?_(-'
        '#Y^M6O=2&v%{fUJ{;=h&!4O1&!63L5h$o1ovWc6nmpI!;G(JV2CYubbkUU}HNBhGFLriqmd1eILbfyQ_)8M%Vg1Dpv69dn9Lg>Tm+xVbH'
        '(1J?)?N8Yw{(%tUKkd_vAeD?WGuEo-XeGH0j}EvTy>boq@{qyBNQo<6uZa26OJZG85QbRxJr9W_nJxpXaUUEgsYmn3|#Zo?<Ml)SPUR^G'
        'zyF7iVWb$%)Jj;5tLWXSX7$Ve56F9qQ4{@ZjbsMvUqC*t(b&c6VwZ;iL~)TRiv~JKJuf^KlE8O^Z'
    )),
    ('rascal_sources_v6', 'fe58bb4ad62eb5fd0d5f3f79b3ab53b8ba3fe2c5d3c6f32f8b37a751c803c7cd', (
        'c-q}sX>J=i7XGhO5Ct?W8QN_p6Qv!DV#~Is<ps8!?!Yh#R23z$Q`;0-'
        'vXvmf++l7qN0_6`N#;G0rDDmF(@Cc>=rNF%#N};#%lF839A~iLK8rlo2$)Z9hNtA33XcY)(`l2GO<9VQ#E*sGENxaQ<2fS%7ZO%4Xpk{M'
        'Xv#>E@{p!W64EG}Qcq?nONmIB$EVz*lE;xCaY}FsHcaA75=tT(GQT3{JoOt1P34m83La*GT2H1tV7pBM`~)LjK!q^8qF9oUQ2{(c7D-'
        'VjVeGS@0#frysy5e=8HmSy_476CAzmC#cm!haELaL+!8giNFQrqt1C!B|&oaH6ut!i(G%JqdRHkVh64#w(sFUjw9wu=L(nlbMR!lDePBf'
        'R&tXU1!2^Bu~<Z308JY?1~Rx#g0D(8BYTqZo4<!k!UvQinnb;p<PPp9ONw43_}dk3$&doL@)3miRHgYE9YJ`8l#NEe38<nC^_axyN~?X~'
        'y+0t3Z5Gl}ae-~{gN^$13txs-'
        '|MwOWGD!s)CDCTt~X>}PO_IMzs&uQb3{*o>xAr*ienzT3Y(Ivc(ltkalrInO4F2CQnG4M(j>5Kmem1!qcI`B}8I|Cx{GV;DG<zummO8J@'
        'exmq+VFe-#sOnTBLEY#IhUHlr;b`RuVdmq_AdG(0^UpAP!h&&V~NXRswb36>4Y(pV(aW37{b&*t)!(X^puaw?U-'
        'u8y_}hxL&~hvPJ3wTc=dc}Vpt1pdIY!Oua(9uYECLn6*nm|p!>fz(rnj6bu8%VJW|T!oP}Zo6zToC!L!NPBd_sG(M=_-'
        'tyJyUYk#t&zqX714LO-1k~=@ct18xgi<oJvw|}lfP#)l3a4|^pr`;BQxKKK$Znl@(VJB5Tu#UC7tkq%ViUz3F*R~ERD#W`hkBr-'
        '>UNfE)Q0ZoH{XHSjR-vY}bjyanva#m?WEk;B)5V%v$amoRa?a2~Nyd4E9gCSL8DszREX4atL_@+k6PhNMK0URg$5iG0AvP$~I1|zC~^qa'
        'uY2He$ee2$-CcI^geLXP6OkZA3zB-'
        'x6NTk7&@>p<#e0m0gGmGZj1f$p!xFP53u)zv9ji<E$o3sS=xxFjaf<;;3SaX!_9(DgC!dIfnhYdDUD{#min&KL0fnBn(cdF7EHcT_&8x0'
        'QIL5xC&Hoy3t$u{&&&4JBIOH}NLnpSJPCvUL^4?Fc{rg`z;0GDWsrtK-08b4U>;i0P2vZZmUVs6Rx#04pQ!K8->7cRgeWFtmO;fd=XD-S'
        'a49{$l1s_nBo~)sBIju)un_v{C(Wb82QUyPS%85A9+`YOV%CZVoMO9q@DfCrN)<RZLNEX!PIVl3b4WNoF$INSq2QrW6<1hz!sIX%jsYcQ'
        '7L(QhX~1!=616_Wk~Og$_)7pRV5&TGZdc5tuqS%GYId|*|2wT{^c881g^666klT1pdn&)_VEe=L`Q>~B%CNvEN}dP#8YPb~^w9ar&@Dut'
        'L`5e|%>Y^8`~Wq+e4wS+-K;%-'
        'eB7vX^A)B?5oEL7KiWb2LS%rEHNb_m0^^+4Z3OI>)iO)AD70K|R_Nf*s?f)eTTZC+XHn?hpfY=nE4W5w*fZP#a;qaWG^os=r%x+$IC*{V'
        'ZO!w=TFgvt2upE#H*XKD1*^R5hFx3+7)Gr#t%9|3nh6#F)^v42S5y^t*htruZvgwk1svdhm?<G(ZQj*vTTCC#;=Rp_mBc|zeYHiA*0NVi'
        '<ZGk_BWvo^w$f2tb2@upLuXaDLoA@Wdb@dHbsvD>llon3Hs0Y1=q)pWM~a>#vLp$xVV$gZ*vcX81GWm4>&0veeFMiP`P(TL)!fx3-'
        '}SO|!G4E`rqfdt7!oM%8A?|MB#we*U84(Z%9)ozGZ8TIqH%nAvg+Q~fR*O-*~As#7KnO~HcQiZ2K-zZyS9MU-'
        '8xZ@_F;C|QFvX)$YHk$S<w!M!*=skyXclMI896f6c^A^-WfVMiSqV)wUkeF84iLf_WDgOrX1QOR$bIoHCvE3q^-'
        'dG0#b#dT<fiDmr^bmd56e<0Q#g=KxvqFOh&?vOclThmVYFxXeP1X*qAvrMJNPOpQM}3o!KYnD$jLrOg|UU#(<1G=h);*y%j7-'
        'V`!?Y;oXil$_~KQgPOvZAUVwEwsLN<ZA6M8k~ro<%MC{gW9OI0gAq9&-'
        'kkTxM`sS&kxoQaZ*E!C8=cx42&xWY0hr49c{n~BUXacY7BYe&;q(le{#<`XSz-'
        'X{w`gl0J?i|z*4dscY5wnj{dV1LBCRJJzn2|OG*Cj<CRIqUPz}S{SJ`A`lBYEQ4YOaX=XS~CCpU-#iT|lNE^o&-!{Y(Dx_vtu-kc4N$;s'
        '&b@B%7d|KfDu<RP|&;g5%g3w&8yp>qDtec56K7ft?mnU6kxFY_5$|G&(agCTFuybrFqs8T2%a;?BjO%+_DfvZ~+#W<K8Vk2VGWG;Rwxo+'
        'N8!&~CbQ{=}((ta-Q8r-N`5`DaR=3Ps-'
        'P}~lmPq&{O4k!|Q!5K!XUP+EsP7w|01ndczBf%T^Nw5S^I91hht?mgh?Mhx<XiHAjb2^>>{Nta1Wk5S~;@T@Wg2tLvu}#_58autI^J%lu'
        'db7_+d~kj>es5^!sG8<5y9BQ{F!8jTI>)iW(CLg3sonro4=7#}KlU>0SqlJ=LQ5{xwSNIt>SUU+pf#ZqdS5^h30)$%Qw0&fC|nLA3WgoB'
        'WWloeM4BApIJXzqgPY6IyTP&ZxxA=2yNkFCeDWs|`|7|#WWZQHVYqf;K=Q~PRcVmSDU^>R_t^R#uD*k7$Ogz71nAH{OW|nLMXW+g0&#@L'
        'rFq^dKgw)BWoNvpWkR3B8(*Ot9(+Rq^}y7TryqHAM_%C0?I-*bV@ZB-th*=MQWX2DMAMXXHbYK=KuWnjZd~=R$M1EIbvZb1n0FrLH;M5Y'
        '9zleVS1s`7DfEi^_W3s*Zk~b6d)L>rVUo)L?Y1$TdZeGpd7N^oPsZfHw-kCP!2C8mMpaz&X!d~V-'
        '|4oyFB+XrqrGph$}gdyzkuWjrKWT_X&&0%tYXD_>?*yT!4mZfXPvMpu8&}rJp!x;?;+6SfuP`92#botv1KZI?~|OjL3*cmr(edZk2M8)A'
        'Ia)9!Fr*vJ0$fshec<x?=B83T(6Nn7wBAHM*X%lS(Gvn2aukMVgocb7a@D&ETgFpwIP7zcNncAn}BdK)7{I|+@69;50HlRlZd%Cg|Ub|$'
        'PFP_cqnER{)h$#QdU;Q7_&0?vjmEx0?t&f%H8U|Lj^3D2~!ffu6Z=ThgLMd<mkxjmF78y5}M!1$^!Pqi^=__$8D+1fe&d!#)@^}!JzZ=X'
        '0eX4k-z)m6?;H|0p9lE3o~E<Z>~LmikF}*>tecmYlucz`Q%o{{XAXG6eiO*-'
        '6~G}7}Bh6gN*|){UD=ng~B9I;ij4C=jXh|g+ei33T5Tu!P>DsewC<hE^n`o2ITni==OYYG479tmlx!1|7I{6UJS_P#R!t3N_tmk?{9`j{'
        'Si3<pf?(V-(U|txEc2i7+Y(S)ia%&CAu!U`uWs-41+5Cb0<s5M31PCt@%*BS?lLuch$~AfaVX_-'
        '~oc=lvg(ZKwj_0bal9OO(DjYAQgfmad@YDZ9D;mpXv%a)F+T_G4SrHc6V34@~Q*v^;W93%_GcT)#{A%8#YZRZ%S+T1(t+UK2-'
        'YNDXH#P6H4|bP4MNYYIhM@2j>dS<C9;`0XGAYdO72pM@*-OT5fgWV*?%`!T;d&I_>su6-'
        'c#k4>XLQOStrtgJxHv0_ek!dj7VzEM;3Ic$uZ6Afaz}9uEGWBY2muGnsic<HA&IKxy6_pU~6|&yx%L9g(Lzy>Y#sJ4NuUvF%5St0`;R$q'
        '5sPjj+SAUPy#aSdb-ggLV0oS5E3drlcN#iO-e)0>pzwYy'
    )),
 ]
for _module_name, _source_sha256, _compressed_chunks in _BUNDLED_V6_MODULES:
    if _module_name in _bundle_sys.modules:
        continue
    _source_bytes = _bundle_zlib.decompress(
        _bundle_b64.b85decode("".join(_compressed_chunks))
    )
    if _bundle_hashlib.sha256(_source_bytes).hexdigest() != _source_sha256:
        raise RuntimeError(f"Bundled module integrity check failed: {_module_name}")
    _virtual_module = _bundle_types.ModuleType(_module_name)
    _virtual_module.__file__ = f"<bundled_v6>/{_module_name}.py"
    _virtual_module.__package__ = ""
    _bundle_sys.modules[_module_name] = _virtual_module
    try:
        exec(compile(_source_bytes, _virtual_module.__file__, "exec"),
             _virtual_module.__dict__)
    except BaseException:
        del _bundle_sys.modules[_module_name]
        raise
# ================== END EMBEDDED SIMULATION ENGINES ==========================

from engine import (FAULTS, MODES, SimConfig, heading360, theoretical_drift_m_per_km, wrap_deg,
                    simulate as simulate_legacy, monte_carlo as monte_carlo_legacy,
                    v_and_v as vv_legacy)
from advanced_engine import (AerospaceConfig, simulate as simulate_aircraft,
                             monte_carlo as monte_carlo_aircraft, v_and_v as vv_aircraft)
from validation_io import compare_reference
from sixdof_engine_v4 import (SixDOFConfig, simulate as simulate_6dof,
                               monte_carlo as monte_carlo_6dof, v_and_v as vv_6dof)
from reference_validation_v4 import (compare_truth_v4, check_manifest, load_truth_csv,
                                     reference_template_manifest)
from rascal_reference_v5 import parse_jsbsim_aircraft_xml
from reference_campaign_v5 import template_campaign, check_campaign
from identification_v5 import identify_and_validate, replay_holdout_trials
from rascal_sources_v6 import (manufacturer_si, source_evidence,
    compare_surrogate_to_manufacturer, check_source_conflicts, dossier,
    audit_uploaded_jsbsim_xml, SIG_URL, PX4_ARCHIVE_URL, JSBSIM_DOC_URL)


def simulate(c):
    return (simulate_6dof(c) if isinstance(c, SixDOFConfig) else
            simulate_aircraft(c) if isinstance(c, AerospaceConfig) else simulate_legacy(c))


def monte_carlo(c, runs):
    return (monte_carlo_6dof(c,runs) if isinstance(c,SixDOFConfig) else
            monte_carlo_aircraft(c,runs) if isinstance(c,AerospaceConfig) else
            monte_carlo_legacy(c,runs))


def v_and_v(df,metrics):
    return (vv_6dof(df,metrics) if 'true_pitch_deg' in df.columns else
            vv_aircraft(df,metrics) if 'true_roll_deg' in df.columns
            else vv_legacy(df,metrics))

st.set_page_config(page_title="UAV Compass Variance Simulator", page_icon="🧭", layout="wide")

# HUD palette follows the Directed Energy Engagement Digital Twin visual language.
# Neon cyan is reserved for compass, heading-reference and azimuth instrumentation.
GREEN = "#7CFF22"
LIME = "#A5FF4D"
ORANGE = "#FF7A18"
ORANGE_LIGHT = "#FF9D2E"
RED = "#FF4D3D"
NEON_BLUE = "#00E5FF"
NEON_BLUE_BRIGHT = "#73F6FF"
NEON_BLUE_DEEP = "#007FAA"
BLUE = NEON_BLUE
TEXT = "#E9FFE1"
MUTED = "#9DBA96"
BG = "#020402"
PANEL = "#061009"
GRID = "#214A19"
CYAN_GRID = "#124552"

st.markdown(f"""
<style>
:root {{
    --hud-green: {GREEN};
    --hud-lime: {LIME};
    --hud-orange: {ORANGE};
    --hud-orange-light: {ORANGE_LIGHT};
    --hud-red: {RED};
    --hud-blue: {NEON_BLUE};
    --hud-blue-bright: {NEON_BLUE_BRIGHT};
    --hud-text: {TEXT};
    --hud-muted: {MUTED};
}}
html, body, [class*="css"] {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}}
.stApp {{
    background:
        radial-gradient(circle at 82% 8%, rgba(0,229,255,0.055), transparent 25%),
        radial-gradient(circle at 72% 0%, rgba(255,122,24,0.055), transparent 32%),
        radial-gradient(circle at 13% 3%, rgba(124,255,34,0.055), transparent 30%),
        linear-gradient(180deg, #010301 0%, #020602 60%, #010301 100%);
    color: var(--hud-text);
}}
.block-container {{
    max-width: 1600px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
}}
h1, h3 {{
    color: var(--hud-lime) !important;
    letter-spacing: .04em;
    text-transform: uppercase;
    text-shadow: 0 0 9px rgba(124,255,34,.15);
}}
h2 {{
    color: var(--hud-orange-light) !important;
    letter-spacing: .035em;
    text-transform: uppercase;
}}
p, label, .stCaption {{color: var(--hud-text);}}
small, [data-testid="stCaptionContainer"] {{color: var(--hud-muted);}}
hr {{border-color: rgba(124,255,34,.2) !important;}}

/* Full sidebar flight-console styling */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #020602 0%, #071208 65%, #020702 100%);
    border-right: 1px solid rgba(124,255,34,.32);
}}
[data-testid="stSidebar"] [data-testid="stForm"] {{
    background: linear-gradient(180deg, rgba(7,22,10,.55), rgba(2,7,3,.85));
    border: 1px solid rgba(124,255,34,.20);
    border-radius: 12px;
    padding: 13px 12px;
}}
[data-testid="stSidebar"] hr {{margin-block: .75rem;}}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: #09170B !important;
    border-color: rgba(124,255,34,.28) !important;
}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: #B4D2AB;
    font-size: .82rem;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{
    box-shadow: 0 0 9px rgba(124,255,34,.3);
}}
.hud-sidebar-brand {{
    padding: 14px 13px 12px;
    margin: 4px 0 12px;
    border: 1px solid rgba(124,255,34,.43);
    border-left: 3px solid var(--hud-green);
    border-radius: 9px;
    background: linear-gradient(110deg, rgba(124,255,34,.11), rgba(0,229,255,.035));
    box-shadow: inset 0 0 18px rgba(124,255,34,.035);
}}
.hud-sidebar-brand .eyebrow {{
    color: var(--hud-orange-light);
    font-size: .65rem;
    letter-spacing: .2em;
    font-weight: 800;
}}
.hud-sidebar-brand .name {{
    color: var(--hud-lime);
    font-size: 1.0rem;
    font-weight: 800;
    letter-spacing: .035em;
    margin-top: 7px;
}}
.hud-sidebar-brand .sub {{
    color: #AAC5A3;
    font-size: .71rem;
    margin-top: 4px;
}}
.hud-side-section {{
    font-size: .74rem;
    color: var(--hud-orange-light);
    font-weight: 800;
    letter-spacing: .095em;
    margin: 12px 0 7px;
    border-bottom: 1px solid rgba(255,122,24,.19);
    padding-bottom: 6px;
}}
.hud-side-note {{
    border: 1px solid rgba(0,229,255,.27);
    border-left: 3px solid var(--hud-blue);
    background: rgba(0,229,255,.04);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: .72rem;
    color: #BCDEE0;
    line-height: 1.55;
}}

/* Headers, instrumentation panels and status */
.hud-hero {{
    border: 1px solid rgba(124,255,34,.38);
    border-radius: 14px;
    padding: 24px 26px 22px;
    margin: 3px 0 16px;
    background:
      radial-gradient(ellipse at 91% 30%, rgba(0,229,255,.12), transparent 36%),
      linear-gradient(108deg, rgba(5,20,8,.98), rgba(2,9,5,.96) 65%, rgba(5,19,18,.98));
    box-shadow: 0 0 26px rgba(124,255,34,.055), inset 0 1px rgba(124,255,34,.10);
    position: relative;
    overflow: hidden;
}}
.hud-hero:before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--hud-green), var(--hud-blue) 63%, var(--hud-orange));
}}
.hud-hero .kicker {{
    color: var(--hud-orange-light);
    font-size: .7rem;
    letter-spacing: .2em;
    font-weight: 800;
    margin-bottom: 9px;
}}
.hud-hero h1 {{
    margin: 0 0 10px !important;
    font-size: clamp(1.36rem, 2.4vw, 2.25rem) !important;
    line-height: 1.18;
}}
.hud-hero h1 span {{color: var(--hud-blue); text-shadow: 0 0 15px rgba(0,229,255,.24);}}
.hud-hero .subtitle {{
    color: #B5D1AD;
    font-size: .79rem;
    letter-spacing: .055em;
    line-height: 1.65;
}}
.hud-hero .statusline {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 15px;
}}
.hud-chip {{
    display: inline-block;
    color: var(--hud-lime);
    background: rgba(124,255,34,.055);
    border: 1px solid rgba(124,255,34,.24);
    border-radius: 5px;
    font-size: .68rem;
    padding: 5px 9px;
    letter-spacing: .055em;
}}
.hud-chip.cyan {{
    color: var(--hud-blue-bright);
    background: rgba(0,229,255,.065);
    border-color: rgba(0,229,255,.28);
}}
.hud-chip.amber {{
    color: #FFBE75;
    border-color: rgba(255,122,24,.32);
    background: rgba(255,122,24,.075);
}}
.hud-chip.fault {{
    color: #FF8D82;
    border-color: rgba(255,77,61,.35);
    background: rgba(255,77,61,.08);
}}
.hud-section {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    border-left: 3px solid var(--hud-green);
    background: linear-gradient(90deg, rgba(124,255,34,.07), transparent 85%);
    padding: 10px 12px;
    margin: 14px 0 13px;
    border-radius: 0 7px 7px 0;
}}
.hud-section.cyan {{
    border-left-color: var(--hud-blue);
    background: linear-gradient(90deg, rgba(0,229,255,.075), transparent 85%);
}}
.hud-section .kicker {{
    color: var(--hud-orange-light);
    font-size: .64rem;
    letter-spacing: .15em;
    font-weight: 700;
}}
.hud-section.cyan .kicker {{color: var(--hud-blue-bright);}}
.hud-section .title {{
    color: var(--hud-lime);
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: .05em;
}}
.hud-section.cyan .title {{color: var(--hud-blue); text-shadow: 0 0 10px rgba(0,229,255,.22);}}
.hud-section .desc {{font-size: .75rem; color: #9AB493; line-height: 1.5;}}

/* Compact engineering readout cards. Values remain data-driven, not screenshots. */
.hud-kpis {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
    margin: 3px 0 13px;
}}
.hud-kpis.four {{grid-template-columns: repeat(4, minmax(0,1fr));}}
.hud-kpi {{
    border: 1px solid rgba(124,255,34,.28);
    border-radius: 10px;
    background: linear-gradient(165deg, rgba(8,22,9,.98), rgba(3,9,4,.98));
    padding: 12px 13px;
    min-height: 87px;
    overflow-wrap: anywhere;
}}
.hud-kpi .label {{
    color: #ACC5A5;
    font-size: .71rem;
    letter-spacing: .045em;
    margin-bottom: 8px;
}}
.hud-kpi .value {{
    color: var(--hud-orange-light);
    font-size: clamp(1rem, 1.7vw, 1.5rem);
    font-weight: 800;
    text-shadow: 0 0 8px rgba(255,122,24,.14);
}}
.hud-kpi.blue {{border-color: rgba(0,229,255,.32); background: linear-gradient(165deg, #05191C, #030A0C);}}
.hud-kpi.blue .label {{color: #A6D9DC;}}
.hud-kpi.blue .value {{color: var(--hud-blue-bright); text-shadow: 0 0 10px rgba(0,229,255,.19);}}
.hud-kpi.green .value {{color: var(--hud-lime); text-shadow: 0 0 9px rgba(124,255,34,.15);}}
.hud-kpi.fault {{border-color: rgba(255,77,61,.32);}}
.hud-kpi.fault .value {{color: #FF8D82;}}
@media (max-width: 1000px) {{
    .hud-kpis {{grid-template-columns: repeat(3, minmax(0, 1fr));}}
    .hud-kpis.four {{grid-template-columns: repeat(2, minmax(0,1fr));}}
}}
@media (max-width: 600px) {{
    .hud-kpis {{grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px;}}
    .hud-kpi {{min-height: 78px; padding: 10px;}}
    .hud-kpi .label {{font-size: .66rem;}}
    .hud-kpi .value {{font-size: 1.08rem;}}
}}

/* Metric panels and plot surfaces */
div[data-testid="stMetric"] {{
    background: linear-gradient(175deg, rgba(7,19,8,.97), rgba(2,8,3,.99));
    border: 1px solid rgba(124,255,34,.31);
    border-radius: 10px;
    padding: 13px 15px;
    min-height: 91px;
    box-shadow: 0 0 15px rgba(124,255,34,.045), inset 0 1px 0 rgba(124,255,34,.045);
}}
div[data-testid="stMetricLabel"] {{color: #A5BE9F;}}
div[data-testid="stMetricValue"] {{
    color: var(--hud-orange-light);
    text-shadow: 0 0 9px rgba(255,122,24,.16);
}}
[data-testid="stPlotlyChart"] {{
    border: 1px solid rgba(124,255,34,.19);
    border-radius: 10px;
    background: #020602;
    overflow: hidden;
}}
[data-testid="stExpander"] {{
    border: 1px solid rgba(124,255,34,.25) !important;
    border-radius: 10px !important;
    background: rgba(4,13,5,.55);
}}
[data-testid="stDataFrame"] {{border: 1px solid rgba(124,255,34,.2); border-radius: 8px;}}
button[data-baseweb="tab"] {{
    color: #BDD4B6;
    background: rgba(6,16,7,.88);
    border: 1px solid rgba(124,255,34,.20);
    border-radius: 8px 8px 0 0;
    font-size: .83rem;
    font-weight: 700;
    letter-spacing: .025em;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--hud-orange-light);
    border-color: rgba(255,122,24,.45);
    box-shadow: inset 0 -3px 0 var(--hud-orange);
}}
button[data-baseweb="tab"][aria-selected="true"]:nth-child(2) {{
    color: var(--hud-blue);
    border-color: rgba(0,229,255,.45);
    box-shadow: inset 0 -3px 0 var(--hud-blue);
}}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {{
    border-radius: 8px;
    border: 1px solid rgba(255,157,46,.52);
    background: linear-gradient(180deg, #221507, #150E05);
    color: #FFB56B;
    font-weight: 800;
    box-shadow: 0 0 11px rgba(255,122,24,.08);
    transition: border-color .16s, box-shadow .16s;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
    border-color: var(--hud-orange-light);
    color: #FFE2B2;
    box-shadow: 0 0 16px rgba(255,122,24,.18);
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{
    color: #071007;
    background: linear-gradient(180deg, #FFAA3C, #FF6A00);
    border-color: #FFBD65;
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {{
    color: #071007;
    background: linear-gradient(180deg, #FFBD63, #FF7920);
}}

/* Compass readings use neon cyan, not the orange general-purpose metric color. */
.compass-readouts {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 10px;
    margin: 10px 0 15px;
}}
.compass-tile {{
    border: 1px solid rgba(0,229,255,.38);
    border-radius: 10px;
    background: linear-gradient(165deg, rgba(5,23,27,.96), rgba(2,10,12,.99));
    box-shadow: 0 0 16px rgba(0,229,255,.07), inset 0 1px rgba(0,229,255,.07);
    padding: 13px 14px;
    min-height: 90px;
}}
.compass-tile .readout-label {{
    color: #A2DBE0;
    font-size: .69rem;
    letter-spacing: .06em;
    margin-bottom: 7px;
}}
.compass-tile .readout-value {{
    color: var(--hud-blue-bright);
    font-size: clamp(1.12rem, 1.9vw, 1.62rem);
    font-weight: 800;
    line-height: 1.2;
    text-shadow: 0 0 10px rgba(0,229,255,.2);
}}
.compass-tile.disturbed {{border-color: rgba(255,122,24,.46);}}
.compass-tile.disturbed .readout-value {{color: var(--hud-orange-light); text-shadow: 0 0 9px rgba(255,122,24,.2);}}
.compass-tile.fused {{border-color: rgba(124,255,34,.34);}}
.compass-tile.fused .readout-value {{color: var(--hud-lime); text-shadow: 0 0 9px rgba(124,255,34,.2);}}
.compass-note {{
    color: #B3E8E9;
    font-size: .78rem;
    line-height: 1.65;
    padding: 11px 14px;
    border: 1px solid rgba(0,229,255,.25);
    background: rgba(0,229,255,.04);
    border-radius: 8px;
    margin: 8px 0 12px;
}}
.compass-note strong {{color: var(--hud-blue-bright);}}
.hud-foot {{
    border-top: 1px solid rgba(124,255,34,.25);
    color: #89AC81;
    padding: 12px 4px;
    font-size: .7rem;
    line-height: 1.5;
    letter-spacing: .05em;
}}
@media (max-width: 760px) {{
    .hud-hero {{padding: 18px 15px;}}
    .hud-hero h1 {{font-size: 1.25rem !important;}}
    .compass-readouts {{grid-template-columns: repeat(2, minmax(0,1fr)); gap: 7px;}}
    .compass-tile {{padding: 11px 9px; min-height: 78px;}}
    .compass-tile .readout-label {{font-size: .63rem;}}
    .compass-tile .readout-value {{font-size: 1.1rem;}}
}}
</style>
""", unsafe_allow_html=True)


def panel_header(kicker: str, title: str, subtitle: str = "", *, compass: bool = False) -> None:
    """Render a noninteractive console section header."""
    color_class = " cyan" if compass else ""
    st.markdown(
        f'<div class="hud-section{color_class}">'
        f'<div class="kicker">{kicker}</div>'
        f'<div class="title">{title}</div>'
        f'<div class="desc">{subtitle}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def hud_kpis(items: list[tuple[str, str, str]], *, four: bool = False) -> None:
    """HUD readouts with per-channel semantic color and responsive mobile layout."""
    tiles = "".join(
        f'<div class="hud-kpi {color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value, color in items
    )
    cls = " four" if four else ""
    st.markdown(f'<div class="hud-kpis{cls}">{tiles}</div>', unsafe_allow_html=True)


def compass_readouts(row: pd.Series) -> None:
    tiles = [
        ("TRUE-NORTH HEADING", row.true_heading_deg, ""),
        ("RAW MAGNETIC HEADING", row.magnetic_heading_deg, "disturbed"),
        ("DECLINATION-CORRECTED MAG", row.mag_true_heading_deg, "disturbed"),
        ("GYRO + MAG FUSED", row.fused_heading_deg, "fused"),
    ]
    cards = "".join(
        f'<div class="compass-tile {cls}"><div class="readout-label">{label}</div>'
        f'<div class="readout-value">{float(value):.2f}°</div></div>'
        for label, value, cls in tiles
    )
    st.markdown(f'<div class="compass-readouts">{cards}</div>', unsafe_allow_html=True)



def theme(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Consolas, monospace"),
        height=height, margin=dict(l=15, r=15, t=45, b=25),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="closest",
        hoverlabel=dict(bgcolor="#071A12", font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showline=True,
                     linecolor="#2D5128", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showline=True,
                     linecolor="#2D5128", tickfont=dict(color=MUTED))
    return fig


def compass_figure(row: pd.Series, declination: float):
    """Neon-cyan compass instrument with distinct truth/magnetic/fused needles."""
    fig, ax = plt.subplots(figsize=(5.25, 5.25), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-1.29, 1.29)
    ax.set_ylim(-1.31, 1.31)

    # Concentric glow strokes simulate a backlit instrument without GPU shaders.
    for radius, width, alpha in [(1.035, 15, .055), (1.026, 8, .11), (1.016, 4, .20), (1.0, 2.0, .95)]:
        ax.add_patch(plt.Circle((0, 0), radius, fill=False,
                                ec=NEON_BLUE, lw=width, alpha=alpha, zorder=2))
    ax.add_patch(plt.Circle((0, 0), .79, fill=False,
                            ec=NEON_BLUE_DEEP, lw=1.0, alpha=.45, zorder=2))
    ax.add_patch(plt.Circle((0, 0), .59, fill=False,
                            ec=NEON_BLUE_DEEP, lw=.7, alpha=.20, zorder=1))
    ax.plot([-.90, .90], [0, 0], color=CYAN_GRID, alpha=.23, lw=.8, zorder=1)
    ax.plot([0, 0], [-.90, .90], color=CYAN_GRID, alpha=.23, lw=.8, zorder=1)
    for deg in range(0, 360, 5):
        angle = math.radians(90 - deg)
        major = (deg % 30 == 0)
        mid = (deg % 15 == 0)
        inner = .89 if major else (.935 if mid else .955)
        ax.plot([inner * math.cos(angle), .985 * math.cos(angle)],
                [inner * math.sin(angle), .985 * math.sin(angle)],
                color=NEON_BLUE if major else NEON_BLUE_DEEP,
                lw=1.45 if major else .75,
                alpha=.9 if major else .60, zorder=3)
    for mark, x, y in [("N", 0, 1.16), ("E", 1.16, 0), ("S", 0, -1.16), ("W", -1.16, 0)]:
        ax.text(x, y, mark, color=NEON_BLUE_BRIGHT if mark == "N" else NEON_BLUE,
                ha="center", va="center", fontsize=15 if mark == "N" else 12,
                fontweight="bold", zorder=6)

    needles = [
        (row.true_heading_deg, NEON_BLUE, "True heading", .80, 3.0, 1.0),
        (row.mag_true_heading_deg, ORANGE, "Magnetic + correction", .71, 2.3, .9),
        (row.fused_heading_deg, LIME, "Fused estimate", .87, 1.8, .87),
    ]
    for hdg, color, name, length, thickness, opacity in needles:
        angle = math.radians(90 - float(hdg))
        ax.arrow(0, 0, length * math.cos(angle), length * math.sin(angle),
                 head_width=.064, head_length=.092, linewidth=thickness,
                 color=color, length_includes_head=True,
                 label=name, alpha=opacity, zorder=4)
    ax.scatter(0, 0, s=75, c=NEON_BLUE_BRIGHT, edgecolors=BG, linewidths=.9, zorder=8)
    ax.set_title(f"COMPASS // TRUE-NORTH REFERENCE  |  DECLINATION {declination:+.1f}°",
                 color=NEON_BLUE_BRIGHT, fontsize=9.5, pad=18, fontweight="bold")
    legend = ax.legend(loc="lower center", bbox_to_anchor=(.5, -.135), ncol=3,
                       facecolor="#041315", edgecolor=NEON_BLUE_DEEP,
                       labelcolor=TEXT, fontsize=7.8, framealpha=.93)
    legend.get_frame().set_linewidth(1)
    fig.tight_layout(pad=1.0)
    return fig




# ============================================================================
# LINKED INTERACTIVE 3-D COMPASS + RASCAL REFERENCE AIRFRAME
# Geometry-only Rascal visualization. The physics plant remains the explicitly
# uncalibrated 12 kg research surrogate; we do NOT infer Rascal flight fidelity
# from using manufacturer wingspan/length on a Plotly model.
# ============================================================================

def _compass_3d_arrow(fig: go.Figure, heading_deg: float, color: str,
                      label: str, radius: float, z: float, width: float) -> None:
    """Add an independently labelled 3-D azimuth needle (0=N, 90=E)."""
    theta = math.radians(90.0 - float(heading_deg))
    direction = np.array([math.cos(theta), math.sin(theta)])
    transverse = np.array([-direction[1], direction[0]])
    stem_end = direction * (radius - 0.13)
    fig.add_trace(go.Scatter3d(
        x=[0.0, float(stem_end[0])], y=[0.0, float(stem_end[1])], z=[z, z],
        mode="lines", name=label, legendgroup=label, showlegend=True,
        line=dict(color=color, width=width),
        hovertemplate=f"{label}: {heading_deg % 360:.2f}°<extra></extra>"
    ))
    tip = direction * radius
    base_a = stem_end + transverse * 0.055
    base_b = stem_end - transverse * 0.055
    fig.add_trace(go.Mesh3d(
        x=[tip[0], base_a[0], base_b[0]],
        y=[tip[1], base_a[1], base_b[1]], z=[z, z, z],
        i=[0], j=[1], k=[2], flatshading=True, color=color,
        opacity=1.0, showlegend=False, hoverinfo="skip", legendgroup=label
    ))


def compass_figure_3d(row: pd.Series, declination: float) -> go.Figure:
    """Rotatable three-dimensional neon-cyan instrument, with distinct headings."""
    fig = go.Figure()
    t = np.linspace(0.0, 2 * np.pi, 121)
    # Shallow machined dial: a dark face, raised toroidal rim and cyan halo.
    base_r = np.r_[0.0, 0.95 * np.cos(t[:-1])]
    base_y = np.r_[0.0, 0.95 * np.sin(t[:-1])]
    faces_n = len(t) - 1
    fig.add_trace(go.Mesh3d(
        x=base_r, y=base_y, z=np.zeros(len(base_r)) - 0.055,
        i=np.zeros(faces_n - 1, dtype=int),
        j=np.arange(1, faces_n, dtype=int),
        k=np.arange(2, faces_n + 1, dtype=int),
        color="#062025", opacity=0.87, hoverinfo="skip", showlegend=False,
        flatshading=True,
    ))
    ring_u = np.linspace(0.0, 2 * np.pi, 100)
    ring_v = np.linspace(0.0, 2 * np.pi, 11)
    uu, vv = np.meshgrid(ring_u, ring_v)
    tube = 0.035
    fig.add_trace(go.Surface(
        x=(1.0 + tube * np.cos(vv)) * np.cos(uu),
        y=(1.0 + tube * np.cos(vv)) * np.sin(uu),
        z=0.06 + tube * np.sin(vv),
        showscale=False, hoverinfo="skip", opacity=0.98,
        colorscale=[[0, NEON_BLUE_DEEP], [0.5, NEON_BLUE], [1, NEON_BLUE_BRIGHT]],
        surfacecolor=(np.cos(vv) + 1) / 2,
        lighting=dict(ambient=0.9, diffuse=0.55, specular=0.3),
    ))
    for radius, color, weight, opacity in [
        (1.035, NEON_BLUE, 12, .12), (1.022, NEON_BLUE, 5, .28),
        (.80, NEON_BLUE_DEEP, 2, .55), (.59, CYAN_GRID, 2, .48),
    ]:
        fig.add_trace(go.Scatter3d(
            x=radius * np.cos(t), y=radius * np.sin(t),
            z=np.full_like(t, 0.082), mode="lines",
            line=dict(color=color, width=weight), opacity=opacity,
            showlegend=False, hoverinfo="skip",
        ))
    # Two bundled traces keep the tick count small and GPU rendering efficient.
    for major in [False, True]:
        xx, yy, zz = [], [], []
        for deg in range(0, 360, 5):
            if (deg % 30 == 0) != major:
                continue
            theta = math.radians(90.0 - deg)
            inner = 0.86 if major else (0.925 if deg % 15 == 0 else 0.947)
            xx.extend([inner * math.cos(theta), .978 * math.cos(theta), None])
            yy.extend([inner * math.sin(theta), .978 * math.sin(theta), None])
            zz.extend([.102, .102, None])
        fig.add_trace(go.Scatter3d(
            x=xx, y=yy, z=zz, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=NEON_BLUE_BRIGHT if major else NEON_BLUE_DEEP,
                      width=5 if major else 2),
        ))
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    a = np.radians(90.0 - np.arange(0, 360, 45))
    fig.add_trace(go.Scatter3d(
        x=1.16 * np.cos(a), y=1.16 * np.sin(a),
        z=np.full(8, 0.17), mode="text", text=labels,
        textfont=dict(color=NEON_BLUE_BRIGHT, size=13),
        showlegend=False, hoverinfo="skip",
    ))
    _compass_3d_arrow(fig, float(row.true_heading_deg), NEON_BLUE,
                      "TRUE heading", .83, .19, 10)
    _compass_3d_arrow(fig, float(row.mag_true_heading_deg), ORANGE,
                      "MAG corrected", .73, .235, 8)
    _compass_3d_arrow(fig, float(row.fused_heading_deg), LIME,
                      "FUSED heading", .92, .275, 6)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[.29], mode="markers", showlegend=False,
        marker=dict(size=7, color=NEON_BLUE_BRIGHT), hoverinfo="skip",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title=dict(text=f"3D COMPASS  |  MAG DECLINATION {declination:+.1f}°",
                   font=dict(color=NEON_BLUE_BRIGHT, size=13), x=.03),
        font=dict(color=TEXT, family="Consolas, monospace"),
        height=480, margin=dict(l=0, r=0, t=54, b=6),
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.025,
                    font=dict(size=10, color=TEXT), bgcolor="rgba(0,0,0,0)"),
        scene=dict(
            bgcolor=BG, xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=.4),
            camera=dict(eye=dict(x=1.22, y=-1.65, z=1.45),
                        up=dict(x=0, y=0, z=1)),
            dragmode="orbit",
        ),
        uirevision="compass_neon_3d",
        showlegend=True,
    )
    return fig


def _rascal_geometry() -> dict[str, object]:
    """SIG-published span, length and wing area; remaining shape is illustrative."""
    reference = manufacturer_si()
    span = float(reference["span_m"])
    length = float(reference["fuselage_length_m"])
    mean_chord = float(reference["wing_area_m2"]) / span
    return dict(span=span, length=length,
                root_chord=mean_chord * 1.30,
                tip_chord=mean_chord * .70)


def _aircraft_rotation(heading_deg: float, pitch_deg: float,
                       roll_deg: float) -> np.ndarray:
    """Right-handed aircraft drawing axes FWD/PORT/UP mapped to East/North/Up.

    Heading: clockwise from true north. Pitch positive nose up. Roll positive
    right wing down. This is a *drawing transform*, not the plant FRD/NED DCM.
    """
    psi, theta, phi = np.radians([heading_deg, pitch_deg, roll_deg])
    forward = np.array([math.sin(psi) * math.cos(theta),
                        math.cos(psi) * math.cos(theta), math.sin(theta)])
    port_level = np.array([-math.cos(psi), math.sin(psi), 0.0])
    up_pitched = np.array([-math.sin(psi) * math.sin(theta),
                           -math.cos(psi) * math.sin(theta), math.cos(theta)])
    port = math.cos(phi) * port_level + math.sin(phi) * up_pitched
    up = -math.sin(phi) * port_level + math.cos(phi) * up_pitched
    return np.column_stack([forward, port, up])


def _aircraft_hull(fig: go.Figure, vertices: np.ndarray, faces: np.ndarray,
                   name: str, color: str, rotation: np.ndarray,
                   *, opacity: float = 1.0, hover: bool = False) -> None:
    rendered = np.asarray(vertices, dtype=float) @ rotation.T
    tri = np.asarray(faces, dtype=int)
    fig.add_trace(go.Mesh3d(
        x=rendered[:,0], y=rendered[:,1], z=rendered[:,2],
        i=tri[:,0], j=tri[:,1], k=tri[:,2],
        name=name, color=color, flatshading=True, opacity=opacity,
        lighting=dict(ambient=.68, diffuse=.67, specular=.28, roughness=.55),
        hovertemplate=(name + "<extra></extra>") if hover else None,
        hoverinfo=None if hover else "skip", showlegend=False,
    ))


def _aircraft_line(fig: go.Figure, points: np.ndarray, rot: np.ndarray,
                   color: str, width: float = 3, name: str = "") -> None:
    pts = np.asarray(points, dtype=float) @ rot.T
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="lines", line=dict(color=color, width=width),
        showlegend=False, hoverinfo="skip", name=name,
    ))


def _aircraft_wing_vertices(span: float, root_chord: float,
                            tip_chord: float, side: int,
                            *, tail: bool = False) -> np.ndarray:
    """Tapered thick wing solid; port is positive local y."""
    halfspan = span / 2
    y_root = side * (.055 if not tail else .018)
    y_tip = side * halfspan
    x_center_root = -.76 if tail else .085
    x_center_tip = -.82 if tail else .02
    z_root = .084 if tail else .135
    z_tip = .10 if tail else .208
    thick = .017 if tail else .030
    return np.array([
        [x_center_root + root_chord/2, y_root, z_root + thick],
        [x_center_root - root_chord/2, y_root, z_root + thick],
        [x_center_tip + tip_chord/2, y_tip, z_tip + thick],
        [x_center_tip - tip_chord/2, y_tip, z_tip + thick],
        [x_center_root + root_chord/2, y_root, z_root - thick],
        [x_center_root - root_chord/2, y_root, z_root - thick],
        [x_center_tip + tip_chord/2, y_tip, z_tip - thick],
        [x_center_tip - tip_chord/2, y_tip, z_tip - thick],
    ])


_AIRCRAFT_BOX_FACES = np.array([
    [0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7],
    [0, 2, 4], [4, 2, 6], [1, 5, 3], [5, 7, 3],
    [0, 4, 1], [1, 4, 5], [2, 3, 6], [3, 7, 6],
], dtype=int)


def aircraft_figure_3d(row: pd.Series, view: str = "Isometric",
                       heading_source: str = "True heading (physical)") -> go.Figure:
    """Interactively orbit a Rascal-shaped airframe linked to simulated attitude.

    Geometry uses manufacturer span/length/area. Its wing planform, tail size,
    fuselage sections and paint are visualization approximations, NOT CAD or CFD.
    """
    cfg = _rascal_geometry()
    heading_fields = {
        "True heading (physical)": "true_heading_deg",
        "Fused heading (estimated)": "fused_heading_deg",
        "Magnetic corrected (sensor)": "mag_true_heading_deg",
    }
    if heading_source not in heading_fields:
        raise ValueError("Unknown display heading source")
    yaw = float(row.get(heading_fields[heading_source], 0.0))
    pitch = float(row.get("true_pitch_deg", 0.0))
    bank = float(row.get("true_roll_deg", 0.0))
    rot = _aircraft_rotation(yaw, pitch, bank)
    fig = go.Figure()
    length = float(cfg["length"])
    span = float(cfg["span"])
    # Elliptical tapered fuselage with a three-dimensional, triangulated shell.
    stations = [
        (-.50 * length, .025, .030),
        (-.41 * length, .052, .060),
        (-.20 * length, .078, .090),
        (.04 * length, .115, .128),
        (.30 * length, .110, .110),
        (.44 * length, .067, .070),
        (.50 * length, .020, .027),
    ]
    n_circ = 12
    vertices = []
    for x, width, height in stations:
        for ang in np.linspace(0, 2*np.pi, n_circ, endpoint=False):
            vertices.append([x, width*math.cos(ang), .012 + height*math.sin(ang)])
    faces = []
    for k in range(len(stations) - 1):
        for j in range(n_circ):
            a = k*n_circ + j
            b = k*n_circ + (j + 1) % n_circ
            c = (k + 1)*n_circ + j
            d = (k + 1)*n_circ + (j + 1) % n_circ
            faces.extend([[a,b,c],[b,d,c]])
    _aircraft_hull(fig, np.asarray(vertices), np.asarray(faces),
                   "High-wing Rascal-style fuselage", "#E6F1F4", rot)

    for side in [-1,1]:
        main = _aircraft_wing_vertices(span, cfg["root_chord"],
                                      cfg["tip_chord"], side)
        _aircraft_hull(fig, main, _AIRCRAFT_BOX_FACES,
                       "Rascal-style high wing", "#DAEDF2", rot)
        # Cyan wing-tip outline is an identification marking, not a surface seam.
        _aircraft_line(fig, main[[2,3]], rot, NEON_BLUE, 5)
        tail = _aircraft_wing_vertices(.90, .23, .16, side, tail=True)
        _aircraft_hull(fig, tail, _AIRCRAFT_BOX_FACES,
                       "Horizontal stabilizer (illustrative)", "#AEC6CD", rot)
    # Vertical fin, at the tail: a thin tapered 3-D volume.
    fin = np.array([
        [-.89,.017,.07],[-.65,.017,.07],[-.84,.017,.33],[-.77,.017,.33],
        [-.89,-.017,.07],[-.65,-.017,.07],[-.84,-.017,.33],[-.77,-.017,.33],
    ])
    _aircraft_hull(fig, fin, _AIRCRAFT_BOX_FACES,
                   "Vertical stabilizer (illustrative)", "#B8D2D9", rot)
    # Visually distinct canopy and motor cowling markings.
    _aircraft_line(fig,
        np.array([[.22,-.107,.075],[.37,-.070,.13],[.52,0,.125],[.37,.070,.13],[.22,.107,.075]]),
        rot, NEON_BLUE_DEEP, 4)
    # Propeller disk located at the modeled nose, no thrust is inferred from it.
    theta = np.linspace(0, 2*np.pi, 73)
    nose_x = .50 * length + .020
    prop_loop = np.column_stack([
        np.full(len(theta),nose_x), .20 * np.cos(theta), .20 * np.sin(theta),
    ])
    _aircraft_line(fig, prop_loop, rot, "#6D9299", 2)
    _aircraft_line(fig, np.array([[nose_x,0,-.17],[nose_x,0,.17]]),rot,NEON_BLUE,5)

    # Fixed world reference prevents orbiting the model from obscuring heading.
    for axis, endpoint, color in [
        ("EAST", [1.70,0,-.47], NEON_BLUE),
        ("NORTH", [0,1.70,-.47], LIME),
        ("UP", [0,0,.90], ORANGE),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0,endpoint[0]], y=[0,endpoint[1]], z=[-.47,endpoint[2]],
            mode="lines+text", text=["",axis], textposition="top center",
            line=dict(color=color,width=3,dash="dash"),
            textfont=dict(color=color,size=10), hoverinfo="skip", showlegend=False,
        ))
    fwd = rot[:, 0]
    fig.add_trace(go.Scatter3d(
        x=[0, 1.25*fwd[0]], y=[0, 1.25*fwd[1]],
        z=[-.47, -.47], mode="lines", line=dict(color=NEON_BLUE, width=7),
        name="True heading projection", showlegend=False,
        hovertemplate=f"True heading {yaw:.1f}°<extra></extra>",
    ))
    preset = {
        "Isometric": dict(x=2.55,y=-2.95,z=1.88),
        "Top": dict(x=.18,y=-.12,z=4.5),
        "Side": dict(x=3.8,y=.3,z=1.0),
        "Chase": dict(x=-2.9 * fwd[0],y=-2.9 * fwd[1],z=1.45),
    }
    camera = preset.get(view, preset["Isometric"])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title=dict(text="SIG RASCAL 110  |  " + heading_source.upper(),
                   font=dict(color=NEON_BLUE_BRIGHT, size=12), x=.03),
        font=dict(color=TEXT, family="Consolas, monospace"),
        height=480, margin=dict(l=0, r=0, t=54, b=6),
        scene=dict(
            bgcolor=BG,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data", camera=dict(eye=camera, up=dict(x=0,y=0,z=1)),
            dragmode="orbit",
        ),
        uirevision=f"rascal_airframe_3d_{view}", showlegend=False,
    )
    return fig

def mission_plot(df: pd.DataFrame, row_idx: int, c: SimConfig, view3d: bool) -> go.Figure:
    q = df.iloc[row_idx]
    xs = df.true_e_m.to_numpy()
    ys = df.true_n_m.to_numpy()
    xe = df.estimated_e_m.to_numpy()
    ye = df.estimated_n_m.to_numpy()
    along = df.along_track_m.to_numpy()
    r = math.radians(c.heading_deg)
    ideal_e = np.sin(r) * along
    ideal_n = np.cos(r) * along
    if view3d:
        has_true_altitude = 'true_altitude_m' in df.columns
        z = df.true_altitude_m.to_numpy() if has_true_altitude else np.full(len(df),120.0)
        z_est = df.estimated_altitude_m.to_numpy() if has_true_altitude else z
        z_nominal = np.full(len(df),float(z[0]))
        fig = go.Figure()
        for name, x, y, heights, color, dash in [
            ("Nominal track", ideal_e, ideal_n, z_nominal, MUTED, "dash"),
            ("Truth trajectory", xs, ys, z, BLUE, "solid"),
            ("Estimated trajectory", xe, ye, z_est, LIME, "solid")
        ]:
            fig.add_trace(go.Scatter3d(x=x, y=y, z=heights, mode="lines", name=name,
                                       line=dict(color=color, width=5, dash=dash)))
        fig.add_trace(go.Scatter3d(x=[q.true_e_m], y=[q.true_n_m], z=[float(z[row_idx])], mode="markers",
                                   name="UAV cursor", marker=dict(color=ORANGE, size=7)))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[float(z[0])], mode="markers",
                                   name="Home", marker=dict(color=TEXT, size=5)))
        fig.update_layout(scene=dict(xaxis_title="East (m)", yaxis_title="North (m)",
                                     zaxis_title="Altitude (m)", aspectmode="manual",
                                     aspectratio=dict(x=1, y=1, z=0.3),
                                     bgcolor=BG), title=("3D aircraft flight trajectory" if has_true_altitude else
                                     "3D navigation overview (constant-altitude plane)"))
        return theme(fig, 520)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ideal_e, y=ideal_n, name="Nominal straight track", mode="lines",
                             line=dict(color=MUTED, dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=ys, name="True UAV path", mode="lines",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=xe, y=ye, name="Estimated path", mode="lines",
                             line=dict(color=LIME, width=2)))
    fig.add_trace(go.Scatter(x=[q.true_e_m], y=[q.true_n_m], name="UAV cursor", mode="markers",
                             marker=dict(color=ORANGE, size=14, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[q.estimated_e_m], y=[q.estimated_n_m], name="Estimated cursor",
                             mode="markers", marker=dict(color=LIME, size=10, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=[0], y=[0], name="Home", mode="markers",
                             marker=dict(color=TEXT, size=12, symbol="x")))
    fig.update_layout(title="Closed-loop mission trajectory", xaxis_title="East (m)",
                      yaxis_title="North (m)", yaxis=dict(scaleanchor="x", scaleratio=1))
    return theme(fig, 515)


def animated_mission(df: pd.DataFrame, c: SimConfig) -> go.Figure:
    """Browser-side playback; avoids Streamlit sleep/rerun loops on mobile."""
    fig = mission_plot(df, 0, c, False)
    indices = list(range(0, len(df), max(1, len(df) // 115)))
    if indices[-1] != len(df)-1:
        indices.append(len(df)-1)
    frames = []
    for i in indices:
        q = df.iloc[i]
        frames.append(go.Frame(name=str(i), traces=[3, 4], data=[
            go.Scatter(x=[q.true_e_m], y=[q.true_n_m], mode="markers",
                       marker=dict(color=ORANGE, size=14, symbol="diamond")),
            go.Scatter(x=[q.estimated_e_m], y=[q.estimated_n_m], mode="markers",
                       marker=dict(color=LIME, size=10, symbol="circle-open"))
        ]))
    fig.frames = frames
    steps = [dict(method="animate", label=f"{df.time_s.iloc[i]:.0f}s", args=[str(i),
        dict(mode="immediate", frame=dict(duration=70, redraw=False), transition=dict(duration=0))])
             for i in indices]
    fig.update_layout(updatemenus=[dict(type="buttons", direction="left", showactive=False,
        x=0.0, y=-0.12, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=70, redraw=False),
                  transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="❚❚ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                  transition=dict(duration=0), mode="immediate")]),
        ])], sliders=[dict(active=0, x=0.23, y=-0.085, len=0.75,
                          currentvalue=dict(prefix="Replay index: "), steps=steps)],
        margin=dict(l=15, r=15, t=55, b=85))
    return fig


def sensor_plot(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Heading error, wrapped (deg)",
                                        "3-axis magnetometer field magnitude (μT)",
                                        "Position estimate: error / 95% bound (m)"))
    t = df.time_s
    for vals, name, color, row in [
        (df.mag_error_deg, "Mag heading error", ORANGE, 1),
        (df.fusion_error_deg, "Fusion heading error", LIME, 1),
        (df.mag_field_ut, "Measured total field (3-axis)", BLUE, 2),
        (df.mag_field_expected_ut, "Expected total field", MUTED, 2),
        (df.position_error_m, "Truth vs estimate", ORANGE, 3),
        (df.position_bound95_m, "Covariance 95% bound", LIME, 3),
    ]:
        fig.add_trace(go.Scatter(x=t, y=vals, mode="lines", name=name,
                                 line=dict(color=color, width=1.9)), row=row, col=1)
    expected = float(df.mag_field_expected_ut.iloc[0])
    for boundary in [.72 * expected, 1.28 * expected]:
        fig.add_hline(y=boundary, row=2, col=1, line_dash="dot", line_color=RED, opacity=0.45)
    fig.update_xaxes(title_text="Elapsed simulation time (s)", row=3, col=1)
    return theme(fig, 780)


def events_df(df: pd.DataFrame) -> pd.DataFrame:
    items = []
    prev_health = None
    prev_mag = True
    prev_gnss = True
    prev_course = False
    for r in df.itertuples():
        if r.health != prev_health:
            items.append((r.time_s, "SUPERVISOR", r.health))
        if not r.mag_accepted and prev_mag:
            items.append((r.time_s, "MAG", "Magnetometer correction rejected"))
        if not r.gnss_accepted and prev_gnss:
            items.append((r.time_s, "GNSS", "No accepted position update"))
        if r.mag_course_disagreement and not prev_course:
            items.append((r.time_s, "MAG", "Course versus magnetometer disagreement"))
        if r.gnss_reinitialized:
            items.append((r.time_s, "GNSS", "Guarded reacquisition reset: NOT NIS-gated acceptance"))
        prev_health, prev_mag, prev_gnss, prev_course = (r.health, r.mag_accepted,
                                                         r.gnss_accepted, r.mag_course_disagreement)
    return pd.DataFrame(items, columns=["Time (s)", "Subsystem", "Event"])


def config_from_sidebar() -> SimConfig:
    default = st.session_state.get("cfg", AerospaceConfig())
    is_advanced = isinstance(default,AerospaceConfig)
    with st.sidebar:
        st.markdown(
            '<div class="hud-sidebar-brand">'
            '<div class="eyebrow">UAS // DIGITAL ENGINEERING LAB</div>'
            '<div class="name">NAVIGATION CONTROL CENTER</div>'
            '<div class="sub">SENSOR INTEGRITY • FLIGHT SIMULATION</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<div class="hud-side-note">CONFIGURE YOUR SCENARIO BELOW, THEN SELECT '
                    '<b>RUN SIMULATION</b> TO UPDATE THE DIGITAL TWIN.</div>',
                    unsafe_allow_html=True)
    with st.sidebar.form("scenario_configuration"):
        st.markdown('<div class="hud-side-section">01 // MISSION & FAULT CONTROL</div>',
                    unsafe_allow_html=True)
        profile = st.selectbox(
            "PHYSICS / ESTIMATION MODEL",
            ["6-DOF fixed-wing research surrogate (v4)",
             "Aircraft-aware bank-to-turn (v3)", "Legacy planar model (v2)"],
            index=0 if isinstance(default,SixDOFConfig) else 1 if is_advanced else 2,
            help="v4: nonlinear six-degree-of-freedom surrogate + 3-axis sensors. The coefficients are ASSUMED, not calibrated against a Rascal 110. v3/v2 are retained for comparison.")
        sixdof_selected = profile.startswith("6-DOF")
        advanced_selected = not profile.startswith("Legacy")
        mode = st.selectbox("Navigation controller", MODES, index=MODES.index(default.mode))
        scenario = st.selectbox("Fault injection", FAULTS, index=FAULTS.index(default.scenario))
        hdg = st.slider("True reference heading (°)", 0, 359, int(default.heading_deg))
        speed = st.slider("Airspeed (m/s)", 13.0 if sixdof_selected else 3.0,
                          31.0 if sixdof_selected else 45.0,
                          float(np.clip(default.speed_mps,13 if sixdof_selected else 3,31 if sixdof_selected else 45)),0.5)
        duration = st.slider("Mission duration (s)", 40, 300, int(default.duration_s), 10)
        dt = st.select_slider("Telemetry display interval (s)" if advanced_selected else "Integration interval (s)",
                              options=[0.25, 0.5, 1.0], value=default.dt_s)
        if advanced_selected:
            st.caption("INTERNAL TIMING: IMU 100 Hz | MAG 10 Hz | GNSS 5 Hz | CONTROL 20 Hz. Display sampling does not alter integration. v4 applies nonlinear 6-DOF plant and three-axis IMU physics.")
            bank = st.slider("Fixed-wing maximum bank (°)", 10, 45,
                             int(getattr(default,"bank_limit_deg",30)), 1)
            if sixdof_selected:
                roll_tau=float(getattr(default,"roll_tau_s",.85))
                st.caption("v4: bank/turn response comes from the aerodynamic moments, body inertia and finite actuator lag. The v3 roll-time-constant parameter is not used.")
            else:
                roll_tau = st.slider("Roll response time constant (s)", .3, 2.5,
                                     float(getattr(default,"roll_tau_s",.85)), .05)
            st.markdown('<div class="hud-side-section">GEOMAGNETIC REFERENCE</div>',unsafe_allow_html=True)
            field_mode = st.selectbox("Earth field source",["Synthetic field","WMM2025"],
                    index=0 if getattr(default,"magnetic_model","Synthetic field")=="Synthetic field" else 1)
            if field_mode == "WMM2025":
                lat = st.number_input("Latitude (°)", -90.0, 90.0, float(getattr(default,"latitude_deg",29.2)), step=.1)
                lon = st.number_input("Longitude (°)", -180.0, 180.0, float(getattr(default,"longitude_deg",-81.0)), step=.1)
                alt_m = st.number_input("WGS84 ellipsoid altitude (m)", -100.0, 30000.0, float(getattr(default,"altitude_ellipsoid_m",100.0)), step=10.0)
                dec_year = st.number_input("Decimal year (WMM valid 2025-2030)",2025.0,2029.99,float(getattr(default,"decimal_year",2026.84)),step=.01)
            else:
                lat,lon,alt_m,dec_year = 29.2,-81.0,100.0,2026.84
        a, b = st.slider("Fault interval (% of flight)", 0, 100, (int(100 * default.fault_start_s/default.duration_s), int(100 * default.fault_end_s/default.duration_s)))
        st.markdown('<div class="hud-side-section">02 // ENVIRONMENT & SENSORS</div>',
                    unsafe_allow_html=True)
        decl = st.slider("Actual declination (° E positive)", -25.0, 25.0, float(default.declination_deg), 0.5)
        decl_assumed = st.slider("Applied declination correction (°)", -25.0, 25.0, float(default.declination_model_deg), 0.5)
        wind_e = st.slider("Wind: east component (m/s)", -8.0, 8.0, float(default.wind_e_mps), 0.2)
        wind_n = st.slider("Wind: north component (m/s)", -8.0, 8.0, float(default.wind_n_mps), 0.2)
        gyro_bias = st.slider("Gyro bias (°/s)", -0.5, 0.5, float(default.gyro_bias_dps), 0.005)
        mag_noise = st.slider("Mag noise (μT, 1σ)", 0.0, 2.0, float(default.magnetometer_noise_ut), 0.05)
        inc = st.slider("Synthetic magnetic inclination (° down)", -75.0, 75.0,
                        float(default.inclination_deg), 1.0)
        field_h = st.slider("Horizontal magnetic field (μT)", 10.0, 40.0,
                            float(default.horizontal_field_ut), 1.0)
        gnss_noise = st.slider("GNSS noise (m, 1σ)", 0.5, 15.0, float(default.gnss_sigma_m), 0.5)
        st.markdown('<div class="hud-side-section">03 // AUTONOMY & REPLAY</div>',
                    unsafe_allow_html=True)
        rtb = st.slider("Conditional RTB threshold (m)", 30, 300, int(default.rtb_threshold_m), 5)
        course_gate = st.slider("Stable-course magnetic discrepancy gate (°)",
                                3.0, 15.0, float(default.course_check_gate_deg), 0.5,
                                help="Only used after a long valid GNSS position baseline during approximately straight flight. It depends on an approximate wind model.")
        seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=int(default.seed))
        run = st.form_submit_button("▶ RUN SIMULATION", use_container_width=True, type="primary")
    if run:
        if b <= a:
            st.sidebar.error("Fault interval must have positive duration; retaining previous scenario.")
        else:
            start = max(0.0, duration * a / 100.0)
            end = duration * b / 100.0
            common=dict(
                heading_deg=hdg, declination_deg=decl, declination_model_deg=decl_assumed,
                speed_mps=speed, wind_e_mps=wind_e, wind_n_mps=wind_n,
                duration_s=duration, dt_s=dt, scenario=scenario,
                fault_start_s=start, fault_end_s=end, mode=mode, calibrated=False,
                magnetometer_noise_ut=mag_noise, gyro_bias_dps=gyro_bias,
                gnss_sigma_m=gnss_noise, rtb_threshold_m=rtb, seed=int(seed),
                inclination_deg=inc, horizontal_field_ut=field_h,
                course_check_gate_deg=course_gate,
            )
            params=(dict(bank_limit_deg=bank,roll_tau_s=roll_tau,magnetic_model=field_mode,
                         latitude_deg=lat,longitude_deg=lon,altitude_ellipsoid_m=alt_m,
                         decimal_year=dec_year) if advanced_selected else {})
            proposed = (SixDOFConfig(**common,**params) if sixdof_selected else
                        AerospaceConfig(**common,**params) if advanced_selected else SimConfig(**common))
            try:
                result, m = simulate(proposed)
                st.session_state.cfg = proposed
                st.session_state.df,st.session_state.metrics=result,m
                st.session_state.pop("mc", None)
            except (ValueError,RuntimeError) as exc:
                st.sidebar.error(f"Simulation not run: {exc}")
    with st.sidebar:
        st.markdown('<div class="hud-side-section">04 // INSTRUMENT CALIBRATION</div>',
                    unsafe_allow_html=True)
        if st.button("🧭 RUN SYNTHETIC COMPASS CALIBRATION", use_container_width=True,
                     help="v3/v4 fit a synthetic 3-axis ellipsoid from simulated orientation samples. This is not flight-hardware calibration."):
            base = st.session_state.get("cfg", SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.))
            proposed = replace(base, calibrated=True)
            try:
                result,m = simulate(proposed)
                st.session_state.cfg = proposed
                st.session_state.df,st.session_state.metrics=result,m
                st.session_state.pop("mc", None)
            except (ValueError, RuntimeError) as exc:
                st.sidebar.error(f"Calibration experiment failed: {exc}")
        if st.button("↺ RESTORE DEFAULT SCENARIO", use_container_width=True):
            st.session_state.cfg = SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.)
            st.session_state.df, st.session_state.metrics = simulate(st.session_state.cfg)
            st.session_state.pop("mc", None)
            st.rerun()
        st.caption("v3/v4 calibration fits an ellipsoid to a separate synthetic 3D orientation sweep. It does not use actual sensor measurements and cannot calibrate transient EMI. v2 retains its simplified calibration model.")
        st.markdown('<div class="hud-side-note"><b>RESEARCH SIMULATION ONLY</b><br>'
                    'No operational flight hardware, GNSS receiver, or telemetry is connected.'
                    '</div>', unsafe_allow_html=True)
    return st.session_state.get("cfg", SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.))


if "cfg" not in st.session_state:
    st.session_state.cfg = SixDOFConfig(duration_s=90,fault_start_s=25.,fault_end_s=65.)
if "df" not in st.session_state:
    st.session_state.df, st.session_state.metrics = simulate(st.session_state.cfg)

c = config_from_sidebar()
df = st.session_state.df
metrics = st.session_state.metrics

health = str(df.health.iloc[-1])
health_css = "fault" if any(x in health.upper() for x in ("UNRELIABLE", "CRITICAL", "RTB")) else (
    "amber" if any(x in health.upper() for x in ("DEGRADED", "WARN")) else ""
)
st.markdown(
    '<div class="hud-hero">'
    '<div class="kicker">AEROSPACE DIGITAL TWIN // NAVIGATION LAB</div>'
    '<h1>UAV <span>Compass</span> Variance Simulator</h1>'
    '<div class="subtitle">PHYSICS-INFORMED FLIGHT SIMULATION / SENSOR FUSION / '
    'FAULT-TOLERANT AUTONOMY / UNCERTAINTY ANALYSIS / V&amp;V</div>'
    '<div class="statusline">'
    '<span class="hud-chip">● ENGINE READY</span>'
    '<span class="hud-chip cyan">◉ COMPASS INSTRUMENTATION</span>'
    f'<span class="hud-chip {health_css}">MISSION HEALTH: {health}</span>'
    '<span class="hud-chip amber">SYNTHETIC / NOT FLIGHT QUALIFIED</span>'
    '</div></div>', unsafe_allow_html=True,
)
hud_kpis([
    ("MAX CROSS-TRACK", f"{metrics['max_cross_track_m']:,.1f} m", ""),
    ("FINAL ESTIMATOR ERROR", f"{metrics['end_position_error_m']:,.1f} m", ""),
    ("MAX FUSED HEADING ERROR", f"{metrics['max_fusion_heading_error_deg']:.2f}°", "blue"),
    ("MAG UPDATES ACCEPTED", f"{metrics['mag_accept_rate_pct']:.1f}%", "green"),
    ("MISSION HEALTH", health, health_css or "green"),
])

st.markdown(
    f'<div class="hud-side-note" style="margin:12px 0 17px;">'
    f'ACTIVE SCENARIO: <b>{c.scenario}</b> &nbsp; | &nbsp; '
    f'PHYSICS PROFILE: <b>{"SIX-DOF SURROGATE v4" if isinstance(c,SixDOFConfig) else "BANK-TO-TURN v3" if isinstance(c,AerospaceConfig) else "LEGACY v2"}</b> &nbsp; | &nbsp; '
    f'CONTROLLER: <b>{c.mode}</b> &nbsp; | &nbsp; '
    f'SEED: <b>{c.seed}</b> &nbsp; | &nbsp; '
    f'CALIBRATION: <b>{"ON" if c.calibrated else "OFF"}</b>'
    '</div>', unsafe_allow_html=True,
)

st.markdown(
    '<div class="hud-side-note" style="margin:10px 0 8px;border-color:rgba(0,229,255,.33);">'
    '<b style="color:#73F6FF;">LINKED 3D FLIGHT INSTRUMENTS</b> &nbsp; '
    'The mission track, 3D compass and Rascal airframe use the same simulation-time selection.'
    '</div>', unsafe_allow_html=True)
# Keep the synchronized inspector in bounds when a new mission is shorter.
if "linked_flight_time_s" in st.session_state:
    st.session_state["linked_flight_time_s"] = float(np.clip(
        st.session_state["linked_flight_time_s"], 0.0, float(df.time_s.iloc[-1])
    ))
shared_flight_time_s = st.slider(
    "Synchronized mission / compass / aircraft time (s)",
    min_value=0.0, max_value=float(df.time_s.iloc[-1]), value=0.0,
    step=float(c.dt_s), key="linked_flight_time_s",
)

mission_tab, compass_tab, integrity_tab, uq_tab, verification_tab, sources_tab = st.tabs([
    "✈ MISSION & PLAYBACK", "🧭 COMPASS INSTRUMENT", "📡 SENSOR INTEGRITY",
    "📊 MONTE CARLO", "✅ V&V / EXPORT", "📚 AIRCRAFT SOURCES",
])

with mission_tab:
    panel_header("01 / GUIDANCE, NAVIGATION & CONTROL", "Mission & Playback",
                 "Inspect truth, estimated state, trajectory deviation, and browser-side playback.")
    selected_t = shared_flight_time_s
    st.caption(f"Linked mission snapshot: {selected_t:.2f} s. Adjust the synchronized time slider above the tabs.")
    idx = min(len(df)-1, max(0, int(round(selected_t / c.dt_s))))
    snap = df.iloc[idx]
    hud_kpis([
        ("TRUE-NORTH HEADING", f"{snap.true_heading_deg:.1f}°", "blue"),
        ("FUSED HEADING", f"{snap.fused_heading_deg:.1f}°", "green"),
        ("ACTUAL CROSS-TRACK", f"{snap.cross_track_m:+,.1f} m", ""),
        ("POSITION 95% BOUND", f"{snap.position_bound95_m:.1f} m", ""),
    ], four=True)
    view3d = st.checkbox("Show 3D mission overview" + (" (actual modeled altitude)" if isinstance(c,SixDOFConfig) else " (constant-altitude visualization)"), value=False)
    st.plotly_chart(mission_plot(df, idx, c, view3d), use_container_width=True)
    with st.expander("▶ Browser-side mission playback", expanded=True):
        st.caption("Play animates the simulated UAV marker. The time inspector above is independent and permits precise numerical inspection.")
        st.plotly_chart(animated_mission(df, c), use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.along_track_m, y=df.cross_track_m,
                             mode="lines", name="True cross-track", line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=df.along_track_m, y=df.estimated_cross_track_m,
                             mode="lines", name="Estimated cross-track", line=dict(color=LIME, width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    fig.update_layout(title="Cross-track error versus along-track distance", xaxis_title="Along-track displacement (m)",
                      yaxis_title="Cross-track offset (m)")
    st.plotly_chart(theme(fig, 340), use_container_width=True)

with compass_tab:
    panel_header("02 / SYNCHRONIZED 3D INSTRUMENTS", "3D Compass & Rascal Airframe",
                 "Orbit, zoom or pan each Plotly model. Both respond to the linked mission time above.",
                 compass=True)
    view_idx = shared_flight_time_s
    q = df.iloc[min(len(df)-1, max(0, int(round(view_idx / c.dt_s))))]
    view_col, heading_col = st.columns(2)
    with view_col:
        aircraft_view = st.selectbox("Aircraft 3D camera", ["Isometric", "Top", "Side", "Chase"],
                                     key="aircraft_3d_view",
                                     help="Orbit and zoom freely after selecting a camera preset.")
    with heading_col:
        aircraft_heading_source = st.selectbox(
            "Aircraft heading display",
            ["True heading (physical)", "Fused heading (estimated)",
             "Magnetic corrected (sensor)"],
            key="aircraft_heading_display",
            help="Only changes the displayed aircraft orientation; never the simulator, autopilot, or truth state.",
        )
    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.plotly_chart(compass_figure_3d(
            q, float(q.get("magnetic_declination_deg", c.declination_deg))),
            use_container_width=True, key="linked_neon_compass_3d",
            config={"displaylogo":False, "scrollZoom":True},
        )
    with right:
        st.plotly_chart(aircraft_figure_3d(q, aircraft_view, aircraft_heading_source),
                        use_container_width=True, key="linked_rascal_airframe_3d",
                        config={"displaylogo":False, "scrollZoom":True})
    st.caption("The compass uses TRUE (cyan), magnetic declination-corrected (amber) and FUSED (green) headings. "
               "The aircraft shows the selected heading source at the SAME timestamp; "
               "bank and pitch use true simulated attitude. Switching the heading source changes "
               "the visualization only, never the aircraft's actual path or flight controller. "
               "Positive roll corresponds to right wing down; positive pitch is nose up.")
    if "true_roll_deg" not in df.columns or "true_pitch_deg" not in df.columns:
        st.info("The selected legacy physics profile does not model all attitude axes. "
                "Missing bank/pitch angles appear as 0° in the 3D airframe visualization.")
    model_reference = manufacturer_si()
    st.caption(f"Geometry reference: SIG Rascal 110 EG ARF | published wingspan "
               f"{model_reference['span_m']:.3f} m, fuselage length "
               f"{model_reference['fuselage_length_m']:.3f} m and wing area "
               f"{model_reference['wing_area_m2']:.3f} m². "
               "Wing planform, tail, fuselage profile and markings are illustrative, not OEM CAD. "
               "The active v4 flight dynamics remain a generic, uncalibrated 12 kg research surrogate, "
               "not a Rascal-specific physics model.")
    compass_readouts(q)
    hud_kpis([
        ("SIMULATED BANK", f"{float(q.get('true_roll_deg', 0.0)):+.1f}°", "blue"),
        ("SIMULATED PITCH", f"{float(q.get('true_pitch_deg', 0.0)):+.1f}°", "blue"),
        ("SIMULATION TIME", f"{float(q.time_s):.2f} s", ""),
        ("MAGNETIC INTEGRITY", "ACCEPTED" if bool(q.mag_accepted) else "REJECTED", "green" if bool(q.mag_accepted) else "fault"),
    ], four=True)
    with st.expander("View the original neon 2D compass", expanded=False):
        original_compass = compass_figure(q, float(q.get("magnetic_declination_deg", c.declination_deg)))
        st.pyplot(original_compass, use_container_width=True)
        plt.close(original_compass)
    panel_header("HEADING ERROR GEOMETRY", "Uncompensated Drift",
                 "Straight-line theoretical reference, not the closed-loop simulated trajectory.",
                 compass=True)
    theoretical = theoretical_drift_m_per_km(float(q.mag_error_deg))
    if np.isfinite(theoretical):
        st.markdown(
            f'<div class="compass-note"><strong>{theoretical:,.1f} m</strong> lateral deviation '
            'per 1 km along-track, with a fixed uncompensated heading offset.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Forward along-track drift geometry is undefined for a 90° or greater uncompensated heading offset.")
    st.caption("This geometric benchmark assumes constant heading error, no crosswind, and no closed-loop correction. The mission plot uses simulated closed-loop dynamics.")
    st.markdown(
        f'<div class="compass-note">MAG SAMPLE: <strong>{"ACCEPTED" if q.mag_accepted else "REJECTED"}</strong>'
        f' &nbsp; | &nbsp; FIELD: <strong>{q.mag_field_ut:.2f} μT</strong>'
        f' &nbsp; | &nbsp; INNOVATION: <strong>{q.mag_innovation_deg:+.2f}°</strong></div>',
        unsafe_allow_html=True,
    )
    st.caption("Actual declination and applied correction are separately configurable. A heading offset from a magnetometer fault is not the same as geographic magnetic variation.")

with integrity_tab:
    panel_header("03 / SENSOR HEALTH & FAULT MONITORING", "Sensor Integrity",
                 "Coupled EKF, multi-rate sampling (v3), innovation gates, independent course checks, and uncertainty diagnostics.")
    st.plotly_chart(sensor_plot(df), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mag corrections rejected", int(metrics["mag_rejections"]))
    c2.metric("GNSS updates rejected", int(metrics["gnss_rejections"]))
    c3.metric("GNSS accepted", f"{metrics['gnss_accept_rate_pct']:.1f}%")
    c4.metric("Max position error", f"{metrics['max_position_error_m']:.1f} m")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("GNSS-course mag discrepancies", int(metrics["mag_course_rejections"]))
    d2.metric("Observed 95% bound coverage", f"{metrics['position_95_coverage_pct']:.1f}%")
    d3.metric("Mean position NEES (2 DOF)", f"{metrics['mean_position_nees']:.2f}")
    d4.metric("Estimated gyro bias at end", ("Not estimated (v4)" if isinstance(c,SixDOFConfig)
                  else f"{df.estimated_gyro_bias_dps.iloc[-1]:+.3f}°/s"))
    if metrics["gnss_reinitializations"] > 0:
        st.warning(f"{int(metrics['gnss_reinitializations'])} guarded GNSS reacquisition reset(s). The reset is not a statistically validated observation update and cannot identify coordinated spoofing after an outage.")
    fig_ekf = go.Figure()
    fig_ekf.add_trace(go.Scatter(x=df.time_s, y=df.gnss_nis, mode="lines",
                                 name="GNSS NIS", line=dict(color=ORANGE, width=1.5)))
    fig_ekf.add_hline(y=16.811893829770927 if isinstance(c,SixDOFConfig) else 9.21034037197618,
                      line_dash="dash", line_color=RED,
                      annotation_text="χ²₆ 99% gate" if isinstance(c,SixDOFConfig) else "χ²₂ 99% gate")
    fig_ekf.update_layout(title="Innovation consistency: GNSS normalized innovation squared",
                           xaxis_title="Simulation time (s)", yaxis_title="NIS")
    st.plotly_chart(theme(fig_ekf, 340), use_container_width=True)
    events = events_df(df)
    st.subheader("Supervisory event log")
    st.dataframe(events, use_container_width=True, hide_index=True)
    st.caption("v4 uses a physically distinct six-DOF research plant, 3-axis gyroscope and accelerometer, complementary attitude filter and six-state 3D kinematic Kalman filter. This is NOT a full strapdown 15-state INS. v3 retains its coupled EN/yaw/bias/wind EKF. All reported uncertainty remains synthetic and must be externally validated.")
    if isinstance(c,AerospaceConfig):
        st.subheader("Aircraft dynamics: roll and coordinated turn")
        roll_fig=go.Figure()
        roll_fig.add_trace(go.Scatter(x=df.time_s,y=df.true_roll_deg,mode="lines",name="True bank",line=dict(color=NEON_BLUE)))
        roll_fig.add_trace(go.Scatter(x=df.time_s,y=df.roll_command_deg,mode="lines",name="Bank command",line=dict(color=ORANGE,dash="dash")))
        roll_fig.update_layout(title=("Nonlinear six-DOF roll response" if isinstance(c,SixDOFConfig) else "Bank-limited turn response (not a 6-DOF airframe)"),xaxis_title="Time (s)",yaxis_title="Right bank (°)")
        st.plotly_chart(theme(roll_fig,350),use_container_width=True)
        if isinstance(c,SixDOFConfig):
            a1,a2,a3,a4=st.columns(4)
            a1.metric("Max modeled pitch",f"{metrics['max_abs_pitch_deg']:.1f}°")
            a2.metric("Min true altitude",f"{metrics['min_altitude_m']:.1f} m")
            a3.metric("Airspeed span",f"{metrics['min_airspeed_mps']:.1f} to {metrics['max_airspeed_mps']:.1f} m/s")
            a4.metric("Max sideslip",f"{metrics['max_sideslip_deg']:.1f}°")
            figdyn=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("Pitch / true altitude","Angle of attack / sideslip"))
            for vals,nm,col,rr in [(df.true_pitch_deg,"Pitch (deg)",NEON_BLUE,1),
                                    (df.true_altitude_m,"Altitude (m)",LIME,1),
                                    (df.angle_of_attack_deg,"Angle of attack (deg)",ORANGE,2),
                                    (df.true_sideslip_deg,"Sideslip (deg)",RED,2)]:
                figdyn.add_trace(go.Scatter(x=df.time_s,y=vals,mode="lines",name=nm,line=dict(color=col)),row=rr,col=1)
            figdyn.update_layout(title="Six-DOF aerodynamic / altitude response",xaxis2_title="Time (s)")
            st.plotly_chart(theme(figdyn,570),use_container_width=True)
            st.caption("Surrogate coefficients are assumed. Controller uses estimated vertical state and synthetic pitot feedback; no real-airframe identification or external SITL calibration.")
        else:
            st.caption(f"Maximum simulated bank: {metrics['max_abs_bank_deg']:.1f}°. Maximum turn rate: {metrics['max_abs_turn_rate_dps']:.1f}°/s. Constant speed and coordinated-turn assumptions apply. No pitch, throttle, stall, or identified aerodynamics.")

with uq_tab:
    panel_header("04 / UNCERTAINTY QUANTIFICATION", "Monte Carlo Sensitivity",
                 "Repeatable noise realizations and mission error distributions.")
    n_runs = (st.slider("Number of simulation realizations",3,20,5,1)
              if isinstance(c,SixDOFConfig) else
              st.slider("Number of simulation realizations",5,60,15,5)
              if isinstance(c,AerospaceConfig)
              else st.slider("Number of simulation realizations",10,150,40,10))
    st.caption("Samples independent sensor noise sequences at the current scenario and configuration. Assumed airframe coefficients, structural disturbances, and many systematic errors remain fixed. Noise-ensemble consistency is not model uncertainty or independent validation.")
    if st.button("▶ RUN MONTE CARLO", type="primary"):
        with st.spinner("Computing Monte Carlo realizations..."):
            st.session_state.mc = monte_carlo(c, n_runs)
            st.session_state.mc_config = c
    mc = st.session_state.get("mc")
    if mc is not None and st.session_state.get("mc_config") == c:
        c1, c2, c3 = st.columns(3)
        c1.metric("Median max cross-track", f"{mc.max_cross_track_m.median():.1f} m")
        c2.metric("95th percentile max error", f"{mc.max_cross_track_m.quantile(.95):.1f} m")
        c3.metric("Exceedance > 100 m", f"{100*(mc.max_cross_track_m>100).mean():.1f}%")
        st.caption(f"Mean model-based 95% bound coverage across noise realizations: {mc.position_95_coverage_pct.mean():.1f}%. This is not independent model validation; calibration can be conservative or scenario dependent.")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mc.max_cross_track_m, marker_color=ORANGE, name="Max cross-track"))
        fig.update_layout(title="Ensemble distribution: maximum absolute cross-track", xaxis_title="Max cross-track (m)", yaxis_title="Realization count")
        st.plotly_chart(theme(fig, 390), use_container_width=True)
        st.dataframe(mc.round(3), use_container_width=True, hide_index=True)
        st.download_button("Download Monte Carlo CSV", mc.to_csv(index=False), "compass_twin_monte_carlo.csv", "text/csv")

with verification_tab:
    panel_header("05 / SYSTEMS ENGINEERING", "Verification & Data Export",
                 "Requirements evidence, reproducible telemetry, and model fidelity limits.")
    checks = v_and_v(df, metrics)
    v1, v2 = st.columns(2)
    v1.metric("Checks passing", f"{int(checks.Pass.sum())} / {len(checks)}")
    v2.metric("Demonstration checks failing", int((~checks.Pass).sum()))
    checks_disp = checks.copy()
    checks_disp["Result"] = checks_disp["Pass"].map({True: "PASS", False: "FAIL"})
    st.dataframe(checks_disp.drop(columns=["Pass"]), use_container_width=True, hide_index=True)
    st.caption("These are prototype numerical / systems checks with illustrative mission thresholds, not certification evidence. An independent SITL / flight data comparison is needed for external validation.")
    if isinstance(c,SixDOFConfig):
        st.subheader("v4 / Independent SITL or flight truth comparison")
        st.caption("Upload a matched, independently produced JSBSim/PX4 or instrumented-flight truth CSV plus a matching SHA-256 provenance manifest. The tool checks schema and hash, not the authenticity of the source. An unmatched flight cannot establish accuracy.")
        ref_file=st.file_uploader("Independent reference truth CSV",type=["csv"],key="reference_v4")
        manifest_file=st.file_uploader("Reference provenance manifest JSON",type=["json"],key="manifest_v4")
        if ref_file is not None:
            raw=ref_file.getvalue()
            st.download_button("Create manifest TEMPLATE for this CSV",json.dumps(reference_template_manifest(raw),indent=2),"reference_manifest_TEMPLATE.json","application/json")
            st.caption("The template is not evidence. Replace all placeholders, document control/wind/time alignment, and preserve the original reference log.")
            if manifest_file is not None:
                try:
                    manifest=json.loads(manifest_file.getvalue().decode('utf-8'))
                    provenance=check_manifest(manifest,raw)
                    df_ref=load_truth_csv(raw)
                    res,scores=compare_truth_v4(df,df_ref)
                    cols=st.columns(4)
                    for j,(key,label,unit) in enumerate([('position_rmse_m','Position RMSE','m'),('heading_rmse_deg','Heading RMSE','°'),('bank_rmse_deg','Bank RMSE','°'),('altitude_rmse_m','Altitude RMSE','m')]):
                        cols[j].metric(label,f"{scores[key]:.2f} {unit}")
                    st.dataframe(pd.DataFrame([{'Metric':k,'Observed':v['observed'],'Provisional limit':v['limit'],'Pass':v['pass']} for k,v in scores['engineering_gates'].items()]),hide_index=True,use_container_width=True)
                    st.warning("These are USER-DECLARED, NOT AUTHENTICATED reference data. Passing provisional gates does NOT certify independent validation or 9/10 fidelity. Verify trial match, aircraft identification, provenance, and holdout status manually.")
                    st.download_button("Download residuals",res.to_csv(index=False),"v4_reference_residuals.csv","text/csv")
                except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                    st.error(f"Reference data rejected: {exc}")
    elif isinstance(c,AerospaceConfig):
        st.subheader("Independent reference telemetry comparison")
        st.caption("Provide MATCHED independent SITL or flight truth telemetry in local East/North (meters), true heading (degrees), right bank (degrees), aligned to simulation time and identical initial conditions. This tool does not authenticate the source or align different missions.")
        reference_file = st.file_uploader("SITL / flight truth CSV", type=["csv"], key="reference_csv")
        if reference_file is not None:
            try:
                ref=pd.read_csv(reference_file)
                residuals,scores=compare_reference(df,ref)
                rr1,rr2,rr3=st.columns(3)
                rr1.metric("Reference position RMSE",f"{scores['position_rmse_m']:.2f} m")
                rr2.metric("Reference heading RMSE",f"{scores['heading_rmse_deg']:.2f}°")
                rr3.metric("Reference bank RMSE",f"{scores['bank_rmse_deg']:.2f}°")
                st.caption("PROVENANCE NOT VERIFIED. Inspect time, origin, vehicle configuration and data source before accepting results as validation.")
                st.download_button("Export matched-reference residuals",residuals.to_csv(index=False),"reference_comparison.csv","text/csv")
            except (ValueError,KeyError) as exc:
                st.error(f"Reference data rejected: {exc}")
    if isinstance(c,SixDOFConfig):
        with st.expander("v5 / Rascal source audit + held-out aerodynamic identification", expanded=False):
            st.info("v5 uses independently supplied aircraft XML and 6-DOF truth telemetry. The live v4 simulator remains an uncalibrated surrogate. Synthetic training or a matching hash does NOT establish independent validation.")
            xml_file=st.file_uploader("Original JSBSim aircraft XML", type=["xml"], key="v5_aircraft_xml")
            truth_file=st.file_uploader("Full-state reference trial telemetry (calibration + holdout CSV)", type=["csv"], key="v5_calibration_truth")
            campaign_file=st.file_uploader("Signed-off campaign manifest JSON (operator declarations)",type=["json"],key="v5_campaign_manifest")
            from identification_v5 import COLUMNS as V5_TRUTH_COLUMNS
            st.download_button("Download reference truth CSV HEADER",",".join(V5_TRUTH_COLUMNS)+"\n","v5_fullstate_truth_schema.csv","text/csv")
            if xml_file is not None:
                try:
                    facts=parse_jsbsim_aircraft_xml(xml_file.getvalue())
                    st.caption("Source XML hash checked locally. Extracted geometry/inertias are not a complete aircraft or validated airframe.")
                    st.dataframe(pd.DataFrame([{'Source quantity':k,'Value':v} for k,v in facts.items()]),hide_index=True,use_container_width=True)
                    st.download_button("Export source fact sheet",json.dumps(facts,indent=2),"v5_jsbsim_source_facts.json","application/json")
                except (ValueError,KeyError) as exc:
                    st.error(f"Source XML rejected: {exc}")
            if xml_file is not None and truth_file is not None:
                try:
                    template=template_campaign(xml_file.getvalue(),truth_file.getvalue())
                    st.download_button("Create provenance manifest TEMPLATE",json.dumps(template,indent=2),"v5_campaign_TEMPLATE.json","application/json")
                except ValueError as exc:
                    st.warning(f"Cannot create manifest template: {exc}")
            if xml_file is not None and truth_file is not None and campaign_file is not None:
                if st.button("IDENTIFY COEFFICIENTS + RUN HELD-OUT OPEN-LOOP REPLAY",key="run_v5_identification"):
                    try:
                        manifest=json.loads(campaign_file.getvalue().decode('utf-8'))
                        provenance,ref_air,trial_data=check_campaign(manifest,xml_file.getvalue(),truth_file.getvalue())
                        model=identify_and_validate(trial_data,ref_air)
                        model['input_sha256']=__import__('hashlib').sha256(truth_file.getvalue()).hexdigest()
                        replay=replay_holdout_trials(trial_data,model,ref_air,
                                                     wind_e_mps=provenance['wind_e_mps'],
                                                     wind_n_mps=provenance['wind_n_mps'])
                        report={'provenance':provenance,'identification':model,'open_loop_holdout':replay}
                        st.session_state['v5_report']=report
                    except (ValueError,TypeError,KeyError,RuntimeError,json.JSONDecodeError) as exc:
                        st.error(f"v5 campaign rejected: {exc}")
                        st.session_state.pop('v5_report',None)
                if 'v5_report' in st.session_state:
                    report=st.session_state['v5_report']
                    st.success("Calibration and open-loop holdout computation complete. Reference provenance is not independently authenticated.")
                    st.dataframe(pd.DataFrame.from_dict(report['open_loop_holdout']['trials'],orient='index'),use_container_width=True)
                    st.download_button("Download full v5 identification and holdout report",json.dumps(report,indent=2),"v5_identification_report.json","application/json")
                    st.warning("No automatic 9/10 grade: airframe identification, matched conditions, true source independence and holdout thresholds must be audited. Actuator dynamics remain untested in this replay.")
    st.subheader("Export experiment")
    if isinstance(c,SixDOFConfig) and 'control_schedule_20hz' in df.attrs:
        st.download_button("Download 20 Hz commanded-control schedule for external model replay",
                           df.attrs['control_schedule_20hz'].to_csv(index=False),
                           'v4_commanded_controls_20hz.csv','text/csv')
        st.caption("Physical surface deflection signs and normalization must be mapped to each JSBSim model. This is an export, not a completed matched JSBSim run.")
    exp = {"configuration": c.__dict__, "summary": metrics,
           "requirements": checks.to_dict(orient="records"),
           "limitations": ("v4: nonlinear 6-DOF surrogate using ASSUMED mass/inertia/aerodynamic coefficients; 3-axis IMU; complementary AHRS and 6-state 3D navigation KF. No Rascal 110 identification, full 15-state INS, external SITL, or flight validation"
                           if isinstance(c,SixDOFConfig) else
                           "v3: fixed-wing bank-to-turn reduced-order aircraft, multi-rate synthetic sensors, 6-state EKF, optional WMM2025; no aircraft identification, 6-DOF aerodynamics, or independent flight validation"
                           if isinstance(c,AerospaceConfig) else
                           "v2 legacy planar point-mass yaw-only simulator; no aircraft identification or independent validation") }
    st.download_button("Download all telemetry (CSV)", df.to_csv(index=False), "uav_compass_navigation_telemetry.csv", "text/csv")
    st.download_button("Download scenario and metrics (JSON)", json.dumps(exp, indent=2), "uav_compass_scenario.json", "application/json")
    with st.expander("Model equations, assumptions and fidelity limits"):
        st.markdown(r"""
**v4 reference frame:** Body FRD and local NED rigid-body equations, heading clockwise from true North, WGS84 model optional via WMM2025. Includes modeled roll, pitch, yaw, body translational/angular velocity, aerodynamic forces/moments, actuators, airspeed and altitude. Plant aerodynamic coefficients, inertia and mass are assumed, not identified from a Rascal 110 or hardware data. The plant is not a validated aircraft. v3/v2 retain simplified motion models.

**Kinematics and yaw:** $\dot E=V_a\sin\psi+w_E$, $\dot N=V_a\cos\psi+w_N$. The rate actuator uses an *exact discrete solution* of a first-order yaw-rate lag and midpoint position integration. This improves timestep robustness, but does not add bank angle, pitch, 6-DOF aerodynamics or real actuator identification.

**Coupled estimation:** The six-state EKF estimates $[E,N,\psi,b_g,w_E,w_N]$ using rate-gyro propagation, magnetometer heading corrections and GNSS position updates. It includes cross-state covariance terms, innovation gating and a Joseph covariance update. Gyro bias and wind are estimated under simplified observability and prior assumptions, and are not verified against actual hardware.

**Magnetic integrity:** Corrections require the proper 3-axis field magnitude, a heading innovation gate, a statistical 1D NIS gate, and, during low-turn motion with a valid long GNSS baseline, agreement with estimated GNSS-derived air heading. Course cross-checks can be unreliable with unmodeled winds, turns, or GNSS attacks. No single simulated detector guarantees fault isolation.

**Uncertainty:** A 2D model-based 95% radial bound and position NEES are reported alongside observed synthetic coverage. Correlated time points and the fact that the truth model is also synthetic prevent these checks from constituting independent calibration or model validation.

**Recovery:** A prolonged observable GNSS signal gap permits a guarded position-state reset only after multiple kinematically plausible returned fixes. The reset is separately logged and can still be deceived by coordinated spoofing after an outage. Conditional RTB requires recent accepted GNSS, low covariance and a recovery holdoff. The terminal point-mass stop is not an aircraft landing, and the controller is not safe for a real vehicle.

**v3 aircraft-aware profile:** Fixed-wing bank-to-turn coordinates $\dot\psi=g\tan\phi/V$, with lagged bank angle, a bank cap, fixed airspeed, and crosswind. It simulates 100 Hz gyro/EKF prediction, 10 Hz tilt-compensated magnetometer updates, 5 Hz GNSS, and 20 Hz closed-loop guidance. Display timestep is downsampling only. WMM2025 is available through `pygeomag` with user-provided geodetic location/date. A separate ellipsoid fit estimates constant hard/soft-iron calibration parameters from an independent simulated 3D orientation sweep. These changes do NOT create 6-DOF dynamics, full IMU strapdown physics or real calibration.

**v4 model and estimator:** v4 uses a physically distinct 6-DOF nonlinear vehicle plant, synthetic three-axis body gyroscope and accelerometer (including gravity removed in specific force), complementary attitude filter, and 6-state 3-D position/velocity Kalman filter. It is NOT a full 15-state tightly coupled INS/EKF. The vertical autopilot uses estimated altitude/vertical velocity, the speed controller uses a synthetic noisy pitot measurement. Some bias/noise and airframe coefficients remain assumed. Angle-of-attack lift is statically clipped; no validated stall departure, ground effect, landing, propulsion dynamics, wind gust process, or real aircraft flight envelope.

**v2 comparison profile:** Preserves the historical planar yaw-rate model, simplified 3-axis magnetic observation, and all original plots. See documentation for differences.

**Scope:** A reproducible research simulator for design trades and sensor-fault demonstration. v4 external truth import requires a declared source manifest and matching hash but does not authenticate the generator. No aircraft-identified aerodynamics, verified HIL/SITL comparison, calibrated flight-log datasets, flight-safety assurance, or operational autonomy. Requirements checks do not imply airworthiness certification.
""")

with sources_tab:
    panel_header("06 / CONFIGURATION BASELINE", "Rascal 110 Source Evidence",
                 "Manufacturer-published configuration versus third-party JSBSim model facts and the existing 6-DOF surrogate.")
    st.info("SOURCE-LED ENGINEERING BASELINE. This tab audits authoritative specifications. It does not recalculate the simulator with unvalidated hybrid airframe data, and does not award a new fidelity grade.")
    oem=manufacturer_si()
    k1,k2,k3,k4=st.columns(4)
    k1.metric("SIG wingspan", f"{oem['span_m']:.3f} m")
    k2.metric("SIG wing area", f"{oem['wing_area_m2']:.3f} m²")
    k3.metric("SIG flying mass", f"{oem['flying_mass_min_kg']:.2f}–{oem['flying_mass_max_kg']:.2f} kg")
    k4.metric("SIG length", f"{oem['fuselage_length_m']:.3f} m")
    st.caption("Source: SIG Manufacturing's current Rascal 110 EG ARF specifications. The historical Rascal XML may represent a different aircraft revision or loading configuration.")
    st.markdown(f"**Manufacturer:** [SIG Rascal 110 EG ARF]({SIG_URL}) · **Historical simulator:** [PX4/HIL Rascal XML]({PX4_ARCHIVE_URL}) · [JSBSim architecture]({JSBSIM_DOC_URL})")
    st.subheader("Documented parameter mismatch: current research surrogate")
    audit=pd.DataFrame(compare_surrogate_to_manufacturer())
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.warning("The v4/v5 model uses an assumed 12 kg loaded mass, exceeding SIG's published 4.99–5.90 kg flying range. The v4/v5 model remains a generic 12 kg research surrogate, not a verified SIG Rascal 110.")
    st.subheader("Source-by-source evidence ledger")
    ledger=pd.DataFrame([e.__dict__ for e in source_evidence()])
    st.dataframe(ledger, use_container_width=True, hide_index=True)
    st.subheader("Configuration and authority conflicts")
    for conflict in check_source_conflicts():
        with st.expander(f"UNRESOLVED / {conflict['issue']}"):
            st.write(f"**Manufacturer:** {conflict['manufacturer']}")
            st.write(f"**Archival aircraft model:** {conflict['archive']}")
            st.write(f"**Required treatment:** {conflict['action']}")
    xml_up=st.file_uploader("Optional: inspect a downloaded, original JSBSim Rascal XML",type=['xml'],key='v6_source_aircraft_xml')
    if xml_up is not None:
        try:
            inspection=audit_uploaded_jsbsim_xml(xml_up.getvalue())
            st.success("Aircraft XML syntactically parsed and its key model properties extracted. This does not authenticate the origin or establish physical fidelity.")
            st.dataframe(pd.DataFrame(inspection['geometry_comparison']),hide_index=True,use_container_width=True)
            st.json(inspection['xml_facts'])
            st.warning(inspection['mass_disposition'])
        except (ValueError,TypeError) as exc:
            st.error(f"Source inspection failed: {exc}")
    st.download_button("⬇ Download source-grounded evidence dossier (JSON)",
        json.dumps(dossier(),indent=2),"rascal_110_source_evidence_v6.json","application/json")
    st.caption("Recommended electrical power is in watts and must not be converted directly to a propeller thrust force. No OEM aerodynamic tables, measured inertia or propulsion thrust curves were identified in the cited primary manufacturer specifications.")

st.markdown(
    '<div class="hud-foot">UAV COMPASS VARIANCE SIMULATOR &nbsp; | &nbsp; '
    'PYTHON • STREAMLIT • NUMPY • PANDAS • MATPLOTLIB • PLOTLY &nbsp; | &nbsp; '
    'SYNTHETIC AEROSPACE RESEARCH ENVIRONMENT • NOT FOR OPERATIONAL FLIGHT USE</div>',
    unsafe_allow_html=True,
)
