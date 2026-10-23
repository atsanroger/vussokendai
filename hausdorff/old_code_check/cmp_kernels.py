import sys, time, warnings
warnings.filterwarnings("ignore")
sys.argv = ['AllClass', '--Fit', '--EO', 'None', '--T', '8', '--Q0', '0.05']
import AllClass_lib as A, gvar as gv, numpy as np
from AllClass_lib import Prefactor, SEW, totaldecaywidth

def parts(ens, T, t0=0.5):
    V = ens.L**3
    d = {i: ens.jacknifed_correlation(ND=1, GammaMatrix=str(i), EO='None') for i in ens.IMP}
    data = {'AT': d[54]/V, 'VT': d[4]/V,
            'AS': (d[51]+d[52]+d[53])/3/V, 'VS': (d[1]+d[2]+d[3])/3/V}
    return data

def spec(ens, S, T, data):
    return ens.spectral(S, T, 0.5, Ext_data=data, Use_Fit=True)

def vus_of(ens, sumv):
    return np.sqrt((sumv*Prefactor*SEW*ens.Inva**3/(totaldecaywidth*ens.ZA**2))**(-1))

schemes = {'orig (A:1,V:0)': dict(AT=1, VT=0, AS=1, VS=0),
           'phys (T:0,S:1)': dict(AT=0, VT=0, AS=1, VS=1)}
for name, ens in (('48I', A.FourEight), ('64I', A.SixFour)):
    for T in (2,3,4,5,6,7,8):
        try:
            data = parts(ens, T)
            row = f"{name} T={T}:"
            for sn, kmap in schemes.items():
                s = sum(spec(ens, kmap[k], T, data[k]) for k in ('AT','VT','AS','VS'))
                row += f"  {sn}: Vus={vus_of(ens, s)}"
            print(row)
        except Exception as e:
            print(f"{name} T={T}: ERROR {type(e).__name__}: {str(e)[:120]}")
