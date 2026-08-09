"""v4 check: the polarization identity is exact WITHIN each family and only
fourth-order accurate ACROSS families (v3 mixed the two).

Design: lens of two unit disks, centres D apart.  gamma_0 = (g_0,h_0),
half-sample errors e_A = tA*eta, e_B = tB*eta with eta = (a,c) constant.
Theta along any straight direction is a closed-form lens area.
"""
from math import acos, sqrt, pi
D, a, c = 1.2, 1.0, 0.8

def lens(R1, R2, DD):
    if DD >= R1+R2: return 0.0
    if DD <= abs(R1-R2): return pi*min(R1, R2)**2
    return (R1*R1*acos((DD*DD+R1*R1-R2*R2)/(2*DD*R1))
            + R2*R2*acos((DD*DD+R2*R2-R1*R1)/(2*DD*R2))
            - 0.5*sqrt((-DD+R1+R2)*(DD+R1-R2)*(DD-R1+R2)*(DD+R1+R2)))
# Theta at (g,h) displaced by (sg, sh) in the two normal directions
Th = lambda sg, sh: lens(sqrt(1+sg), sqrt(1+sh), D)

def d2(f, h=1e-4):
    return (f(h)+f(-h)-2*f(0.0))/h**2
def d4(f, h=1e-2):
    return (f(2*h)-4*f(h)+6*f(0.0)-4*f(-h)+f(-2*h))/h**4

print(f"{'tA':>7} {'tB':>7} | {'ideal gap':>13} {'jack gap':>13} | "
      f"{'jack-ideal':>13} {'-phi4/384':>13}")
for tA, tB in ((0.40, 0.16), (0.20, 0.08), (0.10, 0.04), (0.05, 0.02)):
    gA, hA = tA*a, tA*c
    gB, hB = tB*a, tB*c
    gb, hb = 0.5*(gA+gB), 0.5*(hA+hB)
    dg, dh = gA-gB, hA-hB
    Tb = Th(gb, hb)
    # --- ideal family: exact second derivatives at gamma-bar ---
    Qg  = d2(lambda t: Th(gb+t*dg, hb))              # D^2 in the (g,0) direction
    Qh  = d2(lambda t: Th(gb, hb+t*dh))              # D^2 in the (0,h) direction
    Qfl = d2(lambda t: Th(gb+t*dg, hb+t*dh))         # D^2 in the joint direction
    twoC = Qfl - Qg - Qh                             # = 2 C[dg,dh]
    ideal_SS  = Tb - Qfl/8.0
    ideal_mar = Tb - (Qg+Qh)/8.0
    ideal_cor = twoC/8.0                             # = (1/4) C
    # --- jackknife family: second differences at step 1/2 ---
    qf = 4*(Th(gA,hA) + Th(gB,hB) - 2*Tb)
    qg = 4*(Th(gA,hb) + Th(gB,hb) - 2*Tb)
    qh = 4*(Th(gb,hA) + Th(gb,hB) - 2*Tb)
    jack_SS  = 2*Tb - 0.5*(Th(gA,hA)+Th(gB,hB))
    jack_mar = Tb - (qg+qh)/8.0
    jack_cor = (qf-qg-qh)/8.0
    phi4 = d4(lambda t: Th(gb+t*dg, hb+t*dh))
    print(f"{tA:7.4f} {tB:7.4f} | {ideal_mar-ideal_cor-ideal_SS:13.2e} "
          f"{jack_mar-jack_cor-jack_SS:13.2e} | "
          f"{jack_SS-ideal_SS:13.3e} {-phi4/384:13.3e}"
          f"  ratio {(jack_SS-ideal_SS)/(-phi4/384):6.2f}")
print("\ncols 3-4: within-family identities (should be ~0 to machine precision)")
print("cols 5-6: the CROSS-family gap, which v3 asserted was zero;")
print("          it equals -phi^{(4)}/384 and scales like ||Delta||^4.")
