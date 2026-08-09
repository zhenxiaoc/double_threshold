import numpy as np
from math import acos, sqrt, pi
# Exact two-threshold functional: lens of two disks, radii perturbed along a fixed
# direction.  Theta(gamma_0 + t*eta) = lens(sqrt(1+t*a), sqrt(1+t*c), D) =: phi(t).
D, a, c = 1.2, 1.0, 0.8
def lens(R1,R2,DD):
    if DD>=R1+R2: return 0.0
    if DD<=abs(R1-R2): return pi*min(R1,R2)**2
    return (R1*R1*acos((DD*DD+R1*R1-R2*R2)/(2*DD*R1))+R2*R2*acos((DD*DD+R2*R2-R1*R1)/(2*DD*R2))
            -0.5*sqrt((-DD+R1+R2)*(DD+R1-R2)*(DD-R1+R2)*(DD+R1+R2)))
phi  = lambda t: lens(sqrt(1+t*a), sqrt(1+t*c), D)
dphi = lambda t,h=1e-6: (phi(t+h)-phi(t-h))/(2*h)
th0  = phi(0.0)
d2   = (phi(1e-3)+phi(-1e-3)-2*th0)/1e-6

print(f"theta_0 = {th0:.10f}   D^2 Theta[eta,eta] = phi''(0) = {d2:+.6f}\n")
print("Estimators, with half-sample errors e_A = tA*eta, e_B = tB*eta:")
print("  SS      = 2*phi(tbar) - (phi(tA)+phi(tB))/2")
print("  DD(v2)  = SS - (1/2)[ tA*phi'(tA) + tB*phi'(tB) ]        <- representer at gamma^j")
print("  DD(fix) = SS - [ tA*phi'(tbar) - tA*phi'(tA)/2 ]")
print("                - [ tB*phi'(tbar) - tB*phi'(tB)/2 ]        <- composite score\n")

def cell(tA, tB):
    tb = 0.5*(tA+tB)
    SS  = 2*phi(tb) - 0.5*(phi(tA)+phi(tB))
    v2  = SS - 0.5*(tA*dphi(tA) + tB*dphi(tB))
    fix = SS - (tA*dphi(tb) - 0.5*tA*dphi(tA)) - (tB*dphi(tb) - 0.5*tB*dphi(tB))
    return SS-th0, v2-th0, fix-th0

print("(1) DETERMINISTIC COMMON BIAS  e_A = e_B = b  (the critique's counterexample):")
print(f"   {'t':>8} {'SS err':>13} {'DD(v2) err':>13} {'DD(fix) err':>13} {'-t^2 phi''/2':>13}")
for t in (0.1, 0.05, 0.025, 0.0125):
    s_,v_,f_ = cell(t,t)
    print(f"   {t:8.4f} {s_:13.3e} {v_:13.3e} {f_:13.3e} {-0.5*t*t*d2:13.3e}")
print("   -> DD(v2) error is O(t^2) and equals -t^2 phi''/2 : QUADRATIC, not cubic.\n")

print("(2) INDEPENDENT MEAN-ZERO half-sample errors, tA,tB ~ N(0,sig^2), 400000 draws:")
rng = np.random.default_rng(7)
print(f"   {'sig':>8} {'E[SS err]':>13} {'E[DD(v2) err]':>15} {'E[DD(fix) err]':>15} {'-sig^2 phi''':>13}")
for sig in (0.08, 0.04, 0.02, 0.01):
    tA = rng.normal(0, sig, 400000); tB = rng.normal(0, sig, 400000)
    tb = 0.5*(tA+tB)
    P  = np.vectorize(phi); Dp = np.vectorize(dphi)
    SS = 2*P(tb) - 0.5*(P(tA)+P(tB)) - th0
    v2 = SS - 0.5*(tA*Dp(tA) + tB*Dp(tB))
    fx = SS - (tA*Dp(tb) - 0.5*tA*Dp(tA)) - (tB*Dp(tb) - 0.5*tB*Dp(tB))
    print(f"   {sig:8.4f} {SS.mean():13.3e} {v2.mean():15.3e} {fx.mean():15.3e} {-sig*sig*d2:13.3e}")
print("   -> SS and DD(fix) are unbiased to this order; DD(v2) has bias -sig^2 phi''.")
