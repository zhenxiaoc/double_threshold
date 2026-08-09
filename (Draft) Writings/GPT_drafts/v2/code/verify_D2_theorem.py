import numpy as np
from math import acos, sqrt, pi

# ---------------------------------------------------------------
# Numerical validation of Theorem 3.1 (D2 Theta) in d=2.
# Three closed-form designs, each isolating different terms.
# ---------------------------------------------------------------

def d2_fd(f, h=1e-3):
    "central 2nd difference with Richardson"
    a = (f(h) + f(-h) - 2*f(0.0))/h**2
    b = (f(2*h) + f(-2*h) - 2*f(0.0))/(2*h)**2
    return (4*a - b)/3
def d1_fd(f, h=1e-3):
    a = (f(h) - f(-h))/(2*h)
    b = (f(2*h) - f(-2*h))/(4*h)
    return (4*a - b)/3

print("="*72)
print("DESIGN 1: disk g0 = 1-|x|^2 with RADIAL weight rho(r); u=a const, no h.")
print("  isolates the Q_g curvature/transport terms.")
rho  = lambda r: 1.0 + 0.7*r**2 - 0.3*r**3
rhop = lambda r: 1.4*r - 0.9*r**2
a = 1.3
def Theta1(t):
    R = sqrt(1.0 + t*a)
    # integral_0^R rho(r) 2 pi r dr
    from scipy.integrate import quad
    return quad(lambda r: rho(r)*2*pi*r, 0, R, epsabs=1e-13, epsrel=1e-13)[0]
num = d2_fd(Theta1)
ana = pi*a**2*rhop(1.0)/2      # derived in closed form from Q_g
print(f"  numeric  D2 = {num: .10f}")
print(f"  formula  D2 = {ana: .10f}   (Q_g margin form, rel.err {abs(num-ana)/abs(ana):.2e})")

print("="*72)
print("DESIGN 2: disk g0 = 1-|x|^2, w=1, NON-CONSTANT u(x)=|x|^2.")
print("  isolates the  u (n_g . grad u)  term of Q_g.")
def Theta2(t):
    return pi/(1.0-t)          # {1-(1-t)r^2>=0} -> radius (1-t)^{-1/2}
print(f"  numeric  D1 = {d1_fd(Theta2): .10f}   formula D1 = {pi: .10f}")
print(f"  numeric  D2 = {d2_fd(Theta2): .10f}   formula D2 = {2*pi: .10f}")

print("="*72)
print("DESIGN 3: unit disk  g0=1-|x|^2  cut by straight chord h0 = x1-kappa.")
print("  w=1, u=a, v=c constants  ->  both margin forms vanish, so this")
print("  isolates the TWO corner diagonals and the corner CROSS term.")
def seg_area(R, m):
    if m >= R:  return 0.0
    if m <= -R: return pi*R*R
    return R*R*acos(m/R) - m*sqrt(R*R-m*m)
for kappa in (0.6, 0.2, -0.5):
    for (a, c) in ((1.0,1.0), (1.7,-0.9), (0.4,2.1)):
        def Th(t, kappa=kappa, a=a, c=c):
            return seg_area(sqrt(1.0+t*a), kappa - t*c)
        num = d2_fd(Th, 1e-3)
        s = sqrt(1-kappa**2); cw = -kappa          # cos omega, |sin omega|
        Qg_c = -2*a**2*cw/(4*s)      # -int_C u^2 w cos/( |grad g|^2 |sin| ), |grad g|=2, 2 pts
        Qh_c = -2*c**2*cw/(1*s)      # |grad h| = 1
        cross = 2*2*a*c/(2*1*s)      # 2 * sum_C u v w /(|grad g||grad h||sin|)
        ana = Qg_c + Qh_c + cross
        print(f"  kappa={kappa:+.2f} a={a:+.2f} c={c:+.2f} :"
              f" numeric {num: .8f}   formula {ana: .8f}   relerr {abs(num-ana)/max(abs(ana),1e-12):.2e}")

print("="*72)
print("DESIGN 4: LENS = intersection of two unit disks, centres distance D apart.")
print("  g0=1-|x-p|^2, h0=1-|x-q|^2, w=1, u=a, v=c const.")
print("  Both surfaces curved & oblique; both corner diagonals + cross active.")
def lens(R1, R2, D):
    if D >= R1+R2: return 0.0
    if D <= abs(R1-R2): return pi*min(R1,R2)**2
    t1 = R1*R1*acos((D*D+R1*R1-R2*R2)/(2*D*R1))
    t2 = R2*R2*acos((D*D+R2*R2-R1*R1)/(2*D*R2))
    t3 = 0.5*sqrt((-D+R1+R2)*(D+R1-R2)*(D-R1+R2)*(D+R1+R2))
    return t1+t2-t3
for D in (0.8, 1.2, 1.7):
    for (a, c) in ((1.0,1.0), (1.5,-0.7), (0.3,1.9)):
        def Th(t, D=D, a=a, c=c):
            return lens(sqrt(1.0+t*a), sqrt(1.0+t*c), D)
        num = d2_fd(Th, 1e-3)
        cw = 1 - D*D/2.0
        s  = sqrt(max(1-cw*cw, 0.0))
        Qg_c = -2*a**2*cw/(4*s)
        Qh_c = -2*c**2*cw/(4*s)
        cross = 2*2*a*c/(2*2*s)
        ana = Qg_c+Qh_c+cross
        print(f"  D={D:.2f} a={a:+.2f} c={c:+.2f} :"
              f" numeric {num: .8f}   formula {ana: .8f}   relerr {abs(num-ana)/max(abs(ana),1e-12):.2e}")

print("="*72)
print("DESIGN 4b: ONE-SIDED perturbation v=0 -- cross term is identically zero,")
print("  so any nonzero curvature is the Q_g corner diagonal alone.")
for D in (0.8, 1.2, 1.7):
    for (a, c) in ((1.0, 0.0), (0.0, 1.0), (1.4, 0.0)):
        def Th(t, D=D, a=a, c=c):
            return lens(sqrt(1.0+t*a), sqrt(1.0+t*c), D)
        num = d2_fd(Th, 1e-3)
        cw = 1 - D*D/2.0; s = sqrt(max(1-cw*cw, 0.0))
        ana = -2*a**2*cw/(4*s) - 2*c**2*cw/(4*s) + 2*2*a*c/(2*2*s)
        print(f"  D={D:.2f} (a,c)=({a},{c}) cos(w)={cw:+.3f} :"
              f" numeric {num:+.9f}  formula {ana:+.9f}")

print("="*72)
print("DESIGN 5: SUPPORT EDGE.  Unit disk g0=1-|x|^2, FIXED support X={x1>=kappa},")
print("  w=1, u=a, no second threshold.  Claim (Remark 'edge'): the edge adds")
print("  ONLY the u^2 diagonal  -int u^2 w cos(w_bdry)/(|grad g|^2 |sin w_bdry|).")
for kappa in (0.6, 0.2, 0.0, -0.5):
    for a in (1.0, 1.7):
        num = d2_fd(lambda t, kappa=kappa, a=a: seg_area(sqrt(1.0+t*a), kappa), 1e-3)
        s = sqrt(1-kappa**2); cw = -kappa
        ana = -2*a**2*cw/(4*s)
        print(f"  kappa={kappa:+.2f} a={a:.1f}: numeric {num:+.9f}   formula {ana:+.9f}")
