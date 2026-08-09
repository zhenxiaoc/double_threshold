from math import acos, sqrt, pi
def d2(f,h=1e-3):
    a=(f(h)+f(-h)-2*f(0.0))/h**2; b=(f(2*h)+f(-2*h)-2*f(0.0))/(2*h)**2; return (4*a-b)/3
def lens(R1,R2,D):
    if D>=R1+R2: return 0.0
    if D<=abs(R1-R2): return pi*min(R1,R2)**2
    return (R1*R1*acos((D*D+R1*R1-R2*R2)/(2*D*R1))+R2*R2*acos((D*D+R2*R2-R1*R1)/(2*D*R2))
            -0.5*sqrt((-D+R1+R2)*(D+R1-R2)*(D-R1+R2)*(D+R1+R2)))
def seg(R,m):
    if m>=R: return 0.0
    if m<=-R: return pi*R*R
    return R*R*acos(m/R)-m*sqrt(R*R-m*m)

print("ATTACK: does  q_full - q_g - q_h  isolate the CROSS corner term only,")
print("or does it also pick up the two cos(omega) diagonal corner terms?")
print("(v1 said cross-only; my v2 edit said it also captures the cos(w) diagonals.)\n")
print(f"{'design':>22} {'qfull':>11} {'qg':>11} {'qh':>11} {'qf-qg-qh':>11} {'2C (cross)':>11} {'Qg+Qh corner':>13}")
for D in (0.8, 1.2, 1.7):
    for (a,c) in ((1.0,1.0),(1.5,-0.7),(0.3,1.9)):
        F  = d2(lambda t: lens(sqrt(1+t*a), sqrt(1+t*c), D))
        Gg = d2(lambda t: lens(sqrt(1+t*a), 1.0,          D))
        Hh = d2(lambda t: lens(1.0,          sqrt(1+t*c), D))
        cw = 1-D*D/2; s = sqrt(1-cw*cw)
        cross  = 2*2*a*c/(2*2*s)                       # 2 * sum_C uvw/(|gg||gh||sin|)
        corner_diag = -2*a**2*cw/(4*s) - 2*c**2*cw/(4*s)
        print(f"  lens D={D:.1f} (a,c)=({a:+.1f},{c:+.1f}) {F:11.6f} {Gg:11.6f} {Hh:11.6f}"
              f" {F-Gg-Hh:11.6f} {cross:11.6f} {corner_diag:13.6f}")
for kappa in (0.6,-0.5):
    for (a,c) in ((1.0,1.0),(1.7,-0.9)):
        F  = d2(lambda t: seg(sqrt(1+t*a), kappa-t*c))
        Gg = d2(lambda t: seg(sqrt(1+t*a), kappa))
        Hh = d2(lambda t: seg(1.0,         kappa-t*c))
        s = sqrt(1-kappa**2); cw = -kappa
        cross = 2*2*a*c/(2*1*s)
        corner_diag = -2*a**2*cw/(4*s) - 2*c**2*cw/(1*s)
        print(f"  chord k={kappa:+.1f} (a,c)=({a:+.1f},{c:+.1f}) {F:11.6f} {Gg:11.6f} {Hh:11.6f}"
              f" {F-Gg-Hh:11.6f} {cross:11.6f} {corner_diag:13.6f}")
