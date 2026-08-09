import numpy as np
# ---- clean setting: PERIODIC uniform cubic B-splines on the torus ----------
# removes the clamped-boundary artifacts of the previous probe.
def cardB3(u):           # cardinal cubic B-spline, support [0,4]
    u=np.asarray(u,float); out=np.zeros_like(u)
    m=(u>=0)&(u<1); out[m]=u[m]**3/6
    m=(u>=1)&(u<2); t=u[m]-1; out[m]=(1+3*t+3*t**2-3*t**3)/6
    m=(u>=2)&(u<3); t=u[m]-2; out[m]=(4-6*t**2+3*t**3)/6
    m=(u>=3)&(u<4); t=u[m]-3; out[m]=(1-t)**3/6
    return out
def cardB3p(u):          # derivative
    u=np.asarray(u,float); out=np.zeros_like(u)
    m=(u>=0)&(u<1); out[m]=u[m]**2/2
    m=(u>=1)&(u<2); t=u[m]-1; out[m]=(3+6*t-9*t**2)/6
    m=(u>=2)&(u<3); t=u[m]-2; out[m]=(-12*t+9*t**2)/6
    m=(u>=3)&(u<4); t=u[m]-3; out[m]=-3*(1-t)**2/6
    return out
def V1_periodic(x, m):
    """V1(x)=b'G^{-1}b for periodic uniform cubic B-splines with m cells,
    computed in Fourier (G is circulant).  Returns V1 and V1'."""
    x=np.atleast_1d(x)%1.0
    k=np.arange(m)
    # b_k(x) = sqrt(m) * B3(m*x - k + 2  mod m)   (normalised: int b_k^2 = const)
    U=(m*x[:,None]-k[None,:]+2.0)%m
    B=np.sqrt(m)*cardB3(U); Bp=m*np.sqrt(m)*cardB3p(U)
    # circulant Gram: G = F diag(ghat) F^*  ;  first row from autocorrelation
    g0=np.zeros(m)
    xg=np.linspace(0,1,20001)[:-1]; Bg=np.sqrt(m)*cardB3((m*xg[:,None]-k[None,:]+2.0)%m)
    G=(Bg.T@Bg)/len(xg)
    ghat=np.real(np.fft.fft(G[0]))
    Fb=np.fft.fft(B,axis=1); Fbp=np.fft.fft(Bp,axis=1)
    V =np.real(np.sum(np.conj(Fb )*Fb /ghat,axis=1))/m
    Vp=np.real(np.sum(np.conj(Fbp)*Fb /ghat,axis=1))*2/m
    return V, Vp

print("Diagnostic (periodic, no boundary artifacts): does V1 ~ K1 and V1' ~ K1^2 ?")
for m in (8,16,32,64):
    x=np.linspace(0,1,20001)[:-1]; V,Vp=V1_periodic(x,m)
    print(f"  m=K1={m:3d}:  mean V1={V.mean():8.2f} (=K1? {m})   max|V1'|={np.abs(Vp).max():10.1f}"
          f"   max|V1'|/K1^2={np.abs(Vp).max()/m**2:7.3f}")

print("\nI_K = INT_M psi * (n.grad V) dH^1 along the line x2 = a*x1 + b on the torus.")
print("  K1^3 = K^{1+1/d} (no cancellation) ;  K1^2 = K (full cancellation)")
print(f"\n{'alpha':>10} {'type':>16} | " + " ".join(f"{'m='+str(m):>10}" for m in (16,32,64,128)) + "   slope")
for alpha,typ in ((0.0,'0 (aligned)'),(1.0,'1 (resonant)'),(0.5,'1/2 (rational)'),
                  (1.6,'8/5 (rational)'),(1/np.pi,'1/pi (irrat.)'),
                  (np.sqrt(5)-1,'sqrt5-1 (irrat.)'),(np.sqrt(2),'sqrt2 (irrat.)')):
    vals=[];ms=[]
    for m in (16,32,64,128):
        s1=np.linspace(0,1,400001)[:-1]; s2=(alpha*s1+0.137)%1.0
        V1,dV1=V1_periodic(s1,m); V2,dV2=V1_periodic(s2,m)
        nr=np.sqrt(1+alpha**2); psi=1.0+0.3*np.cos(2*np.pi*s1)-0.2*np.sin(2*np.pi*s2)
        dnV=(-alpha*dV1*V2+V1*dV2)/nr
        I=np.trapezoid(psi*dnV,s1)*nr
        vals.append(abs(I)/m**2); ms.append(m)
    sl=np.polyfit(np.log(ms),np.log([v*mm*mm for v,mm in zip(vals,ms)]),1)[0]
    print(f"{alpha:10.5f} {typ:>16} | " + " ".join(f"{v:10.2f}" for v in vals) + f"   {sl:5.2f}")
print("\n(entries = |I_K|/K1^2 ; flat => I_K ~ K ; linear growth => I_K ~ K^{1.5})")
