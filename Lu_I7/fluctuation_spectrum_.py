import numpy as np
import scipy as sp
from qutip import *
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_6j, wigner_3j
from tqdm import tqdm
#import matplotlib.pyplot as plt
from scipy.linalg import inv
from numpy import linalg as LA
from scipy.optimize import minimize_scalar
import math
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigs

#Lu's quantum numbers
I = 7
S = 1
Je = 0
Jg = 1
Lg = 2
F = [i for i in range(int(I-Jg),int(I + Jg + 1),1)]
F.insert(0,int(I+Je))


#number of atomic states

N = [2*F[i] + 1 for i in range(len(F))]
N = sum(N)


#Zeeman stuff


muB = 9.3e-24 #Bohr magneton J T^-1
hbar = 1.054e-34 # J s

muB = muB/hbar #hbar = 1
muB = muB/(2*np.pi)
muB = muB/1e6  # so it's in 1/2π (MHz T^-1)

# 1Gauss is 0.0001 T
B = 4e-4
print(muB*B) 



#Lande g factor for F
me = 9.1093837015e-31 
mp = 1.67262192369e-27 

mI = 3.169 #nuclear magnetic moment of 176Lu in units of μN
gI = mI/I

gJg = 1 + (Jg*(Jg +1) + S*(S + 1) - Lg*(Lg + 1))/(2*Jg*(Jg + 1))
print(gJg)

#gF = [1,0,1,2]
gF=[]
gF.append(-gI*me/mp)

for i in range(3):
    gF.append(gJg*(F[i +1]*(F[i+1] + 1) + Jg*(Jg+1) - I*(I+1))/(2*F[i+1]*(F[i +1] + 1)))
    
def base(F,i,mF):
    #numbering states from 0 to N-1 starting from -mF to mF
    # 0 is |F',-mF'>
    if i==0:
        b = basis(N,mF+F[i])
    elif i==1:
        b = basis(N,mF + F[i] + 2*F[0] + 1)
    elif i==2:
        b = basis(N,mF + F[i] + 2*F[0] + 1 + 2*F[1] + 1)
    else:
        b = basis(N,mF + F[i] + 2*F[0] + 1 + 2*F[1] + 1 + 2*F[2] +1)
    return b   

#Dissipation

GammaJgJe = 2.44745 #1/2π (MHz) 3D1 to 3P0 

def GammaFgFe(F,ig,Je,Jg,I,GammaJgJe):
    return float((2*Je + 1)*(2*F[0] + 1)*wigner_6j(Je,F[0],I,F[ig],Jg,1)**2)*GammaJgJe
    #return GammaJgJe

def cg(F,ig,mFg,ie,mFe,q):
    return float(CG(F[ig],mFg,1,q,F[ie],mFe).doit())
    #return 1

c_ops_ = []
qs = [-1,0,1]

for ig in range(1,len(F)):
    for mfg in range(-F[ig],F[ig] + 1):
        for mfe in range(-F[0], F[0] + 1):
            for q in qs:
                if cg(F,ig,mfg,0,mfe,q) != 0:
                    cops =np.sqrt(1/(2*F[0] + 1))*cg(F,ig,mfg,0,mfe,q)*np.sqrt(GammaFgFe(F,ig,Je,Jg,I,GammaJgJe*2*np.pi))*base(F,ig,mfg)*base(F,0,mfe).dag()
                    c_ops_.append(cops)
                else:
                    continue

#Hamiltonians

def H_I_(F,Omega_p):
    HI=0*basis(N,0)*basis(N,0).dag()
    for mFe in range(-F[0],F[0] + 1):
        for ig in range(1,len(F)):
            for mFg in range(-F[ig],F[ig]+1):
                for q in range(-1,2):
                    HI += 2*np.pi*cg(F,ig,mFg,0,mFe,q)*Omega_p[ig-1,q+1]/2*base(F,0,mFe)*base(F,ig,mFg).dag()
                
    return -1/np.sqrt(2*F[0]+1)*(HI + HI.dag()) 

def H_0_(F,Delta):
    H0 = 0*basis(N,0)*basis(N,0).dag()
    for l in range(len(F)):
        for mF in range(-F[l],F[l]+1):
            H0 += 2*np.pi*(Delta[l] + gF[l]*muB*mF*B)*base(F,l,mF)*base(F,l,mF).dag() 
    return H0

def H_1(F,Omega_p):
    HI=0*basis(N,0)*basis(N,0).dag()
    for mFe in range(-F[0],F[0] + 1):
        for ig in range(1,len(F)):
            for mFg in range(-F[ig],F[ig]+1):
                for q in range(-1,2):
                    HI += -1j*eta[ig-1]*2*np.pi*cg(F,ig,mFg,0,mFe,q)*Omega_p[ig-1,q+1]/2*base(F,0,mFe)*base(F,ig,mFg).dag()
                
    return 1/np.sqrt(2*F[0]+1)*(HI + HI.dag()) 


def S_(F,Delta,Omega_p,omega0,c): #fluctuation spectrum
    H0_ = H_0_(F,Delta)
    HI_ = H_I_(F,Omega_p) 
    

    H0 = H0_ + HI_ 
    

    L0 = 0*spre(c[0])*spost(c[0].dag())
    for i in range(len(c)):
        L0 += spre(c[i])*spost(c[i].dag()) - 0.5*(spre(c[i].dag()*c[i]) + spost(c[i].dag()*c[i]))

    L0 += -1j*(spre(H0) - spost(H0))
    
    #L0 = np.array(L0)
    L0 = L0.data_as('ndarray')

    #H0 = H0.to("CSR").tidyup(atol=1e-8)

    rho = steadystate(H0,c) #!

    V1 = H_1(F,Omega_p)
    V1rho = V1*rho 
    
    V1rho = operator_to_vector(V1rho)
    #V1rho = np.array(V1rho)
    V1rho = V1rho.data_as('ndarray')

    #eye = np.array(operator_to_vector(tensor(qeye(N),qeye(Nmotion))))
    #eye = np.eye(N**2,dtype=np.complex_)
    inve = -inv((L0+1j*2*np.pi*omega0*np.eye(N**2)))

    s = inve@V1rho
    #s = -spsolve(L0+1j*2*np.pi*omega0*eye,V1rho)

    S = np.zeros((N,N),dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            #S[j,i] = s[N*i+j][0]
            S[j,i] = s[N*i+j]
            
    #V1 = np.array(V1)
    V1 =V1.data_as('ndarray')
    S = np.matmul(V1,S)

    return np.matrix.trace(S)

#system parameters
c = 299792458
λ = 646e-9
f = c/λ 
f_1 = 11.2e9
f_2 = 10.5e9

#176Lu+
amu = 1.66053886e-27
hbar = 1.054571817e-34
M = 176*amu

omega0  = 1.4

eta = []

eta2 = 2*np.pi*f/c*np.sqrt(hbar/(2*M*2*np.pi*omega0*10**(6)))
eta.append(eta2)

eta3 = 2*np.pi*(f-f_1)/c*np.sqrt(hbar/(2*M*2*np.pi*omega0*10**(6)))
eta.append(eta3)

eta4 = -2*np.pi*(f-f_2)/c*np.sqrt(hbar/(2*M*2*np.pi*omega0*10**(6)))
eta.append(eta4)

#eta = [0.01,0.01,-0.01] 

#Rabi frequencies Ω_F,F' #1/2π (MHz)
omega = [70,19,70]

#turn on/off polarization
Omega_p= np.zeros((3,3)) # (ig-1, q + 1) transition from F state to F' with polarization q

Omega_p[0,0] = Omega_p[0,2] = omega[0]       #σ+ and σ- for F=F[1] to F'= F[0]
Omega_p[1,0] = Omega_p[1,2] = omega[1]      #σ+ and σ- for F=F[2] to F'=F[0]
Omega_p[2,0] = Omega_p[2,2] = omega[2]     #σ+ and σ- for F=F[3] to F'=F[0]]

B = 4e-04
Delta = [0,10,22.1,20]

omega0=1.4

Ss = []
for v in tqdm(np.linspace(-3,3,2000)):
    Ss.append(S_(F=F,Delta=Delta,Omega_p=Omega_p,omega0=v,c=c_ops_))

with open(""+str(path_name)+"data_B4_D21_10_D23_20_O21_O23_"+str(omega[0])+"_O22_"+str(omega[1])+"_omega_1.4.txt",'wb') as f:
    np.savetxt(f, np.transpose([np.linspace(-3,3,2000),Ss]),fmt='%.8f')

