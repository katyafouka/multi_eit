import numpy as np
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
#print(muB*B) 



#Lande g factor for F
me = 9.1093837015e-31 
mp = 1.67262192369e-27 

mI = 3.169 #nuclear magnetic moment of 176Lu in units of μN
gI = mI/I

gJg = 1 + (Jg*(Jg +1) + S*(S + 1) - Lg*(Lg + 1))/(2*Jg*(Jg + 1))
#print(gJg)

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
#GammaJgJe = 2.5

def GammaFgFe(F,ig,Je,Jg,I,GammaJgJe):
    return float((2*Je + 1)*(2*F[0] + 1)*wigner_6j(Je,F[0],I,F[ig],Jg,1)**2)*GammaJgJe
    #return GammaJgJe

def cg(F,ig,mFg,ie,mFe,q):
    return float(CG(F[ig],mFg,1,q,F[ie],mFe).doit())

#single collapse operator for each transtion
c_ops = []
qs = [-1,0,1]

for ig in range(1,len(F)):
    for mfg in range(-F[ig],F[ig] + 1):
        for mfe in range(-F[0], F[0] + 1):
            for q in qs:
                if cg(F,ig,mfg,0,mfe,q) != 0:
                    cops =np.sqrt(1/(2*F[0] + 1))*cg(F,ig,mfg,0,mfe,q)*np.sqrt(GammaFgFe(F,ig,Je,Jg,I,GammaJgJe))*base(F,ig,mfg)*base(F,0,mfe).dag()
                    c_ops.append(cops)
                else:
                    continue

#Hamiltonians


def H1_(F,eta,Omega_p):
    H1=0*basis(N,0)*basis(N,0).dag()
    for mFe in range(-F[0],F[0] + 1):
        for ig in range(1,len(F)):
            for mFg in range(-F[ig],F[ig]+1):
                for q in range(-1,2):
                    H1 += eta[ig-1]*cg(F,ig,mFg,0,mFe,q)*Omega_p[ig-1,q+1]/2*base(F,0,mFe)*base(F,ig,mFg).dag()
                
    return -1/np.sqrt(2*F[0]+1)*H1

def H_I(F,Omega_p):
    HI=0*basis(N,0)*basis(N,0).dag()
    for mFe in range(-F[0],F[0] + 1):
        for ig in range(1,len(F)):
            for mFg in range(-F[ig],F[ig]+1):
                for q in range(-1,2):
                    HI += cg(F,ig,mFg,0,mFe,q)*Omega_p[ig-1,q+1]/2*base(F,0,mFe)*base(F,ig,mFg).dag()
                
    return -1/np.sqrt(2*F[0]+1)*(HI + HI.dag()) 

def H_0(F,Delta):
    H0 = 0*basis(N,0)*basis(N,0).dag()
    for l in range(len(F)):
        for mF in range(-F[l],F[l]+1):
            H0 += (Delta[l] + gF[l]*muB*mF*B)*base(F,l,mF)*base(F,l,mF).dag() 
    return H0

def steck(F,N,Delta,Omega_p,eta,c,Dmin, Dmax,nn): #fictitious lasers business
    H0_ = H_0(F,Delta)
    HI_ = H_I(F,Omega_p) 
    

    H0 = H0_ + HI_ 
    #H0 = H0.to("CSR").tidyup(atol=1e-8)

    rho0 = steadystate(H0,c)

    L0 = 0*spre(c[0])*spost(c[0].dag())
    for i in range(len(c)):
        L0 += spre(c[i])*spost(c[i].dag()) - 0.5*(spre(c[i].dag()*c[i]) + spost(c[i].dag()*c[i]))
    #L0 = sum(c)

    L0 += -1j*(spre(H0) - spost(H0))
    
    #L0 = np.array(L0)
    L0 = L0.data_as('ndarray')

    H_1 = H1_(F,eta,Omega_p)
    L_1 = -1j*(spre(H_1) - spost(H_1))

    #L_1 = np.array(L_1)
    L_1 = L_1.data_as('ndarray')

    H1 = H_1.dag()
    L1 = -1j*(spre(H1) - spost(H1))

    #L1 = np.array(L1)
    L1 = L1.data_as('ndarray')
   
    #popes = np.zeros(nn,dtype='complex64')
    #coh = np.zeros(nn,dtype='complex64')
    abs = np.zeros(nn,dtype='complex64')
    
    k=0

    for Deltap in tqdm(np.linspace(Dmin,Dmax,nn)):


        S3 = -np.matmul(inv(L0-3j*Deltap*np.eye(N**2)),L1)
        #S2 = -np.matmul(inv(L0-2j*Deltap*np.eye(N**2)),L1)
        S2 = -np.matmul(inv(L0-2j*Deltap*np.eye(N**2)+ np.matmul(L_1,S3)),L1)
        S1 = -np.matmul(inv(L0-1j*Deltap*np.eye(N**2) + np.matmul(L_1,S2)),L1)
        #S1 = -np.matmul(inv(L0-1j*Deltap*np.eye(N**2)),L1) 

        T_3 = -np.matmul(inv(L0+3j*Deltap*np.eye(N**2)),L_1)
        #T_2 = -np.matmul(inv(L0+2j*Deltap*np.eye(N**2)),L_1)
        T_2 = -np.matmul(inv(L0+2j*Deltap*np.eye(N**2)+ np.matmul(L1,T_3)),L_1)
        T_1 = -np.matmul(inv(L0+1j*Deltap*np.eye(N**2)+ np.matmul(L1,T_2)),L_1)
        #T_1 = -np.matmul(inv(L0+1j*Deltap*np.eye(N**2)),L_1)
    
        L = np.matmul(L_1,S1) + L0 +np.matmul(L1,T_1)
    
        eigenvalues, eigenvectors = eigs(L,k=1,sigma = 0+0j)

        rhoss = eigenvectors[:,0]

        rhos = np.zeros((N,N),dtype='complex64')
        for i in range(N):
            for j in range(N):
                rhos[j,i] = rhoss[j+i*N]

        rho = Qobj(rhos)
        if rho.tr() != 0:
            rho = rho/rho.tr()

        rho_ = operator_to_vector(rho)
        #rho_ = np.array(rho_)
        rho_ = rho_.data_as('ndarray')

        rho1 = np.matmul(S1,rho_)
        #rho2 = np.matmul(S2,rho_)
        #rho3 = np.matmul(S3,rho_)

        rho_1 = np.matmul(T_1,rho_)
        #rho_2 = np.matmul(T_2,rho_)
        #rho_3 = np.matmul(T_3,rho_)

        rhos1 = np.zeros((N,N),dtype='complex64')
        #rhos2 = np.zeros((N,N),dtype='complex64')
        #rhos3 = np.zeros((N,N),dtype='complex64')
        rhos_1 = np.zeros((N,N),dtype='complex64')
        #rhos_2 = np.zeros((N,N),dtype='complex64')
        #rhos_3 = np.zeros((N,N),dtype='complex64')
        for i in range(N):
            for j in range(N):
                rhos1[j,i] = rho1[j+i*N]
                #rhos2[j,i] = rho2[j+i*N]
                #rhos3[j,i] = rho3[j+i*N]
                rhos_1[j,i] = rho_1[j+i*N]
                #rhos_2[j,i] = rho_2[j+i*N]
                #rhos_3[j,i] = rho_3[j+i*N]
            
        Rho1 = Qobj(rhos1)
        #Rho2 = Qobj(rhos2)
        #Rho3 = Qobj(rhos3)
        Rho_1 = Qobj(rhos_1)
        #Rho_2 = Qobj(rhos_2)
        #Rho_3 = Qobj(rhos_3)

        #ree = 0+0j
        #rcoh = 0+0j
        rabs= 0+0j
        for ig in range(1,len(F)):
            for mfg in range(-F[ig],F[ig] + 1):
                for mfe in range(-F[0],F[0]+1):
                    for q in range(-1,2):
                        #ree+=2*np.pi*1/(2*F[0] + 1)*cg(F,ig,mfg,0,mfe,q)**2*GammaFgFe(F,ig,Je,Jg,I,GammaJgJe)*(
                            #expect(base(F,0,mfe)*base(F,0,mfe).dag(),rho))

                        #rcoh+=-1j*2*np.pi*cg(F,ig,mfg,0,mfe,q)*Omega_p[ig-1,q+1]/2*1/np.sqrt(2*F[0]+1)*(
                            #expect(base(F,ig,mfg)*base(F,0,mfe).dag(),rho) -(
                                #expect(base(F,0,mfe)*base(F,ig,mfg).dag(),rho)))

                        rabs+=-1j*2*np.pi*eta[ig-1]*cg(F,ig,mfg,0,mfe,q)*Omega_p[ig-1,q+1]/2*1/np.sqrt(2*F[0]+1)*(
                            expect(base(F,ig,mfg)*base(F,0,mfe).dag(),Rho_1)-expect(base(F,0,mfe)*base(F,ig,mfg).dag(),Rho1))

        #popes[k] = ree
        #coh[k] = rcoh
        abs[k] = rabs
        k+=1
        

    return abs


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

#Rabi frequencies Ω_F,F' #1/2π (MHz)
omega = [70,19,70]

#turn on/off polarization
Omega_p= np.zeros((3,3)) # (ig-1, q + 1) transition from F state to F' with polarization q

Omega_p[0,0] = Omega_p[0,2] = omega[0]       #σ+ and σ- for F=F[1] to F'= F[0]
Omega_p[1,0] = Omega_p[1,2] = omega[1]      #σ+ and σ- for F=F[2] to F'=F[0]
Omega_p[2,0] = Omega_p[2,2] = omega[2]     #σ+ and σ- for F=F[3] to F'=F[0]

B = 5e-4
Delta = [0,10,22.633,20]

Dmin_ = -3
Dmax_ = 3
step = 1e-02
nn_= int((Dmax_-Dmin_)/step)
print(nn_)

params = dict(F=F,N=N,Delta=Delta,Omega_p=Omega_p,eta=eta,c=c_ops,Dmin=Dmin_, Dmax=Dmax_,nn=nn_)

pops = steck(**params)

with open(""+str(path_name)+"data_B5_D21_10_D23_20_O21_O23_"+str(omega[0])+"_O22_"+str(omega[1])+"_omega_1.4_steck.txt",'wb') as f:
    np.savetxt(f, np.transpose([np.linspace(Dmin_,Dmax_,nn_),pops]),fmt='%.8f')