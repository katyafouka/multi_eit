import numpy as np
from qutip import *
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_6j, wigner_3j
import sys
import os

#toy model's quantum numbers
I = 1
S = 1
Je = 0
Jg = 1
Lg = 2
F = [i for i in range(int(I-Jg),int(I + Jg + 1),1)]
F.insert(0,int(I+Je))

Natomic = [2*F[i] + 1 for i in range(len(F))]
Natomic = sum(Natomic)

Nmotion = 15
a = tensor(qeye(Natomic),destroy(Nmotion))
y = a+a.dag()
     
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

II = 7
S = 1
Je = 0
Jg = 1
Lg = 2
FF = [i for i in range(int(II-Jg),int(II + Jg + 1),1)]
FF.insert(0,int(II+Je))

#Zeeman stuff
muB = 9.3e-24 #Bohr magneton J T^-1
hbar = 1.054e-34 # J s

muB = muB/hbar #hbar = 1
muB = muB/(2*np.pi)
muB = muB/1e6  # so it's in 1/2π (MHz T^-1)

# 1Gauss is 0.0001 T
B = 5e-4

#Lande g factor for F
me = 9.1093837015e-31 
mp = 1.67262192369e-27 

mI = 3.169 #nuclear magnetic moment of 176Lu in units of μN
gI = mI/II

gJg = 1 + (Jg*(Jg +1) + S*(S + 1) - Lg*(Lg + 1))/(2*Jg*(Jg + 1))
#print(gJg)

#gF = [1,0,1,2]
gF=[]
gF.append(-gI*me/mp)

for i in range(3):
    gF.append(gJg*(FF[i +1]*(FF[i+1] + 1) + Jg*(Jg+1) - II*(II+1))/(2*FF[i+1]*(FF[i +1] + 1)))

def base(F,i,mF):
    #numbering states from 0 to N-1 starting from -mF to mF
    # 0 is |F',-mF'>
    if i==0:
        b = tensor(basis(Natomic,mF+F[i], dtype="csr"),qeye(Nmotion))
    elif i==1:
        b = tensor(basis(Natomic,mF + F[i] + 2*F[0] + 1, dtype="csr"),qeye(Nmotion))
    elif i==2:
        b = tensor(basis(Natomic,mF + F[i] + 2*F[0] + 1 + 2*F[1] + 1, dtype="csr"),qeye(Nmotion))
    else:
        b = tensor(basis(Natomic,mF + F[i] + 2*F[0] + 1 + 2*F[1] + 1 + 2*F[2] +1, dtype="csr"),qeye(Nmotion))
    return b   


def GammaFgFe(F,ig,Je,Jg,I,GammaJgJe):
    return float((2*Je + 1)*(2*F[0] + 1)*wigner_6j(Je,F[0],I,F[ig],Jg,1)**2)*2*np.pi*GammaJgJe
    #return GammaJgJe

def cg(F,ig,mFg,ie,mFe,q):
    return float(CG(F[ig],mFg,1,q,F[ie],mFe).doit())
    #return 1



#Hamiltonian
def H_I(F,Omega_p):
    HI=0*tensor(basis(Natomic,0, dtype="csr"),qeye(Nmotion))*tensor(basis(Natomic,0, dtype="csr"),qeye(Nmotion)).dag()
    for mFe in range(-F[0],F[0] + 1):
        for ig in range(1,len(F)):
            for mFg in range(-F[ig],F[ig]+1):
                for q in range(-1,2):
                    HI += 2*np.pi*cg(F,ig,mFg,0,mFe,q)*Omega_p[ig-1,q+1]/2*(-1j*eta[ig-1]*y).expm(dtype="csr")*base(F,0,mFe)*base(F,ig,mFg).dag()
                
    return 1/np.sqrt(2*F[0]+1)*(HI + HI.dag()) 

def H_0(F,Delta):
    H0 = 0*tensor(basis(Natomic,0, dtype="csr"),qeye(Nmotion))*tensor(basis(Natomic,0, dtype="csr"),qeye(Nmotion)).dag()
    for l in range(len(F)):
        for mF in range(-F[l],F[l]+1):
            H0 += 2*np.pi*(Delta[l] + gF[l]*muB*mF*B)*base(F,l,mF)*base(F,l,mF).dag() 
    return H0

#Setting up

#Rabi frequencies Ω_F,F' #1/2π (MHz)
omega = [23.5,10,23.5]

#turn on/off polarization
Omega_p= np.zeros((3,3)) # (ig-1, q + 1) transition from F state to F' with polarization q

Omega_p[0,0] = Omega_p[0,2] = omega[0]       #σ+ and σ- for F=F[1] to F'= F[0]
Omega_p[1,0] = Omega_p[1,2] = omega[1]      #σ+ and σ- for F=F[2] to F'=F[0]
Omega_p[2,0] = Omega_p[2,2] = omega[2]     #σ+ and σ- for F=F[3] to F'=F[0]

Delta = [0,10,20.5,20]

H0 = H_0(F,Delta)
HI = H_I(F,Omega_p)
H = 2*np.pi*omega0*a.dag()*a + H0 + HI

#make sure Hamiltonian is sparse
H = H.to("CSR").tidyup(atol=1e-8)

#GammaJgJe = 2.5
GammaJgJe = 2.44745 #1/2π (MHz) 3D1 to 3P0

#single collapse operator for each transtion
c_ops = []
qs = [-1,0,1]

for ig in range(1,len(F)):
    for mfg in range(-F[ig],F[ig] + 1):
        for mfe in range(-F[0], F[0] + 1):
            for q in qs:
                if cg(F,ig,mfg,0,mfe,q) != 0:
                    cops =np.sqrt(1/(2*F[0] + 1))*cg(F,ig,mfg,0,mfe,q)*np.sqrt(GammaFgFe(F,ig,Je,Jg,I,GammaJgJe))*base(F,ig,mfg)*base(F,0,mfe).dag()
                    c_ops.append(cops.to("CSR"))
                else:
                    continue


# Parallelisation index. The indices in the .job file should  be 0--Npoints-1.
j = int(sys.argv[1])

#initial atomic state
psi = basis(Natomic,2*F[0] + 1 + 2*F[1] + 1 + 2*F[2] +1, dtype="csr")

#initial state
psi0 = tensor(psi,basis(Nmotion,j, dtype="csr"))
#rho0 = tensor(psi*psi.dag(),basis(Nmotion,j, dtype="csr")*basis(Nmotion,j, dtype="csr").dag())
times = np.linspace(0.0, 2000.0, 2000)
ntraj=500

#expectation values I want to calculate
exp = [a.dag()*a]


for ig in range(0,len(F)):
    for mf in range(-F[ig],F[ig] + 1):
        exp.append(base(F,ig,mf)*base(F,ig,mf).dag())

data = mcsolve(H,psi0,times,c_ops=c_ops,e_ops=exp,ntraj=ntraj,options={'progress_bar': 'text','max_step':0.05,"map": "parallel", 'num_cpus': int(os.getenv('SLURM_CPUS_PER_TASK')),"improved_sampling": True})


#averages
exps = [times]
for k in range(len(exp)):
    exps.append(data.average_expect[k])
#exps.insert(0,times)

#standard deviations
exps_std = []
for k in range(len(exp)):
    exps_std.append(data.std_expect[k])



with open(""+str(path_name)+"data_B"+str(B*1e4)+"_D10_"+str(Delta[1])+"_D12_"+str(Delta[3])+"_O10_O11_"+str(omega[0])+"_O11_"+str(omega[1])+"_"+str(j)+"_Gamma_"+str(GammaJgJe)+"_ntraj_"+str(ntraj)+"_improved_step.txt",'wb') as f:
    #np.savetxt(f, np.transpose([times,np.mean(exp,axis=0),np.std(exp,axis=0)]),fmt='%.8f')
    np.savetxt(f, np.transpose(exps),fmt='%.8f')

with open(""+str(path_name)+"data_B"+str(B*1e4)+"_D10_"+str(Delta[1])+"_D12_"+str(Delta[3])+"_O10_O11_"+str(omega[0])+"_O11_"+str(omega[1])+"_"+str(j)+"_Gamma_"+str(GammaJgJe)+"_ntraj_"+str(ntraj)+"_stds_improved_step.txt",'wb') as f:
    #np.savetxt(f, np.transpose([times,np.mean(exp,axis=0),np.std(exp,axis=0)]),fmt='%.8f')
    np.savetxt(f, np.transpose(exps_std),fmt='%.8f')
