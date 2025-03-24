from qutip import *
import numpy as np
from scipy import constants
from tqdm import tqdm
from scipy.optimize import minimize_scalar

#constants
hbar = constants.hbar
amu = constants.m_p
pi = constants.pi
kB = constants.Boltzmann
MHz = 10**6

#parameters (MHz/2π)
Gamma = 20
Omegar = 17
Omegag = 2*Omegar
Deltar = 70
Deltag = Deltar
omega = Gamma/10

#size of hilbert spaces
Natomic = 3
Nmotion = 15

#system operators
a = tensor(qeye(Natomic),destroy(Nmotion))
y0 = 0.01
y = y0*(a + a.dag())

e = tensor(basis(Natomic,0),qeye(Nmotion))
r = tensor(basis(Natomic,1),qeye(Nmotion))
g = tensor(basis(Natomic,2),qeye(Nmotion))


#Hamitlonian
H1 = 2*pi*(Deltag*g*g.dag() + Deltar*r*r.dag())
H2 = 2*pi*(0.5*(Omegag*e*g.dag()*(-1j*y).expm() + Omegar*e*r.dag()*(1j*y).expm() + Omegag*g*e.dag()*(1j*y).expm() + Omegar*r*e.dag()*(-1j*y).expm()))
H3 = 2*pi*omega*a.dag()*a
H = H1 + H2 + H3

#make sure Hamiltonian is sparse
H = H.to("CSR").tidyup(atol=1e-8)

#collapse operators
c2 = np.sqrt(2*pi*Gamma/2)*r*e.dag()
c3 = np.sqrt(2*pi*Gamma/2)*g*e.dag()

c_ops = [c2.to("CSR"),c3.to("CSR")]

#create initial thermal state
def func(x,Nmax,nbar):
    
    Z = (1-np.exp(-(Nmax+1)*x))/(1-np.exp(-x))
    n = np.exp(-Nmax*x)*(np.exp((Nmax + 1)*x) -(Nmax+1)*np.exp(x) + Nmax)/(np.exp(x) -1)**2
    
    
    return np.abs(nbar - n/Z)

Nmax=14
nbar=3
res = minimize_scalar(func,args=(Nmax,nbar),bounds=(0, 1), method='bounded')
x = res.x
omega0 = 2*np.pi*1.4*10e6
Z = (1-np.exp(-(Nmax+1)*x))/(1-np.exp(-x))
T = hbar*omega0/(kB*x)

P = []
for i in range(Nmax+1): #Nmotion=15 but for <n> =3 keep up to i=10
    P.append(np.exp(-i*x)/Z)

rho_thermal = 0*basis(Nmax+1,0, dtype="csr")*basis(Nmax+1,0, dtype="csr").dag()
for i in range(Nmax+1):
    rho_thermal+= P[i]*basis(Nmax+1,i, dtype="csr")*basis(Nmax+1,i, dtype="csr").dag()

#initial state
rho0 = tensor(basis(Natomic,0, dtype="csr")*basis(Natomic,0, dtype="csr").dag(),rho_thermal) 

#options = Options(nsteps=10000)
tlist = np.linspace(0,30000,15000)

#rho = mesolve(H,rho0,tlist,c_ops,[a.dag()*a],options=options,progress_bar=True)
rho = mesolve(H,rho0,tlist,c_ops=c_ops,e_ops=[a.dag()*a],options={"progress_bar": "text","nsteps": 10000})


with open("cooling_EIT_Omegap_"+str(Omegag)+"_Omegar_"+str(Omegar)+"_Delta_"+str(Deltag)+"_Gamma_"+str(Gamma)+"_eta_"+str(y0)+".txt",'wb') as f:
    np.savetxt(f, np.transpose([tlist,rho.expect[0]]),fmt='%.8f')
