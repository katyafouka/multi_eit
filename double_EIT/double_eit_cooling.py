from qutip import *
import numpy as np
from scipy import constants
from tqdm import tqdm
from scipy.optimize import minimize_scalar

#constants
pi = constants.pi

#parameters (MHz/2π) following notation from  PRL 126, 023604
Gamma = 20
Deltad = 60
Omegasp = 16.5
Omegasm = Omegasp
Omegap = Omegasp
deltab = 10
Deltap = Deltad + deltab #metavlito
omega = 2

#size of hilbert spaces
Natomic = 4
Nmotion = 15

#system operators
a = tensor(qeye(Natomic),destroy(Nmotion))
y0 = 0.01
y = y0*(a + a.dag())


e = tensor(basis(Natomic,0),qeye(Nmotion))
p = tensor(basis(Natomic,1),qeye(Nmotion))
z = tensor(basis(Natomic,2),qeye(Nmotion))
m = tensor(basis(Natomic,3),qeye(Nmotion))


#Hamiltonian for a moving ion
H1 = pi*(Omegasm*(-1j*y).expm()*e*p.dag() - Omegap*(1j*y).expm()*e*z.dag() + Omegasp*(-1j*y).expm()*e*m.dag())
H2 = 2*pi*(0.5*Omegasm*(1j*y).expm()*p*e.dag() + (Deltad + deltab)*p*p.dag())
H3 = 2*pi*(-0.5*Omegap*(-1j*y).expm()*z*e.dag() + Deltap*z*z.dag())
H4 = 2*pi*(0.5*Omegasp*(1j*y).expm()*m*e.dag() + (Deltad - deltab)*m*m.dag())

H5 = 2*pi*omega*a.dag()*a


H = H1 + H2 + H3 + H4 + H5

#make sure Hamiltonian is sparse
H = H.to("CSR").tidyup(atol=1e-8)

#collapse operators
c1 = np.sqrt(2*pi*Gamma/3)*p*e.dag()
c2 = np.sqrt(2*pi*Gamma/3)*z*e.dag()
c3 = np.sqrt(2*pi*Gamma/3)*m*e.dag()

c = [c1.to("CSR"),c2.to("CSR"),c3.to("CSR")]


#create initial thermal state
def func(x,Nmax,nbar):
    
    Z = (1-np.exp(-(Nmax+1)*x))/(1-np.exp(-x))
    n = np.exp(-Nmax*x)*(np.exp((Nmax + 1)*x) -(Nmax+1)*np.exp(x) + Nmax)/(np.exp(x) -1)**2
    
    
    return np.abs(nbar - n/Z)

Nmax=14
nbar=3
res = minimize_scalar(func,args=(Nmax,nbar),bounds=(0, 1), method='bounded')
x = res.x
Z = (1-np.exp(-(Nmax+1)*x))/(1-np.exp(-x))


P = []
for i in range(Nmax+1): #Nmotion=15 but for <n> =3 keep up to i=10
    P.append(np.exp(-i*x)/Z)

rho_thermal = 0*basis(Nmax+1,0, dtype="csr")*basis(Nmax+1,0, dtype="csr").dag()
for i in range(Nmax+1):
    rho_thermal+= P[i]*basis(Nmax+1,i, dtype="csr")*basis(Nmax+1,i, dtype="csr").dag()

#initial state
rho0 = tensor(basis(Natomic,0, dtype="csr")*basis(Natomic,0, dtype="csr").dag(),rho_thermal) 

#options = Options(nsteps=10000)
tlist = np.linspace(0,1000,1000)

#rho = mesolve(H,rho0,tlist,c_ops,[a.dag()*a],options=options,progress_bar=True)
rho = mesolve(H,rho0,tlist,c_ops=c,e_ops=[a.dag()*a],options={"progress_bar": "text","nsteps": 10000})


with open(""+str(path_name)+"cooling_data/cooling_double_EIT_2.txt",'wb') as f:
    np.savetxt(f, np.transpose([tlist,rho.expect[0]]),fmt='%.8f')
