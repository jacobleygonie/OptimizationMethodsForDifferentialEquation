import numpy as np
import matplotlib.pyplot as plt # pour plot functons
plt.switch_backend('tkAgg')  # necessary for OS SUSE 13.1 version, 
# otherwise, the plt.show() function will not display any window

import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

# Parametres du probleme
lg = 10        # intervalle en x=[-lg,lg]
dx = 0.1       # dx = pas d'espace
cfl = 0.49     # cfl = dt/dx^2
dt = dx*dx*cfl # dt = pas de temps
Tfinal = 0.5   # Temps final souhaite

nx = 2*lg/dx + 1 # nx = nombre d'intervals
imax=10

#print(int(nx))
x = np.linspace(-10.,10.,2*imax+1)

# Initialize u0
u0 = np.zeros(len(x))
#print(len(u0))

# Set specific u0 values (same as in the scilab program)
for k in range (len(x)):
    if (1.0 - x[k]**2) < 0:
       u0[k] = 0
    else:
       u0[k] = 1.0 - x[k]**2 # donnee initiale
      
       
# Initialize u by the initial data u0
uimp = u0.copy() # il faut faire une copie sinon on va changer u0 en changeant u
uexp = u0.copy()

# Construction de la matrice (creuse) pour le schema explicite 
# u(-10,t)=u(10,t)=0 on peut donc enlever ces deux coordonnees du systeme
#u_j^{n+1}=nu*cfl*u_{j-1}^n+(1-2*nu*cfl)u_j^n+nu*cfl*u_{j+1}^n
Aexp=sp.sparse.diags([cfl, 1-2*cfl, cfl], [-1, 0, 1], shape=(nx-2, nx-2))

# Construction de la matrice (creuse) pour le schema implicite 
#-nu*cfl*u_{j-1}^{n+1}+(1+2*nu*cfl)u_j^{n+1}-nu*cfl*u_{j+1}^{n+1}=u_j^n
Aimp=sp.sparse.diags([-cfl, 1+2*cfl, -cfl], [-1, 0, 1], shape=(nx-2, nx-2))

# Sanity check to see how matrix looks like
#print(Aexp.toarray())
#print(Aimp.toarray())

# Factorisation LU de la matrice Aimp
AimpFact = sp.sparse.linalg.factorized(Aimp)

# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier)
x = np.linspace(-imax,imax,2*imax+1) 


  # Print solution
for n in range(1,nt+1):

    # Solution 1: Solve explicit system
    uexp[1:-1]=Aexp*uexp[1:-1]

    # Solution 2: schema implicite 
#    uimp = sp.sparse.linalg.spsolve(Aimp,uimp)
    uimp[1:-1]=AimpFact(uimp[1:-1])

   # Print solution
    if n%5 == 0:
        plt.figure(2)
        plt.clf()
        
        plt.plot(x,u0,'b',x,uexp,'r')
        plt.xlabel('$x$')
        plt.title('Schema explicite')
        
    
        plt.pause(0.1)
        plt.show()