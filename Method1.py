# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 23:47:13 2016

@author: c.senik
"""

# Load packages (need to be installed first if 
# not yet done - but is not difficult)
import numpy as np
import matplotlib.pyplot as plt # pour plot functons
plt.switch_backend('tkAgg')  # necessary for OS SUSE 13.1 version, 
# otherwise, the plt.show() function will not display any window


import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

########################################################
# Parametres du probleme
l1 = -1.5 
L1 =1.5     # intervalle de longueur pour u1   =[l1,L1]
l2 = 0.5 
L2=1.5    # intervalle de longueur pour u2   =[l2,L2]
u10 = 1.0 # valeur initiale que l'on donne à u1 (question I.1.2)
u20 = 0.0
pho = 0.002 
B = np.array([1,1]) ### on définit la valeur de A, lambda et B pour la question I.2.2. (méthode gradient optimal)
lamb = 4
A = np.array([[1,0],[0,lamb]])
print (A)

########################################################
####################    Questions I.1.1. Courbes de niveaux
def f(u1,u2) : 
    return (u1-1)**2 +100*(u1**2-u2)**2

u1 , u2 = np.meshgrid(np.linspace(l1,L1,201),np.linspace(l2,L2,201))
z = f(u1,u2)
graphe = plt.contourf(u1,u2,z,20) #ici 20 est le nombre de courbes de niveau que l'on souhaite
plt.colorbar()
plt.show()


###################     TEST 1 pour vérifier le plot 
def g(x):
    return np.sin(x)
X = np.arange(0, 3*np.pi, 0.01)
Y = [ g(x) for x in X ]
plt.plot(X, Y)
plt.show()

#################      Questions I.1.2. méthode de gradient à pas constant
def u(n) : 
    u1=u10
    u2=u20
    for k in range (n) :
        u1 = u1 - pho*f(u1,u2).diff(u1)  
        u2 = u2 - pho*f(u1,u2).diff(u2)
    return np.array([u1,u2])

k=0
u1=u10
u2=u20
while (f(u(k)[0],u(k)[1])>10**(-3)):
    k=k+1
plt.plot([u(j)[0] for j in range(k+1)],'bo')
plt.plot([u(j)[1] for j in range(k+1)])
plt.show()

#################      Questions I.1.3 -> on change uniquement la donnée initiale de pho et on applique les fonctions précédentes 

#################      Questions I.2.1. méthode gradient optimal -> vérification par écrit à rédiger 

#################      Questions I.2.2. méthode gradient optimal

def uopt(n) : 
    u1=u10
    u2=u20
    for k in range (n) :
        u1 = u1 - pho(k)*f(u1,u2).diff(u1)
        u2 = u2 - pho(k)*f(u1,u2).diff(u2)
    return np.array([u1,u2])

def pho(k) : 
    return  