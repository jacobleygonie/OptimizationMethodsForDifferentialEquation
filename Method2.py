# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:06:21 2016

@author: c.senik
"""

########################################################
# Load packages (need to be installed first if 
# not yet done - but is not difficult)
import math
import numpy as np
import matplotlib.pyplot as plt # pour plot functons
import scipy.sparse as sp # pour la construction de matrice creuse
from scipy.sparse import linalg

########################################################
# Parametres du probleme
u1l = -1.5
u1r = 1.5 ## intervalle pour u1 [u1l, u1r]
u2l = -1.5
u2r = 1.5 ## intervalle pour u2 [u2l, u2r]
pu1 = 0.01 #pas pour u1 
pu2 = 0.01 #pas pour u2
nl = 20 #nombre de lignes de niveauw désirées pour question I.1.1.
U0 = [1.,0.] #condition initiale Questiont I.1.2.
pho=0.02  #le pas de descente
pho0=0.5         #le pas de descente initiale pour la méthode de gradient optimal
eps=0.001      #le seuil d'arrêt de la descente
compteur=0    #nombre d'itérés
U0Test=[1.,1.] #condition initiale test
###### Question I.1.1. Tracer les lignes de niveau ######

def J(u1,u2):
    return (u1-1.)**2 + 100.*(u1**2 - u2)**2
U1 = np.arange(u1l, u1r, pu1)
U2 = np.arange(u2l, u2r, pu2)
U1, U2 = np.meshgrid(U1, U2)
Z = J(U1, U2)
plt.contourf(U1, U2, Z, nl)
plt.colorbar()
plt.show()


def gradJ(u,compteur):
    compteur=compteur+1
    return [2.*(u[0]-1.) + 200.*(u[0]**2-u[1])*2.*u[0] , -200.*(u[0]**2-u[1]) ]
# On  ́ecrit la descente de gradient.
# Noter que l’on ne fait qu’une seule  ́evalutation de la fonction gradient.
def methodeDescente(pho,u0,eps,compteur):
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (np.linalg.norm(gradJ(a,compteur))>eps):
        grad=gradJ(a,compteur)
        a[0]=a[0]-pho*grad[0]
        a[1]=a[1]-pho*grad[1]
        
        u1.append(a[0])
        u2.append(a[1])
    
## On va tracer les itérés de l'abcisse et de l'ordonnée de u respectivement 
    lg=len(u1)
    x=np.linspace(0,lg-1,lg)
    p1=plt.plot(x,u1,marker='o')
    p2=plt.plot(x,u2,marker='v')
    plt.title("Méthodes du pas de gradient")  # Problemes avec accents (plot_directive) !
    plt.legend([p1, p2], ["Abscisse", "Ordonnée"])
    plt.show()
    
## exemple   
methodeDescente(pho,U0,eps,0) 
  
  
###### test sur la fonction norme L2 au carré, de minimum (0,0) ######
def gradTest(u,compteur):
    return [2*u[0],2*u[1]]
def methodeDescenteTest(pho,u0,eps,compteur):
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (np.linalg.norm(gradTest(a,compteur))>eps):
        grad=gradTest(a,compteur)
        a[0]=a[0]-pho*grad[0]
        a[1]=a[1]-pho*grad[1]
        
        u1.append(a[0])
        u2.append(a[1])
    lg=len(u1)
    x=np.linspace(0,lg-1,lg)
    p1=plt.plot(x,u1,marker='o')
    p2=plt.plot(x,u2,marker='v')
    plt.title("Méthodes du pas de gradient")  # Problemes avec accents (plot_directive) !
    plt.legend([p1, p2], ["Abscisse", "Ordonnée"])
    plt.show()
 
## exemple sur le test, enlevez les ## pour essayer, et jouer avec les paramètres pho, donnée initiale teste et seuil epsilon pour constater que cela marche!
## methodeDescenteTest(pho,U0Test,eps,0) 
 
###### Question I.2.1 #####
 
def methodeDescenteOpt(pho0,u0,eps,compteur,lamb):
    pho=pho0
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (np.linalg.norm(gradJ(a,compteur))>eps):
        grad=gradJ(a,compteur)
        a[0]=a[0]-pho*grad[0]
        a[1]=a[1]-pho*grad[1]
        
        u1.append(a[0])
        u2.append(a[1]) 