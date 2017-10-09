# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:42:06 2016

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
pho=0.009 #le pas de descente
eps=0.01 #le seuil d'arrêt de la descente
U0Test=[1.,1.] #condition initiale test

lamb = 10
U0Partie2 = [5.,0.003]
A=np.array([[1,0],[0,lamb]]) #données Question 1.2.2. (gradient optimal)
b = np.array([1.,1.])
epsPartie2 = 0.01
pho0=0.1 #le pas de descente
alpha10=1.0
alpha20=1.0
sigma10=1.3
sigma20=1.4
x10=3.0
x20=7.0
T=[[1.,0.127],[2.,0.2],[3.,0.3],[4.,0.25],[5.,0.32],[6.,0.5],[7.,0.7],[8.,0.9]]   ##Valeur des xi et yi pour la partie méthode de Newton
###### Question I.1.1. Tracer les lignes de niveau ######

def J(u1,u2):
    return (u1-1.)**2 + 100.*((u1**2 - u2)**2)

def contour(u1,u2):
    U1 = np.arange(u1l, u1r, pu1)
    U2 = np.arange(u2l, u2r, pu2)
    U1, U2 = np.meshgrid(U1, U2)
    Z = J(U1, U2)
    plt.contourf(U1, U2, Z, nl)
    plt.colorbar()
    plt.show()

###### Question I.1.2. Méthode de pas constant ######
def gradJ(u):
    return [2.*(u[0]-1.) + 200.*(u[0]**2-u[1])*2.*u[0] , -200.*(u[0]**2-u[1]) ]
# On ́ecrit la descente de gradient.
# Noter que l’on ne fait qu’une seule évalutation de la fonction gradient.
def methodeDescente(pho,u0,eps):
    compt = 0
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (np.linalg.norm(a[0]**2+a[1]**2)>eps):
            grad=gradJ(a)
            a[0]=a[0]-pho*grad[0]
            a[1]=a[1]-pho*grad[1]
            compt = compt +1

            u1.append(a[0])
            u2.append(a[1])

## On va tracer les itérés de l'abcisse et de l'ordonnée de u respectivement
    lg=len(u1)
    x=np.linspace(0,lg-1,lg)
    plt.plot(x,u1,marker='o', label="suite des u1")
    plt.plot(x,u2,marker='v', label="suite des u2")
    plt.legend()
    plt.title("Méthode du pas de gradient") # Problemes avec accents
    plt.show()
    print()
    print ("Le nombre d'itérations nécessaires est :")
    print(compt)

## exemple
##methodeDescente(pho,U0,eps)



##### test sur la fonction norme L2 au carré, de minimum (0,0) #####
def gradTest(u):
    return [2*u[0],2*u[1]]
def methodeDescenteTest(pho,u0,eps):
    compt=0
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (np.linalg.norm(a[0]**2+a[1]**2)>eps):
            grad=gradTest(a)
            a[0]=a[0]-pho*grad[0]
            a[1]=a[1]-pho*grad[1]

            u1.append(a[0])
            u2.append(a[1])
            compt=compt+1

    lg=len(u1)
    x=np.linspace(0,lg-1,lg)
    plt.plot(x,u1,marker='o',label="suite des u1")
    plt.plot(x,u2,marker='v', label="suite des u2")
    plt.title("Méthode du pas de gradient") # Problemes avec accents
    plt.legend()
    plt.show()
    print()
    print ("Le nombre d'itérations nécessaires est :")
    print(compt)

## exemple sur le test, enlevez les ## pour essayer, et jouer avec les
#paramètres pho, donnée initiale teste et seuil epsilon pour constater
#que cela marche!
#methodeDescenteTest(pho,U0Test,eps)

###### Question I.2.1. Méthode de gradient optimal ######



def scal (u,v): ## fonction qui permet de calculer le produit scalaire de 2 vecteurs u et v
    lu = len(u)
    lv = len(v)
    s=0
    if (lu!=lv) :
        return 'error out of range'
    else :
        for i in range (lu) :
            s=s+u[i]*v[i]
        return s


def J2(u):
    return 0.5*scal(A.dot(u),u) -scal(b,u)

def gradJPartie2(u):
    return A.dot(u)-b

def pho(u): ## calcule pho en fonction de u (ie pho(k) pour u=uk)
    rk = gradJPartie2(u)
    num = scal(rk,rk)
    den=scal (A.dot(rk),rk)
    if(den!=0):
        return num/den
    else :
        return 'error null denominator'

def methodeDescentePartie2(u0,eps): ## qui trace u1 et u2
    compt=0
    a=u0
    r0 = np.linalg.norm(gradJPartie2(a))
    r=r0
    u1=[u0[0]]
    u2=[u0[1]]
    while (r>eps*r0):
        grad=gradJPartie2(a)
        a[0]=a[0]-pho(a)*grad[0]
        a[1]=a[1]-pho(a)*grad[1]
        compt=compt+1
        r=np.linalg.norm(gradJPartie2(a))
        u1.append(a[0])
        u2.append(a[1])

## On va tracer les itérés de l'abcisse et de l'ordonnée de u respectivement
    lg=len(u1)
    x=np.linspace(0,lg-1,lg)
    plt.plot(x,u1,marker='o', label="suite des u1")
    plt.plot(x,u2,marker='v', label="suite des u2")
    plt.legend()
    plt.title("Méthodes optimale de gradient") # Problemes avec accents
    plt.show()
    print()
    print ("Le nombre d'itérations nécessaires est :")
    print(compt)


#methodeDescentePartie2(U0Partie2,epsPartie2)

##### Q.I.2.3. Tracer l'évolution du gradient #####

def gradientEvolution(u0,eps):   #On reprend quasiment tout le code de la question précédente
    compt=0
    a=u0
    r0 = np.linalg.norm(gradJPartie2(a))
    r=[1.]
    r1=r0
    while (r1>eps*r0):
        grad=gradJPartie2(a)
        a[0]=a[0]-pho(a)*grad[0]
        a[1]=a[1]-pho(a)*grad[1]
        compt=compt+1
        r1=np.linalg.norm(gradJPartie2(a)/r0)
        r.append(r1)

    lg=len(r)
    x=np.linspace(0,lg-1,lg)
    plt.plot(x,r,marker='o', label="suite des rk/rO")
    plt.yscale('log')
    plt.grid(True,which="both")
    plt.xlabel(r"norme du gradient")
    plt.ylabel(r"itere")
    plt.legend()
    plt.title("Methodes optimale de gradient")
    plt.show()
    print()
    print ("Le nombre d'itérations nécessaires est :")
    print(compt)

#gradientEvolution(U0Partie2,epsPartie2)

##### Q.I.2.4. dichotomie sur Rosenbrock #####

def Condition(u,pho1):
    rk=gradJ(u)
    if(J(u[0],u[1])>(J(u[0]-pho1*rk[0],u[1]-pho1*rk[1]))):
        return 0
    else:
        return 1

def dichotomie(u0,pho0):    ## Dépend de la valeur du pas initial
    compteur=0
    pho1=pho0
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (compteur<20000):
        if(Condition(a,pho1)==0):
            compteur=compteur+1
            grad=gradJ(a)
            a[0]=a[0]-pho1*grad[0]
            a[1]=a[1]-pho1*grad[1]
            u1.append(a[0])
            u2.append(a[1])
            pho1=pho0
        else:
            compteur=compteur+1
            pho1=pho1/2
    lg=len(u1)
    x=np.linspace(0,lg-1,lg)
    plt.plot(x,u1,marker='o', label="suite des u1")
    plt.plot(x,u2,marker='v', label="suite des u2")
    plt.grid(True,which="both")
    plt.xlabel(r"iteres")
    plt.ylabel(r"valeurs abscisses et ordonnées")
    plt.legend()
    plt.title("Methodes dichotomique de gradient")
    plt.show()

##dichotomie(U0,pho0)

##la même chose sur la fonction test

def ConditionTest(u,pho1):
    rk=gradTest(u)
    if(u[0]**2+u[1]**2>((u[0]-pho1*rk[0])**2+(u[1]-pho1*rk[1])**2)):
        return 0
    else:
        return 1

def dichotomieTest(u0,pho0):    ## Dépend de la valeur du pas initial
    compteur=0
    pho1=pho0
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (compteur<20):
        if(ConditionTest(a,pho1)==0):
            compteur=compteur+1
            grad=gradTest(a)
            a[0]=a[0]-pho1*grad[0]
            a[1]=a[1]-pho1*grad[1]
            u1.append(a[0])
            u2.append(a[1])
            pho1=pho0
        else:
            compteur=compteur+1
            pho1=pho1/2
    lg=len(u1)
    x=np.linspace(0,lg-1,lg)
    plt.plot(x,u1,marker='o', label="suite des u1")
    plt.plot(x,u2,marker='v', label="suite des u2")

    plt.xlabel(r"iteres")
    plt.ylabel(r"valeurs abscisses et ordonnées")
    plt.legend()
    plt.title("Methodes dichotomique de gradient")
    plt.show()

#dichotomieTest(U0,pho0)
def g(x,a1,a2,s1,s2,x1,x2):
    res1=a1*np.exp(-0.5*((x-x1)**2)/(s1**2))
    res2=a2*np.exp(-0.5*((x-x2)**2)/(s2**2))
    return (res1+res2)

def f(i,a1,a2,s1,s2,x1,x2):        #composante i-ième de f
    return T[i][1]-g(T[i][0],a1,a2,s1,s2,x1,x2)
    
def vecteurf(a1,a2,s1,s2,x1,x2):
    V=[]
    for i in range(8):
        V.append(f(i,a1,a2,s1,s2,x1,x2))
    return V

def composanteDifferentielle(i,j,a1,a2,s1,s2,x1,x2): #ici m=8, et n=6. Df est donc une matrice de taille 8*6. En (i,j), on lit dfi/dxj 
    if(j==0):                        #Dans l'ordre les arguments sont a1,a2,s1,s2,x1,x2 et sont les directions de bases de R6
        return (np.exp(-0.5*((T[i][0]-x1)**2)/(s1**2)))   ##ici donc on dérive fi en fonction de a1
    if(j==1): 
        return np.exp(-0.5*((T[i][0]-x2)**2)/(s2**2))      ##ici en fonction de a2
    if(j==2):
        return a1*np.exp(-0.5*((T[i][0]-x1)**2)/(s1**2))*((T[i][0]-x1)**2)/(s1**3)  ##ici en fonction de s1
    if(j==3):
        return a2*np.exp(-0.5*((T[i][0]-x2)**2)/(s2**2))*((T[i][0]-x2)**2)/(s2**3)  ##ici en fonction de s2
    if(j==4):
        return a1*np.exp(-0.5*((T[i][0]-x1)**2)/(s1**2))*(T[i][0]-x1)/(s1**2)   ##ici en fonction de x1
    if(j==5):
        return a2*np.exp(-0.5*((T[i][0]-x2)**2)/(s2**2))*(T[i][0]-x2)/(s2**2)   ##ici en fonction de x2
    else: 
        return ("erreur sur l'indice de derivation")
        
def Df(a1,a2,s1,s2,x1,x2):
    Df1=[]
    Df2=[]    
    for i in range (8):
        for j in range (6):
           Df2.append(composanteDifferentielle(i,j,a1,a2,s1,s2,x1,x2))
        Df1.append(Df2)
        Df2=[]
    return Df1


def ApproximationH(a1,a2,s1,s2,x1,x2):    ##Comme suggéré par l'énoncé le produit de DfT avec Df
    Df1=Df(a1,a2,s1,s2,x1,x2)
    Df2=np.transpose(Df1)
    return (np.dot(Df2,Df1))

def gradientNewton(a1,a2,s1,s2,x1,x2):                 ##Newton en dimension 1, c'est faire x(k+1)=x(k)-f'(x(k))/f(x(k)). À ce stade on a f', il nous faut tout de même f, c'est à dire le gradient de J
    Df1=Df(a1,a2,s1,s2,x1,x2)           ##remarquons que le gradient de J, c'est, avec les notations de l'énoncé, Df transposée fois f.
    Df1=np.transpose(Df1)
    f1=vecteurf(a1,a2,s1,s2,x1,x2)       
    return np.dot(Df1,f1) 
    
def Differentielle(f,x,eps):    ##epsilon est le seuil que l'on se donne pour approximer les dérivées partielles sous la forme (f(x+eps*e_i)-f(x))/eps
    n=len(x)
    m=len(f(x))
    Diff=np.eye(m,n)
    e=np.zeros(n)
    for i in range (n):
        if (i==0):
            e[i]=1
        else:
            e[i-1]=0
            e[i]=1       
        for j in range (m):
            Diff[j][i]=(1./eps)*(f(x+eps*e)[j]-f(x)[j])
    return Diff
    
def Fonctiontest(x):
    res1=0.
    res2=0.
    for i in range (len(x)):
        res1=res1+x[i]**2
        res2=res2+x[i]**3
    return [res1,res2]
        
print(Differentielle(Fonctiontest,[1.,2.,3.,1.,5.,1.5],0.001))