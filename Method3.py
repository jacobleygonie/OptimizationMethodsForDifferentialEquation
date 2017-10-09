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
rho=0.002 #le pas de descente
eps=0.001 #le seuil d'arrêt de la descente
U0Test=[1.,0.7] #condition initiale test

lamb = 10
U0Partie2 = [5.,0.003]
A=np.array([[1,0],[0,lamb]]) #données Question 1.2.2. (gradient optimal)
b = np.array([1.,1.])
epsPartie2 = 0.01
pho0=0.001 #le pas de descente
alpha10=1.0  #Ce sont les paramètres initiaux de la descente de Newton
alpha20=1.0
sigma10=1.0
sigma20=1.0
x10=3.0
x20=6.0
T=[[1.0,0.127],[2.0,0.2],[3.0,0.3],[4.0,0.25],[5.0,0.32],[6.0,0.5],[7.0,0.7],[8.0,0.9]]   ##Valeur des xi et yi pour la partie méthode de Newton
X=[alpha10,alpha20,sigma10,sigma20,x10,x20]   ##Le premier vecteur des itérés dans la descente de Newton

dx= 0.2                     #pas d'espace
imax=int(10.0/dx)                            # borne pour le nombre de données discrétisées en espace pour IV. Pour tout temps n, on a les variables spatiales u_j^n, où j est dans [-imax,imax]
dt=0.001                    #pas de temps avec dt<<dx
b2=1.0
Tfinal=10. 
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier)


###### Question I.1.1. Tracer les lignes de niveau ######

def J(u1,u2):
    return (u1-1.)**2 + 100.*(u1**2 - u2)**2

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
def methodeDescente(rho,u0,eps):
    compt = 0
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (np.linalg.norm((a[0]-1)**2+100*((a[0]**2)-a[1]**2)>eps)):
            grad=gradJ(a)
            a[0]=a[0]-rho*grad[0]
            a[1]=a[1]-rho*grad[1]
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
#methodeDescente(rho,U0,eps)



##### test sur la fonction norme L2 au carré, de minimum (0,0) #####
def gradTest(u):
    return [2*u[0],2*u[1]]
def methodeDescenteTest(rho,u0,eps):
    compt=0
    a=u0
    u1=[u0[0]]
    u2=[u0[1]]
    while (np.linalg.norm(a[0]**2+a[1]**2)>eps):
            grad=gradTest(a)
            a[0]=a[0]-rho*grad[0]
            a[1]=a[1]-rho*grad[1]

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
#methodeDescenteTest(rho,U0Test,eps)

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
    plt.xlabel(r"itéré")
    plt.ylabel(r"norme du gradient")
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
    while (compteur<200):
        if(Condition(a,pho1)==1):
            compteur=compteur+1
            grad=gradJ(a)
            a[0]=a[0]-pho1*grad[0]
            a[1]=a[1]-pho1*grad[1]
            u1.append(a[0])
            u2.append(a[1])
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
    
dichotomie(U0,pho0)

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
    while (compteur<2000):            ## Ajuster ce paramètre pour observer plus finement le résultat
        if(ConditionTest(a,pho1)==0):
            compteur=compteur+1
            grad=gradTest(a)
            a[0]=a[0]-pho1*grad[0]
            a[1]=a[1]-pho1*grad[1]
            u1.append(a[0])
            u2.append(a[1])
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
    
#dichotomieTest(U0Test,pho0)
    
    
##### Q.II.2.2 Gauss Newton en dimension 2, i.e J est une fonction de 6 arguments#####

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
        return (-np.exp(-0.5*((T[i][0]-x1)**2)/(s1**2)))   ##ici donc on dérive fi en fonction de a1
    if(j==1): 
        return -np.exp(-0.5*((T[i][0]-x2)**2)/(s2**2))      ##ici en fonction de a2
    if(j==2):
        return -a1*np.exp(-0.5*((T[i][0]-x1)**2)/(s1**2))*((T[i][0]-x1)**2)/(s1**3)  ##ici en fonction de s1
    if(j==3):
        return -a2*np.exp(-0.5*((T[i][0]-x2)**2)/(s2**2))*((T[i][0]-x2)**2)/(s2**3)  ##ici en fonction de s2
    if(j==4):
        return -a1*np.exp(-0.5*((T[i][0]-x1)**2)/(s1**2))*(T[i][0]-x1)/(s1**2)   ##ici en fonction de x1
    if(j==5):
        return -a2*np.exp(-0.5*((T[i][0]-x2)**2)/(s2**2))*(T[i][0]-x2)/(s2**2)   ##ici en fonction de x2
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
    f1=np.arange(8)
    f1=vecteurf(a1,a2,s1,s2,x1,x2)      
    return np.dot(Df1,f1) 

def GaussNewton(alpha10,alpha20,sigma10,sigma20,x10,x20,N):    #N est le nombre d'itération
    a1=alpha10
    a2=alpha20        ##ce sont les initialisations de nos paramètres. On va les stocker dans un tableau
    s1=sigma10
    s2=sigma20
    x1=x10
    x2=x20
    Itere=[[a1,a2,s1,s2,x1,x2]]    #Itere sera le tableau avec les N valeurs des itérés des 6 arguments
    for k in range (N):
        a1=Itere[k][0]
        a2=Itere[k][1]        
        s1=Itere[k][2]
        s2=Itere[k][3]
        x1=Itere[k][4]
        x2=Itere[k][5]
        Gradient=gradientNewton(a1,a2,s1,s2,x1,x2)        
        Hessienne= np.linalg.inv(ApproximationH(a1,a2,s1,s2,x1,x2))                    ##c'est ce qui nous intéresse, l'inverse de la hessienne
        
        Itere.append(Itere[k]-np.dot(Hessienne,Gradient))    ##le k+1 eme vecteur à partir du k-ème
        
        
    
    Itere0=[]    #6 tableaux de valeurs pour les itérés de a1,a2,s1,s2,x1,x2
    Itere1=[] 
    Itere2=[] 
    Itere3=[] 
    Itere4=[] 
    Itere5=[]     
    for k in range (N):
        Itere0.append(Itere[k][0])
        Itere1.append(Itere[k][1])
        Itere2.append(Itere[k][2])
        Itere3.append(Itere[k][3])
        Itere4.append(Itere[k][4])
        Itere5.append(Itere[k][5])
    x=np.arange(0,N)
    plt.subplot(221)
    plt.plot(x,Itere0, marker='o', label="suite des a1")
    plt.plot(x,Itere1, marker='v', label="suite de a2")
    plt.grid(True,which="both")
    plt.xlabel(r"iteres")
    plt.ylabel(r"Amplitudes")
    plt.legend()
    plt.subplot(222)
    plt.plot(x,Itere2, marker='o', label="suite des s1")
    plt.plot(x,Itere3, marker='v', label="suite de s2")
    plt.grid(True,which="both")
    plt.xlabel(r"iteres")
    plt.ylabel(r"Variances")
    plt.legend()
    plt.title("Methodes dichotomique de gradient") 
    plt.subplot(223)
    plt.plot(x,Itere4, marker='o', label="suite des x1")
    plt.plot(x,Itere5, marker='v', label="suite de x2")
    plt.grid(True,which="both")
    plt.xlabel(r"iteres")
    plt.ylabel(r"Moyennes")
    plt.legend()
    
    
##Un petit exemple avec les données de l'énoncé, attention aux valeurs initiales qui changent tout!
#GaussNewton(alpha10,alpha20,sigma10,sigma20,x10,x20,10)
    
#####Q.IV.1.1 La question est théorique mais on introduit ici les outils nécessaires à la suite#####
    
def gaussienne(x):
    return np.exp(-0.5*(x**2))/(np.sqrt(2*np.pi))
    
V=[]
for k in range (2*imax+1):
    V.append(gaussienne(-10+20*(k/(2*imax))))
 
Diagonale=(1.0+2.0*b2*dt/(dx**2))*np.ones(2*imax+1)
SousDiagonale=(-b2*dt/(dx**2))*np.ones(2*imax)           ##On construit A qui est tridiagonale (entre autres grâce aux conditions de Dirichlet au bord)
SurDiagonale=(-b2*dt/(dx**2))*np.ones(2*imax)
A2=np.diag(Diagonale)+np.diag(SurDiagonale,1)+np.diag(SousDiagonale,-1)
def DiagonaleU(Un):
    DiagonaleUn=np.eye(2*imax+1)  ## C'est la matrice dont les composantes sont les u_j^n **3
    for k in range (2*imax+1):
        DiagonaleUn[k][k]=dt*(Un[k]**3)
    return DiagonaleUn

DiagonaleU0=np.eye(2*imax+1) 
for k in range (2*imax+1):
        DiagonaleU0[k][k]=dt*gaussienne(-10+20*(k/(2*imax)))
    
#####Q.IV.1.2 Méthode du gradient de pas optimal pour inverser le shcéma implicite linéarisé de l'équation de la chaleur non linéaire#####

def J4(u,Un):
    return 0.5*scal((A2+DiagonaleU).dot(u),u) -scal(Un,u)
    
def gradJPartie4(u,DiagonaleCube,Un):
    return (np.dot(A2+DiagonaleCube,u)-Un)

def pho4(u,DiagonaleCube,Un): ## calcule pho en fonction de u (ie pho(k) pour u=uk)
    rk = gradJPartie4(u,DiagonaleCube,Un)
    num = scal(rk,rk)
    den=scal(np.dot((A2+DiagonaleCube),rk),rk)
    if(den!=0):
        return num/den
    else : 
        return 'error null denominator'

    

def methodeDescentePartie4(Un,eps): ## Résout U(n)=A*U(n+1) par descente à pas optimal
    u=Un                        ## Attention à ne pas se perdre, on initialise ici la descente avec u=Un et on va réitérer la descente sur u pour se rapprocher de U(n+1)
    DiagonaleCube=DiagonaleU(Un)
    pho=pho4(u,DiagonaleCube,Un)                    ##pas optimal pour la 1ère boucle
    r0 = np.linalg.norm(gradJPartie4(u,DiagonaleCube,Un))                   
    r=r0
    while(r>r0*eps):                        ##On a choisit de s'arrêter quand le gradient était suffisamment petit.
        grad=gradJPartie4(u,DiagonaleCube,Un)
        u=u-pho*grad
        r=np.linalg.norm(gradJPartie4(u,DiagonaleCube,Un))              
        pho=pho4(u,DiagonaleCube,Un)                        # Avant l'entrée dans la prochaine boucle, calcul du nouveau pas optimal
    return u               ## C'est notre approximation de U(n+1)
       

def EvolutionTemporelle4(V,eps,tempo):     ##Fixez l'intervalle de temps entre deux courbes consécutives avec tempo, par exemple tempo=0.1
  # Print solution
    Un=V
    x=np.linspace(-10,10,2*imax+1)
    for n in range(1,nt+1):

    # Solution de la descente  

        Un=methodeDescentePartie4(Un,eps)

   # Print solution
        if n%5 == 0:
            plt.figure(2)
            plt.clf()
        
            plt.plot(x,V,'b',x,Un,'r')
            plt.xlabel('$x$')
            plt.ylabel('$Température$')
            plt.title('Evolution temporelle pour equation de la chaleur')
        
    
            plt.pause(tempo)  

## On va tracer les itérés de l'abcisse et de l'ordonnée de u respectivement
    
    
#####Q.IV.2.1 et Q.IV.2.2#####
    
#Ici G (la fonction issue de la question IV.2.1) est une fonction de R^(2imax+1) dans R^(2imax+1)    
#Il n'est pas immédiat que la recherche de la solution d'un problème du type G(u)=0 se ramène à un problème de Gausse Newton.
#En réalité, il suffit de voir que l'on veut résoudre de manière équivalente: 1/2||G(U)||^2=0
    
#D'abord, même si cela n'est pas utile au code de la descente (on n'a besoin que du gradient de G)
#Donnons la forme de G pour répondre à la 1ère question, on prend Q=1b=1
    
V0=[]
for i in range (2*imax+1):
    V0.append(1.)
    
def G(V):
    n=len(V)     ## Normalement cette longeur vaut 2*imax +1
    T=[V[0]**4-1.0-(V[1]-2*V[0])/(dx**2)]   ## première composante de G(V) avec la condition au bord 
    for i in range(n-2):
        T.append(V[i+1]**4-1.0-(V[i]+V[i+2]-2*V[i+1])/(dx**2))    ##composantes médiantes de G(V)
    T.append(V[n-1]**4-1.0-(V[n-2]-2*V[n-1])/(dx**2))    ##dernière composante de G(V) avec la condition au bord
    return T   
    
def composanteDifferentielleIV(i,j,V): # DG est donc une matrice de taille (2imax+1)*(2imax+1). En (i,j), on lit dGi/dxj(V)
    n=len(V)
    if(i==0):
        if(j==0):
            return ((4.*(V[0]**3)+(2./(dx**2))))
        if(j==1):
            return (-V[1]/(dx**2))
        else:
            return 0.0
    if(i==n-1):
        if(j==n-1):
            return (4.*(V[n-1]**3)+(2./(dx**2)))
        if(j==n-2):
            return (-V[n-2]/(dx**2))
        else:
            return 0.0
    else:
        if(j==i):
            return (4.*(V[i]**3)+(2./(dx**2)))
        if(j==i-1):
            return (-V[i-1]/(dx**2))
        if(j==i+1):
            return (-V[i+1]/(dx**2))
        else:
            return 0.0
            
def DG(V):
    n=len(V)
    DG1=[]
    DG2=[]    
    for i in range (n):
        for j in range (n):
           DG2.append(composanteDifferentielleIV(i,j,V))
        DG1.append(DG2)
        DG2=[]
    return DG1
    
def ApproximationHIV(V):    ##Comme suggéré par l'énoncé le produit de DfT avec Df donne la hessienne
    DG1=DG(V)
    DG2=np.transpose(DG1)
    return (np.dot(DG2,DG1))
    
def gradientNewtonIV(V):                 ##Newton en dimension 1, c'est faire x(k+1)=x(k)-f'(x(k))/f(x(k)). À ce stade on a f', il nous faut tout de même f, c'est à dire le gradient de J
    DG1=DG(V)           ##remarquons que le gradient de G, c'est, avec les notations de l'énoncé, DG transposée fois G.
    DG1=np.transpose(DG1)
    G1=G(V)     
    return np.dot(DG1,G1)
    
#On va faire la descente de Newton, en inversant le gradient selon la technique du pas optimal.
#Concrètement, la descente consiste à faire u(k+1)=u(k)-HessienneG^(-1)(u(k))gradG(uk)
#Cela revient à HessienneG(uk)u(k+1)=HessieneG(uk)u(k)-gradG(uk)
#C'est ce système que l'on résout par descente à pas optimal.
    
def JIV(V):        ##On minimise donc une certaine 1/2*<Au,u> - <b,u>
    return 0.5*scal(ApproximationHIV(V).dot(V),V) -scal(ApproximationHIV(V).dot(V)-gradientNewtonIV(V),V)

def gradJPartieIV(V):    ##c'est le fameux rk qui sert à trouver le pas optimal
    return ApproximationHIV(V).dot(V)-(ApproximationHIV(V).dot(V)-gradientNewtonIV(V))

def phoIV(V): ## calcule pho en fonction de V (ie pho(k) pour u=uk)
    rk = gradJPartieIV(V)
    num = scal(rk,rk)
    den=scal (ApproximationHIV(V).dot(rk),rk)
    if(den!=0):
        return num/den
    else : 
        return 'error null denominator'

def methodeDescentePartieIV(u0,N): ## N est le nombre d'itérations que l'on souhaite effectuer
    V=u0
    i=0
    while (i<N):
        grad=gradJPartieIV(V)
        V=V-phoIV(V)*grad
        i=i+1
      
    return V
    
def solutionApprochee(u0,N,M):   ## N le nombre d'itérations pour la descente, M le nombre d'itérations de la méthode de Gauss-Newton
    V=u0                                ## Attention, de nombreuses fonctions à grands coûts sont imbriquées. Prendre N=10, M=100,dx=0.2 environ sont les meilleures possibilités étant donné la puissance à disposition
    for k in range (M):
        V=methodeDescentePartieIV(V,N)
    x=np.linspace(-10,10,2*imax+1)
    plt.plot(x,V, marker='o', label="solution spatiale approchée")
    plt.xlabel(r"abscisse")
    plt.ylabel(r"Amplitude")
    plt.legend()
    plt.show()
        
        


    
