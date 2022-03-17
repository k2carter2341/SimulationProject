import random
from statistics import variance
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt

N = 1000  # number of cells
rho = 0.8  # density
vol = N/rho # volume
L = np.power(vol, 1 / 3)  # length of the simulation
print("L = ", L)

minDist = 0.8 # the minimum distance each 
iseed = 10

random.seed(iseed)
#--------------------------------------------------------------------------
#  generate 3 random numbers between 0 and L
#--------------------------------------------------------------------------
def x_rand(L):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    z = random.uniform(0, L)
    return x, y, z

""" Initialize array x, y, z"""
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

#--------------------------------------------------------------------------
#  function to generate randomc ocnfiguration s.t. distance between 2 particles > minDist
#--------------------------------------------------------------------------
def InitConf(minDist):
    x[0], y[0], z[0] = x_rand(L) # choose an arbitrary position for the very 1st particle
    i = 0
    while i < N:
            x_t, y_t, z_t = x_rand(L) # trial position
            iflag = 1 # flag for accepting trial position in x, y, z list if dist > minDist
            for j in range(i): # look for all possible pairs
                dx = x[j] - x_t
                dx = dx - L * np.round(dx/L) # minimun image distance

                dy = y[j] - y_t
                dy = dy - L * np.round(dy/L)

                dz = z[j] - z_t
                dz = dz - L * np.round(dz/L)

                dr2 = dx**2 + dy**2 + dz**2
                if(dr2 < minDist*minDist):
                    iflag = 0 # iflag=0 means don't accept the trial position: see later lines
                    break
            if(iflag==1): # this line will reach (i) by above break statement or (ii) after finishing above for loop
                x[i] = x_t; y[i] = y_t; z[i] = z_t; i = i + 1

#--------------------------------------------------------------------------
#  function to calculate distance of 2 particles (x1,y1,z1) and (x2,y2,z2)
#--------------------------------------------------------------------------
def dist(x1, y1, z1, x2, y2, z2, Lx, Ly, Lz):   #Use these variables in gaussian def
    dx = x1 - x2
    dx = dx - Lx * np.round(dx/Lx) # minimum image distance

    dy = y1 - y2
    dy = dy - Ly * np.round(dy/Ly)

    dz = z1 - z2
    dz = dz - Lz * np.round(dz/Lz)

    dr2 = dx**2 + dy**2 + dz**2
    dr = np.sqrt(dr2)

    return dr
#=========================================================================
#=========================================================================
#  MAIN fnction to call all functiona as required
#=========================================================================
#=========================================================================

InitConf(minDist) # calling function to generate initial configuration

# Call dist() to calculate all the diatances. Just for checking 
fp = open("dist.txt", mode="w")
npair = int(N*(N-1)/2)
print(" \n")
print("npair= %s" %(npair))

r = np.zeros(npair)
index = np.zeros(npair)
ind = -1
for i in range(N):
    for j in range(i+1, N, 1):
        ind = ind + 1
        r1 = dist(x[i], y[i], z[i], x[j], y[j], z[j], L, L, L)
        r[ind] = r1
        fp.write("%s  %s\n" %(ind, r[ind]))
        index[ind] = ind 
fp.close()
#--------------------------------------------------------------------------
# Plot scattered plot
#--------------------------------------------------------------------------
fp = open("point.txt", mode="w")
for i in range(N):
    line = str(x[i]) + " " + str(y[i]) + " " + str(z[i])
    fp.write("%s\n" % (line))
fp.close()

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(x, y, z)
plt.show()
#--------------------------------------------------------------------------
# Plot distances among all particles
#--------------------------------------------------------------------------
fig = plt.figure()
plt.plot(index, r, "o")
plt.axhline(y=minDist, color='r', linestyle='--')
plt.show()

#my attempt at gaussian distribution
#figure out theory of:joint propability distribution, jacobian determinant and jacobian matrix khan academy, box-muller method -- this should all go into theory section***

#Uses the module? to find distribution... but not sure how this gets us to velocities just quite yet
#def gaussian(x1, y1, z1, x2, y2, z2)

#Dont think this gives me what i want
mu = 100
sigma = 50
dr = []
    
for i in range(1000): 
    temp = random.gauss(mu, sigma)
    dr.append(temp) 
        
# plotting a graph 
plt.plot(dr) 
plt.show()


#BELOW IS THE CODE FROM THE TEXTBOOK AND MY NOTES DISECTING IT
# Start of the function: FUNCTION gasdev(idum)
#I think this is a comment in the book?: C USES ran1
# INTEGER idum
# REAL gasdev
# Returns a normally distributed deviate with zero mean and unit variance, using ran1(idum)
# as the source of uniform deviates.
#  variable type: INTEGER iset   
# variable type floating: REAL fac,gset,rsq,v1,v2,ran1  
# Stores return variable: SAVE iset,gset  
# Like lists: DATA iset/0/
# same as java and python: if (idum.lt.0) iset=0 Reinitialize. 
# same as java and python: if (iset.eq.0) then We don’t have an extra deviate handy, so
# 1 v1=2.*ran1(idum)-1. pick two uniform numbers in the square extendv2=2.*ran1(idum)-1. ing from -1 to +1 in each direction,
# rsq=v1**2+v2**2 see if they are in the unit circle,
# if(rsq.ge.1..or.rsq.eq.0.)goto 1 and if they are not, try again.
# fac=sqrt(-2.*log(rsq)/rsq) Now make the Box-Muller transformation to get
# two normal deviates. Return one and save
# the other for next time.
# gset=v1*fac
# gasdev=v2*fac
# iset=1 Set flag.
# else We have an extra deviate handy,
# gasdev=gset so return it,
# iset=0 and unset the flag.
# endif
# return
# END