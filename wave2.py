'''George Lees Jr.
2D Wave pde '''

from numpy import *
import numpy as np
import matplotlib.pyplot as plt 
import cwave2
import sys
np.set_printoptions(threshold=np.nan)

#declare variables
#need 3 arrays u_prev is for previous time step due to time derivative
t_end = double(sys.argv[1])			#How long the while loop goes for
Lx = int(sys.argv[2])				#Length of x and y dims of grid
Ly = int(sys.argv[3])				#Length of x and y dims of grid
						
dx=dy = 1						#derivative of x and y respectively
x=y = np.array(xrange(Lx))				#linspace to set the initial condition of wave
u_prev=np.ndarray(shape=(Lx,Ly), dtype=np.double)	#u_prev 2D grid for previous time step needed bc of time derivative dt/dy
u=np.ndarray(shape=(Lx,Ly), dtype=np.double)		#u 2D grid
u_next=np.ndarray(shape=(Lx,Ly), dtype=np.double)	#u_next for advancing the time step #also these are all numpy ndarrays
c = 1 							#setting constant velocity of the wave
dt = (1/float(c))*(1/sqrt(1/dx**2 + 1/dy**2))		#we have to set dt specifically to this or numerical approx will fail!
print dt

#set Initial Conditions and Boundary Points
#I(x) is initial shape of the wave

def I(x,y): return exp(-(x-Lx/2.0)**2/2.0 -(y-Ly/2.0)**2/2.0)

#set up initial wave shape

for i in xrange(Lx):
	for j in xrange(Ly):
	    u[i,j] = I(x[i],y[j])


#set up previous time step array

for i in xrange(1,Lx-1):
	for j in xrange(1,Ly-1):
	    	u_prev[i,j] = u[i,j] + 0.5*((c*dt/dx)**2)*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) + \
			0.5*((c*dt/dy)**2)*(u[i,j-1] - 2*u[i,j] + u[i,j+1]) 

#set boundary conditions to 0

for j in xrange(Ly): u_prev[0,j] = 0
for i in xrange(Lx): u_prev[i,0] = 0
for j in xrange(Ly): u_prev[Lx-1,j] = 0
for i in xrange(Lx): u_prev[i,Ly-1] = 0

#call C function from Python
cwave2.cwave_prop( u_prev , u , u_next, Lx, Ly, t_end )
#returned u (2D np.ndarray)

'''from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save(outfile,u)
fig = plt.figure()
plt.imshow(u,cmap=plt.cm.ocean)
plt.colorbar()'''

