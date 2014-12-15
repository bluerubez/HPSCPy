'''George Lees Jr.
2D Wave pde '''

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

#declare variables
#need 3 arrays u_prev is for previous time step due to d/dt

Lx=Ly = (100)
dx=dy = 1
x=y = np.array(xrange(Lx))
u_prev=np.ndarray(shape=(Lx,Ly), dtype=np.double)
u=np.ndarray(shape=(Lx,Ly), dtype=np.double)
u_next=np.ndarray(shape=(Lx,Ly), dtype=np.double)
c = 1 #constant velocity
dt = (1/float(c))*(1/sqrt(1/dx**2 + 1/dy**2))
t_old=0;t=0;t_end=100

#set Initial Conditions and Boundary Points
#I(x) is initial shape of the wave
#f(x,t) is outside force that creates waves set =0

def I(x,y): return exp(-(x-Lx/2.0)**2/2.0 -(y-Ly/2.0)**2/2.0)
def f(x,t,y): return 0

#set up initial wave shape

for i in xrange(100):
	for j in xrange(100):
	    u[i,j] = I(x[i],y[j])


#copy initial wave shape for printing later

u1=u.copy()

#set up previous time step array

for i in xrange(1,99):
	for j in xrange(1,99):
	    	u_prev[i,j] = u[i,j] + 0.5*((c*dt/dx)**2)*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) + \
			0.5*((c*dt/dy)**2)*(u[i,j-1] - 2*u[i,j] + u[i,j+1]) + \
			dt*dt*f(x[i], y[j], t)



#set boundary conditions to 0

for j in xrange(100): u_prev[0,j] = 0
for i in xrange(100): u_prev[i,0] = 0
for j in xrange(100): u_prev[Lx-1,j] = 0
for i in xrange(100): u_prev[i,Ly-1] = 0



while t<t_end:
	t_old=t; t +=dt
	#the wave steps through time
	for i in xrange(1,99):
		for j in xrange(1,99):
	        	u_next[i,j] = - u_prev[i,j] + 2*u[i,j] + \
	                	((c*dt/dx)**2)*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) + \
				((c*dt/dx)**2)*(u[i,j-1] - 2*u[i,j] + u[i,j+1]) + \
	                	dt*dt*f(x[i], y[j], t_old)

	#set boundary conditions to 0

	for j in xrange(100): u_next[0,j] = 0
	for i in xrange(100): u_next[i,0] = 0
	for j in xrange(100): u_next[Lx-1,j] = 0
	for i in xrange(100): u_next[i,Ly-1] = 0

	#set prev time step equal to current one
	u_prev = u.copy(); u = u_next.copy(); 

print u
fig = plt.figure()
plt.imshow(u,cmap=plt.cm.ocean)
plt.colorbar()
plt.show()
