from numpy import *
cimport numpy as np

def cwave_prop( np.ndarray[double,ndim=2] u_prev, np.ndarray[double,ndim=2] u, np.ndarray[double,ndim=2] u_next, int Lx , int Ly , double t_end):
	
	cdef double t = 0
	cdef double t_old = 0
	cdef int i,j
	cdef double c = 1
	cdef double dx = 1
	cdef double dy = 1
	cdef double dt = (1/(c))*(1/(sqrt(1/dx**2 + 1/dy**2)))
	
	while t<t_end:
		t_old=t; t +=dt

		#wave steps through time and space

		for i in xrange(1,Lx-1):
			for j in xrange(1,Ly-1):
				u_next[i,j] = - u_prev[i,j] + 2*u[i,j] + \
	                	((c*dt/dx)**2)*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) + \
				((c*dt/dx)**2)*(u[i,j-1] - 2*u[i,j] + u[i,j+1])

		#set boundary conditions of grid to 0

		for j in xrange(Ly): u_next[0,j] = 0
		for i in xrange(Lx): u_next[i,0] = 0
		for j in xrange(Ly): u_next[Lx-1,j] = 0
		for i in xrange(Lx): u_next[i,Ly-1] = 0

		#set prev time step equal to current one
		for i in xrange(Lx):
			for j in xrange(Ly):		
				u_prev[i,j] = u[i,j]; 
				u[i,j] = u_next[i,j]; 


	
	return u
