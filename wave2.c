#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#define n 100

void wave_prop(float** u_prev ,int Lx,int Ly,float** u ,int Lx2,int Ly2,float** u_next,int Lx3,int Ly3 ){


int dx=1;
int dy=1;
float c=1;
float dt =1;
int t_old=0;int t=0;int t_end=150;
int x[Lx];
int y[Ly];

for(int i=0;i<=99;i++){
		x[i]=i;
		y[i]=i;
	}

int row, columns;
for (int row=0; row<Lx-1; row++)
{
         printf("%d     ", x[row]);
    printf("\n");
 }
/*while(t<t_end){
	t_old=t; t +=dt;
	//the wave steps through time
	for (int i=1;i<99;i++){
		for (int j=1;j<99;j++){
	        	u_next[i][j] = - u_prev[i][j] + 2*u[i][j] + \
	                	(c*dt/dx)*(c*dt/dx)*u[i-1][j] - 2*u[i][j] + u[i+1][j] + \
				(c*dt/dx)*(c*dt/dx)*u[i][j-1] - 2*u[i][j] + u[i][j+1];
				}
			 }

	//set boundary conditions to 0

	for (int j=0;j<=99;j++){ u_next[0][j] = 0;}
	for (int i=0;i<=99;i++){ u_next[i][0] = 0;}
	for (int j=0;j<=99;j++){ u_next[Lx-1][j] = 0;}
	for (int i=0;i<=99;i++){ u_next[i][Ly-1] = 0;}

	//memcpy(dest, src, sizeof (mytype) * rows * coloumns);
	memcpy(u_prev, u, sizeof (float) * Lx * Ly);
	memcpy(u, u_next, sizeof (float) * Lx * Ly);

	}*/
}

