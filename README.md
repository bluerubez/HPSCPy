HPSCPy
=======

This project highlights Python's abilities as a glue language and proves that Python can be a
powerful tool for High Performance Scientific Computing. I solved the 2D wave PDE over a mesh 
grid to help showcase how to speed up a pure Python code. One implementation to speed up the 
program was exporting the main finite differencing loop to C compiled code via Cython. Another 
implementation was using mpi4py. A very useful animation was accomplished with Python after 
the results were returned showing how convienent it is to have a scripting language on top of 
the lower level or parallel performance modules.

