import dolfinx

import matplotlib as mpl
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 50
dt = T / num_steps  # time step size

nx, ny = 50,50

domain = mesh.create_rectangle(MPI.COMM_WORLD, np.array([-2,-2]), np.array([2,2]))
