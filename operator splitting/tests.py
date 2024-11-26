import monodomain
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
import ufl
from mpi4py import MPI


def simple_error(h, dt, theta, lagrange_order):

    def initial_v(x):
        return 0*x[0]

    def initial_s(x):
        return -np.cos(2*np.pi * x[0]) * np.cos(2*np.pi * x[1])
    
    pde = monodomain.PDESolver(h, dt, theta, M = 1, lam = 1)

    N = int(np.ceil(1/h))
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)
    pde.set_mesh(domain, lagrange_order)
    pde.initialize_vn(initial_v)
    I_stim = 8 * ufl.pi**2 * pde.lam/(1+pde.lam) * ufl.sin(pde.t) * ufl.cos(2*ufl.pi*pde.x[0]) * ufl.cos(2*ufl.pi*pde.x[1])

    pde.set_stimulus(I_stim)
    pde.setup_solver()

    sn = pde.interpolate_func(initial_s)
    ode = monodomain.ODESolver(odefile="simple", scheme="forward_explicit_euler", initial_v=pde.vn, initial_states=[sn], state_names=["s"])

    solver = monodomain.MonodomainSolver(pde, ode)
    vn, x, t = solver.solve(T=1)
    v_ex = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)

    comm = vn.function_space.mesh.comm
    error = fem.form(ufl.sqrt((vn - v_ex)**2) * ufl.dx) # L2 error
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))

    return E

def convergence_plot(num_spatial, num_temporal, start_spatial, start_temporal, theta, lagrange_order):
    hs = [1/(2**i) for i in range(start_spatial+num_spatial, start_spatial, -1)]
    dts = [1/(2**i) for i in range(start_temporal+num_temporal, start_temporal, -1)]

    errors = np.zeros((num_temporal, num_spatial))

    for itime in range(num_temporal):
        for ispace in range(num_spatial):
            errors[itime, ispace] = simple_error(hs[ispace], dts[itime], theta, lagrange_order)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5), sharey=True)
    fig.suptitle(f'L2 error for Lagrange order {lagrange_order}. Order of convergence ' + r'$p$. $\theta=$' + f'{theta}')
    
    for itime in range(len(dts)):
        order = np.polyfit(np.log(hs),np.log(errors[itime]), 1)[0]
        ax1.loglog(hs, errors[itime], '-o', label = 'dt = {:.3f}  '.format(dts[itime])+ 'p = {:.2f}'.format(order))
    ax1.set_xlabel(r'$h$')
    ax1.set_ylabel('Error')
    ax1.set_title(r'Error as a function of $h$ for different d$t$.')
    ax1.legend()

    for ispace in range(len(hs)):
        order = np.polyfit(np.log(dts),np.log(errors[:,ispace]), 1)[0]
        ax2.loglog(dts, errors[:,ispace], '-o', label = 'h = {:.3f}  '.format(hs[ispace])+ 'p = {:.2f}'.format(order))
    ax2.set_xlabel(r'd$t$')
    ax2.set_ylabel('Error')
    ax2.set_title(r'Error as a function of d$t$ for different $h$.')
    ax2.legend()

    fig.show()