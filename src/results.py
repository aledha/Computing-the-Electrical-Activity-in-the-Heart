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
    
    def I_stim(x, t, lam, M):
        return 8 * ufl.pi**2 * lam/(1+lam) * ufl.sin(t) * ufl.cos(2*ufl.pi*x[0]) * ufl.cos(2*ufl.pi*x[1])
    
    pde = monodomain.PDESolver(h, dt, theta, C_m = 1.0, chi = 1.0, M = 1, lam = 1)

    N = int(np.ceil(1/h))
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)
    pde.set_mesh(domain, lagrange_order)
    pde.initialize_vn(initial_v)

    pde.set_stimulus(I_stim)
    pde.setup_solver()

    sn = pde.interpolate_func(initial_s)
    scheme = "forward_explicit_euler" if theta > 1 else "theta_rule"
    num_nodes = len(pde.vn.x.array)
    ode = monodomain.ODESolver(odefile="simple", scheme=scheme, num_nodes=num_nodes, initial_states=[pde.vn, sn], state_names=["v", "s"])

    solver = monodomain.MonodomainSolver(pde, ode)
    vn, x, t = solver.solve(T=1)
    v_ex = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)

    comm = vn.function_space.mesh.comm
    error = fem.form(ufl.sqrt((vn - v_ex)**2) * ufl.dx) # L2 error
    E = comm.allreduce(fem.assemble_scalar(error), MPI.SUM)

    return E

def spatial_convergence_plot(hs, dt, theta, lagrange_order):
    num_spatial = len(hs)
    errors = np.zeros(num_spatial)

    for i_space in range(num_spatial):
        errors[i_space] = simple_error(hs[i_space], dt, theta, lagrange_order)
    
    order = (np.log(errors[-1]) - np.log(errors[-2])) / (np.log(hs[-1]) - np.log(hs[-2]))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.loglog(hs, errors, '-o', label = 'd = {:.3f}  '.format(dt)+ 'p = {:.2f}'.format(order))
    ax.set_xlabel(r'$h$')
    ax.set_ylabel('Error')
    ax.set_title(f'L2 error for Lagrange order {lagrange_order}. ' + r'$\theta=$' + f'{theta}')
    ax.legend()
    ax.set_xticks(hs, hs)
    ax.minorticks_off()
    fig.show()

def temporal_convergence_plot(h, dts, theta, lagrange_order):
    num_temporal = len(dts)
    errors = np.zeros(num_temporal)

    for i_time in range(num_temporal):
        errors[i_time] = simple_error(h, dts[i_time], theta, lagrange_order)
    
    order = (np.log(errors[-1]) - np.log(errors[-2])) / (np.log(dts[-1]) - np.log(dts[-2]))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.loglog(dts, errors, '-o', label = 'h = {:.3f}  '.format(h)+ 'p = {:.2f}'.format(order))
    ax.set_xlabel(r'$dt$')
    ax.set_ylabel('Error')
    ax.set_title(f'L2 error for Lagrange order {lagrange_order}. ' + r'$\theta=$' + f'{theta}')
    ax.legend()
    ax.set_xticks(dts, dts)
    ax.minorticks_off()
    fig.show()
    
def dual_convergence_plot(hs, dts, theta, lagrange_order):
    num_spatial = len(hs)
    num_temporal = len(dts)

    errors = np.zeros((num_temporal, num_spatial))

    for i_time in range(num_temporal):
        for i_space in range(num_spatial):
            errors[i_time, i_space] = simple_error(hs[i_space], dts[i_time], theta, lagrange_order)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    fig.suptitle(f'L2 error for Lagrange order {lagrange_order}. ' + r'$\theta=$' + f'{theta}')
    
    for i_time in range(len(dts)):
        order = (np.log(errors[i_time,-1]) - np.log(errors[i_time,-2])) / (np.log(hs[-1]) - np.log(hs[-2]))

        ax1.loglog(hs, errors[i_time], '-o', label = 'dt = {:.3f}  '.format(dts[i_time])+ 'p = {:.2f}'.format(order))
    ax1.set_xlabel(r'$h$')
    ax1.set_ylabel('Error')
    ax1.set_title(r'Error as a function of $h$ for different d$t$.')
    ax1.legend()
    ax1.set_xticks(hs, hs)
    ax1.minorticks_off()

    for i_space in range(len(hs)):
        order = (np.log(errors[-1, i_space]) - np.log(errors[-2, i_space])) / (np.log(dts[-1]) - np.log(dts[-2]))
        ax2.loglog(dts, errors[:,i_space], '-o', label = 'h = {:.3f}  '.format(hs[i_space])+ 'p = {:.2f}'.format(order))
    ax2.set_xlabel(r'd$t$')
    ax2.set_ylabel('Error')
    ax2.set_title(r'Error as a function of d$t$ for different $h$.')
    ax2.legend()
    ax2.set_xticks(dts, dts)
    ax2.minorticks_off()
    fig.show()

def convergence_plot(hs, dts, theta, lagrange_order):
    if isinstance(hs, float):
        temporal_convergence_plot(hs, dts, theta, lagrange_order)
    elif isinstance(dts, float):
        spatial_convergence_plot(hs, dts, theta, lagrange_order)
    else:
        dual_convergence_plot(hs, dts, theta, lagrange_order)