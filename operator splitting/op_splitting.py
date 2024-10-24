from dolfinx import fem, mesh, io
import dolfinx.fem.petsc as petsc
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import ufl
import matplotlib.pyplot as plt
import simple as model

lam = 1
T = 1.0  # Final time

def v_exact(t):
    return lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.sin(t)

def initial_s(x):
    return -np.cos(2*np.pi * x[0]) * np.cos(2*np.pi * x[1])

def initial_v(x):
    return 0*x[0]

def forward_euler_step(init, t, dt):
    s, v = model.forward_explicit_euler([init[1], init[0]], t, dt, 0)
    return v, s

def solve(h, dt, theta, lagrangeOrder, xdmfTitle=None):
    gamma = dt * lam / (1+lam)
    N = int(np.ceil(1/h))

    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", lagrangeOrder))
    t = fem.Constant(domain, 0.0)
    x = ufl.SpatialCoordinate(domain)

    I_stim = 8 * ufl.pi**2 * lam/(1+lam) * ufl.sin(t) * ufl.cos(2*ufl.pi*x[0]) * ufl.cos(2*ufl.pi*x[1])
    
    vn = fem.Function(V)
    vn.name = "vn"
    vn.interpolate(initial_v)

    sn = fem.Function(V)
    sn.name = "sn"
    sn.interpolate(initial_s)

    vntheta = fem.Function(V)
    sntheta = fem.Function(V)

    v = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)
    dx = ufl.dx(domain=domain)
    a = phi * v * dx + gamma * theta * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
    L = phi * (vntheta + dt * I_stim) * dx - gamma * (1-theta) * ufl.dot(ufl.grad(phi), ufl.grad(vntheta)) * dx
    compiled_a = fem.form(a)
    A = petsc.assemble_matrix(compiled_a)
    A.assemble()

    compiled_L = fem.form(L)
    b = fem.Function(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    if xdmfTitle:
        xdmf = io.XDMFFile(domain.comm, xdmfTitle, "w")
        xdmf.write_mesh(domain)
        xdmf.write_function(vn, t)

    while t.value < T:
        # Step 1
        t.value += theta * dt
        vntheta.x.array[:], sntheta.x.array[:] = forward_euler_step([vn.x.array, sn.x.array], t.value, theta * dt)

        # Step 2
        b.x.array[:] = 0
        petsc.assemble_vector(b.vector, compiled_L)
        
        solver.solve(b.vector, vn.vector)
        vn.x.scatter_forward()

        # Step 3
        if theta < 1.0:
            t.value += (1-theta) * dt
            vn.x.array[:], sn.x.array[:] = forward_euler_step([vn.x.array.copy(), sntheta.x.array], t.value, (1-theta) * dt)
        else:
            sn.x.array[:] = sntheta.x.array.copy()
        
        if xdmfTitle:
            xdmf.write_function(vn, t.value)
    if xdmfTitle:
        xdmf.close()
    return vn, x, t

def error(h, dt, theta, lagrangeOrder):
    vn, x, t = solve(h, dt, theta, lagrangeOrder)
    v_ex = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)

    comm = vn.function_space.mesh.comm
    error = fem.form(ufl.sqrt((vn - v_ex)**2) * ufl.dx)
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    return E

def create_error_matrix(numSpatial, numTemporal, startSpatial, startTemporal, theta, lagrangeOrder):
    hs = [1/(2**i) for i in range(startSpatial+numSpatial, startSpatial, -1)]
    dts = [1/(2**i) for i in range(startTemporal+numTemporal, startTemporal, -1)]

    errors = np.zeros((numTemporal, numSpatial))

    for itime in range(numTemporal):
        for ispace in range(numSpatial):
            errors[itime, ispace] = error(hs[ispace], dts[itime], theta, lagrangeOrder)
    return errors, hs, dts

def analyse_error(numSpatial, numTemporal, startSpatial, startTemporal, theta, lagrangeOrder):
    errors, hs, dts = create_error_matrix(numSpatial, numTemporal, startSpatial, startTemporal, theta, lagrangeOrder)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5), sharey=True)
    fig.suptitle(f'L2 error for Lagrange order {lagrangeOrder}. Order of convergence ' + r'$p$. $\theta=$' + f'{theta}')
    
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