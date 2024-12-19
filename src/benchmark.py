import ufl.tensors
import monodomain
import numpy as np
from dolfinx import mesh
import ufl
from mpi4py import MPI
from pint import UnitRegistry
import json
ureg = UnitRegistry()

initial_states = {
        "V": -85.23,  # mV
        "Xr1": 0.00621,
        "Xr2": 0.4712,
        "Xs": 0.0095,
        "m": 0.00172,
        "h": 0.7444,
        "j": 0.7045,
        "d": 3.373e-05,
        "f": 0.7888,
        "f2": 0.9755,
        "fCass": 0.9953,
        "s": 0.999998,
        "r": 2.42e-08,
        "Ca_i": 0.000126,  # millimolar
        "R_prime": 0.9073,
        "Ca_SR": 3.64,  # millimolar
        "Ca_ss": 0.00036,  # millimolar
        "Na_i": 8.604,  # millimolar
        "K_i": 136.89,  # millimolar
    }

chi = 1400 * ureg("1/cm")
C_m = 1 * ureg("uF/cm**2")

intra_trans, extra_trans = 0.019 * ureg("S/m"), 0.24 * ureg("S/m") # Sm^-1
intra_long, extra_long = 0.17 * ureg("S/m"), 0.62 * ureg("S/m")
trans_conductivity = intra_trans * extra_trans / (intra_trans + extra_trans)
long_conductivity = intra_long * extra_long / (intra_long + extra_long)

trans_conductivity_scaled = (trans_conductivity / (chi * C_m)).to("mm**2/ms").magnitude
long_conductivity_scaled = (long_conductivity / (chi * C_m)).to("mm**2/ms").magnitude
M = ufl.tensors.as_tensor(np.diag([trans_conductivity_scaled, trans_conductivity_scaled, long_conductivity_scaled]))

def benchmark(h, dt, theta, lagrange_order):
    pde = monodomain.PDESolver(h, dt, theta, M)
    mesh_comm = MPI.COMM_WORLD

    stim_amplitude = 50000 * ureg("uA/cm**3")
    amplitude_magnitude = (stim_amplitude / (C_m*chi)).to("mV/ms").magnitude

    def I_stim(x, t):
        condition = ufl.And(ufl.And(x[0]<=1.5, x[1]<=1.5), ufl.And(x[2]<=1.5, t <= 2))
        return ufl.conditional(condition, amplitude_magnitude, 0)

    Lx, Ly, Lz = 3, 7, 20 #mm
    domain = mesh.create_box(mesh_comm, [[0,0,0], [Lx,Ly,Lz]], n = [int(Lx/h), int(Ly/h), int(Lz/h)])

    pde.set_mesh(domain, lagrange_order)
    pde.set_stimulus(I_stim)
    pde.setup_solver()

    num_nodes = pde.V.dofmap.index_map.size_global

    ode = monodomain.ODESolver(odefile="tentusscher_panfilov_2006_epi_cell", 
                            scheme="generalized_rush_larsen", 
                            num_nodes=num_nodes, v_name="V",
                            initial_states=initial_states)
    ode.set_param("stim_amplitude", 0)

    solver = monodomain.MonodomainSolver(pde, ode)
    points = np.array([[0,0,0],
                        [0,Ly,0],
                        [0,0,Lz],
                        [0,Ly,Lz],
                        [Lx,0,0],
                        [Lx,Ly,0],
                        [Lx,0,Lz],
                        [Lx,Ly,Lz],
                        [Lx/2,Ly/2,Lz/2]
                        ])
    activation_times = solver.solve_activation_times(points, T=100)
    return activation_times

def write_to_csv(h, dt, theta=1, lagrange_order=1):
    activation_times = benchmark(h, dt, theta, lagrange_order)
    activation_times_dict = {f"P{i+1}": round(activation_times[i], 10) for i in range(len(activation_times))}
    with open('activation_times.txt', 'a') as file:
        file.write(json.dumps({f'h={h}, dt={dt}': activation_times_dict}))
        file.write('\n')

hs = [0.5, 0.2]
dts = [0.01, 0.005, 0.01]

for dt in dts:
    for h in hs:
        write_to_csv(h, dt)
        print(f"Completed h={h}, dt={dt}")
