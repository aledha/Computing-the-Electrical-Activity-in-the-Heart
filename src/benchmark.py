import ufl.tensors
import monodomain
import numpy as np
from dolfinx import mesh
import ufl
from mpi4py import MPI
from pint import UnitRegistry
import json
import matplotlib.pyplot as plt
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
    pde.setup_solver("CG")

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
    line = np.linspace([0,0,0], [Lx, Ly, Lz], 50)
    time_points, time_line = solver.solve_activation_times(points, line, T=100)
    return time_points, time_line

def write_to_csv(h, dt, theta=1, lagrange_order=1):
    time_points, time_line = benchmark(h, dt, theta, lagrange_order)
    time_points_dict = {f"P{i+1}": round(time_points[i], 10) for i in range(len(time_points))}
    with open('activation_times.txt', 'a') as file:
        file.write(json.dumps({f'h={h}, dt={dt}': time_points_dict}))
        file.write('\n')
    with open('activation_times_line.csv', 'a') as file:
        file.write(f'h={h}, dt={dt}\n')
        np.savetxt(file, time_line, delimiter=',')
        file.write('\n')

def read_file_and_plot(h05, h02, h01):
    Lx, Ly, Lz = 3, 7, 20 #mm
    dist = np.linspace(0, np.sqrt(Lx**2 + Ly**2 + Lz**2), len(h05))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(dist, h05, '-b')
    ax.plot(dist, h02, '-g')
    ax.plot(dist, h01, '-r')
    ax.set_ylabel('activation time (ms)')
    ax.set_xlabel('distance (mm)')
    ax.set_ylim(0, 150)
    ax.grid(True)
    ax.set_title(f'Activation time along line')
    fig.savefig('activation_line', bbox_inches='tight')



hs = [0.1]
dts = [0.05, 0.01, 0.005]
discs = [(0.5,0.005), (0.2,0.005), (0.5,0.005)]
h05 = np.array([1.219999999999999973e+00, 1.215000000000000080e+00, 1.235000000000000098e+00, 1.570000000000000062e+00, 1.919999999999999929e+00, 2.660000000000000142e+00, 3.370000000000000107e+00, 4.160000000000000142e+00, 5.084999999999999964e+00, 5.884999999999999787e+00, 6.054999999999999716e+00, 6.855000000000000426e+00, 1.001500000000000057e+01, 1.050500000000000078e+01, 1.094500000000000028e+01, 1.496000000000000085e+01, 1.508999999999999986e+01, 1.549000000000000021e+01, 1.596000000000000085e+01, 2.017999999999999972e+01, 2.046000000000000085e+01, 2.060999999999999943e+01, 2.487999999999999901e+01, 2.524500000000000099e+01, 2.555999999999999872e+01, 2.593499999999999872e+01, 3.021999999999999886e+01, 3.037500000000000000e+01, 3.069999999999999929e+01, 3.505499999999999972e+01, 3.538499999999999801e+01, 3.555499999999999972e+01, 3.579500000000000171e+01, 3.993999999999999773e+01, 4.023499999999999943e+01, 4.035499999999999687e+01, 4.415500000000000114e+01, 4.435999999999999943e+01, 4.461500000000000199e+01, 4.505499999999999972e+01, 4.885000000000000142e+01, 4.910000000000000142e+01, 4.919500000000000028e+01, 5.274000000000000199e+01, 5.311999999999999744e+01, 5.347500000000000142e+01, 5.375999999999999801e+01, 5.553499999999999659e+01, 5.564999999999999858e+01, 5.578499999999999659e+01])
h02 = np.array([1.229999999999999982e+00, 1.229999999999999982e+00, 1.250000000000000000e+00, 1.395000000000000018e+00, 1.915000000000000036e+00, 2.595000000000000195e+00, 3.265000000000000124e+00, 3.924999999999999822e+00, 4.594999999999999751e+00, 5.294999999999999929e+00, 5.969999999999999751e+00, 6.674999999999999822e+00, 7.429999999999999716e+00, 8.164999999999999147e+00, 8.869999999999999218e+00, 9.685000000000000497e+00, 1.053999999999999915e+01, 1.131000000000000050e+01, 1.204499999999999993e+01, 1.297499999999999964e+01, 1.388000000000000078e+01, 1.446499999999999986e+01, 1.543999999999999950e+01, 1.641499999999999915e+01, 1.707499999999999929e+01, 1.792500000000000071e+01, 1.893499999999999872e+01, 1.994000000000000128e+01, 2.044500000000000028e+01, 2.147500000000000142e+01, 2.251000000000000156e+01, 2.307499999999999929e+01, 2.401000000000000156e+01, 2.505499999999999972e+01, 2.607499999999999929e+01, 2.652499999999999858e+01, 2.757499999999999929e+01, 2.863500000000000156e+01, 2.913500000000000156e+01, 3.007999999999999829e+01, 3.114000000000000057e+01, 3.211999999999999744e+01, 3.256499999999999773e+01, 3.360999999999999943e+01, 3.461500000000000199e+01, 3.510499999999999687e+01, 3.604500000000000171e+01, 3.704999999999999716e+01, 3.753000000000000114e+01, 3.786999999999999744e+01])
h01 = np.array([1.229999999999999982e+00, 1.229999999999999982e+00, 1.254999999999999893e+00, 1.399999999999999911e+00, 1.935000000000000053e+00, 2.595000000000000195e+00, 3.254999999999999893e+00, 3.924999999999999822e+00, 4.620000000000000107e+00, 5.334999999999999964e+00, 6.080000000000000071e+00, 6.839999999999999858e+00, 7.615000000000000213e+00, 8.410000000000000142e+00, 9.210000000000000853e+00, 1.004499999999999993e+01, 1.084999999999999964e+01, 1.169500000000000028e+01, 1.251500000000000057e+01, 1.333999999999999986e+01, 1.419500000000000028e+01, 1.499499999999999922e+01, 1.585999999999999943e+01, 1.668499999999999872e+01, 1.753500000000000014e+01, 1.837500000000000000e+01, 1.921999999999999886e+01, 2.008500000000000085e+01, 2.089499999999999957e+01, 2.178999999999999915e+01, 2.260999999999999943e+01, 2.347500000000000142e+01, 2.432000000000000028e+01, 2.514999999999999858e+01, 2.601500000000000057e+01, 2.682499999999999929e+01, 2.769999999999999929e+01, 2.853000000000000114e+01, 2.940500000000000114e+01, 3.024500000000000099e+01, 3.108999999999999986e+01, 3.195499999999999829e+01, 3.275500000000000256e+01, 3.365500000000000114e+01, 3.445499999999999829e+01, 3.531499999999999773e+01, 3.614000000000000057e+01, 3.695000000000000284e+01, 3.773499999999999943e+01, 3.811999999999999744e+01])

read_file_and_plot(h05, h02, h01)
# for h in hs:
#     for dt in dts:
#         write_to_csv(h, dt)
#         print(f"Completed h={h}, dt={dt}")
