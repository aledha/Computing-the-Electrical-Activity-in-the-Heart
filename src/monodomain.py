from dolfinx import fem, mesh, io
import dolfinx.fem.petsc as petsc
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import ufl
import gotranx
from pathlib import Path
from dataclasses import dataclass
import importlib
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

def translateODE(odeFileName, schemes):
    odeFolder = str(Path.cwd().parent) + "/odes/"
    model_path = Path(odeFolder + odeFileName + ".py")
    if not model_path.is_file():
        ode = gotranx.load_ode(odeFolder + odeFileName + ".ode")
        code = gotranx.cli.gotran2py.get_code(ode, schemes)
        model_path.write_text(code)
    else:
        print("ODE already translated")

@dataclass
class PDESolver:
    h: float
    dt: float
    theta: float
    C_m: ufl.Constant
    chi: ufl.Constant
    M: ufl.Constant
    lam: ufl.Constant

    def __post_init__(self)->None:
        self.alpha = self.dt / (self.chi * self.C_m) * self.lam / (1+self.lam)
        self.beta = self.dt / self.C_m
        self.N = int(np.ceil(1/self.h))

    def set_mesh(self, domain, lagrange_order) -> None:
        self.domain = domain

        self.V = fem.functionspace(domain, ("Lagrange", lagrange_order))
        self.t = fem.Constant(domain, 0.0)
        self.x = ufl.SpatialCoordinate(domain)

        self.vn = fem.Function(self.V)
        self.vn.name = "vn"
    
    def initialize_vn(self, initial_v):
        self.vn.interpolate(initial_v)
    
    def interpolate_func(self, func):
        fem_func = fem.Function(self.V)
        fem_func.interpolate(func)
        return fem_func

    def set_stimulus(self, I_stim):
        self.I_stim = I_stim(self.x, self.t, self.lam, self.M)
    
    def setup_solver(self):
        v = ufl.TrialFunction(self.V)
        phi = ufl.TestFunction(self.V)
        dx = ufl.dx(domain=self.domain)
        a = phi * v * dx + self.alpha * self.theta * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
        L = phi * (self.vn + self.beta * self.I_stim) * dx - self.alpha * (1-self.theta) * ufl.dot(ufl.grad(phi), self.M * ufl.grad(self.vn)) * dx
        compiled_a = fem.form(a)
        A = petsc.assemble_matrix(compiled_a)
        A.assemble()

        self.compiled_L = fem.form(L)
        self.b = fem.Function(self.V)
        
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
    
    def solve_pde_step(self):
        self.b.x.array[:] = 0
        petsc.assemble_vector(self.b.vector, self.compiled_L)
        
        self.solver.solve(self.b.vector, self.vn.vector)
        self.vn.x.scatter_forward()


class ODESolver:
    def __init__(self, odefile, scheme, num_nodes, initial_states = [], state_names = [], v_name = "v"):
        try:
            self.model = importlib.import_module(f"odes.{odefile}")
        except ImportError as e:
            raise ImportError(f"Failed to import {odefile}: {e}")
        
        init = self.model.init_state_values()
        self.states = np.tile(init, (num_nodes, 1)).T
        
        self.v_index = self.model.state_index(v_name)

        for state, name in zip(initial_states, state_names):
            state_index = self.model.state_index(name)
            if isinstance(state, np.ndarray):
                self.states[state_index, :] = state
            else:
                self.states[state_index, :] = state.x.array

        self.params = self.model.init_parameter_values()
        self.odesolver = getattr(self.model, scheme)

    def set_param(self, name, value):
        param_index = self.model.parameter_index(name)
        self.params[param_index] = value
    
    def solve_ode_step(self, t, dt):
        self.states[:] = self.odesolver(self.states, t, dt, self.params)

    def update_vn_array(self, vn):
        self.states[self.v_index, :] = vn.x.array[:]

    def get_vn(self):
        return self.states[self.v_index,:]

@dataclass
class MonodomainSolver:
    pde: PDESolver
    ode: ODESolver
    def __post_init__(self):
        self.t = self.pde.t
        self.dt = self.pde.dt
        self.theta = self.pde.theta

    def step(self):
        # Step 1
        self.ode.solve_ode_step(self.t.value, self.theta*self.dt)
        self.t.value += self.theta * self.dt

        # Step 2
        self.pde.vn.x.array[:] = self.ode.get_vn()
        self.pde.solve_pde_step()
        self.ode.update_vn_array(self.pde.vn)

        # Step 3
        if self.theta < 1.0:
            self.ode.solve_ode_step(self.t.value, (1 - self.theta)*self.dt)
            self.t.value += (1 - self.theta) * self.dt

        self.pde.vn.x.array[:] = self.ode.get_vn()

    def solve(self, T, vtx_title=None):
        if vtx_title:
            vtx = io.VTXWriter(MPI.COMM_WORLD, vtx_title + ".bp", [self.pde.vn], engine="BP4")
        while self.t.value < T + self.dt:
            self.step()
            if vtx_title:
                vtx.write(self.t.value)

        return self.pde.vn, self.pde.x, self.t


        



