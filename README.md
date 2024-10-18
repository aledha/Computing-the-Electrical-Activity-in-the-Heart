This is a repository for my master's thesis, which focuses on simulating the electrical activity in the heart using the finite element method, specifically the computational environment [dolfinx](https://github.com/FEniCS/dolfinx). 
I am writing in collaboration with Simula Research Laboratories, and my supervisors are Henrik Finsberg, Joakim Sundnes, and Jørgen Dokken.

My plan for this thesis is to
* Read and understand the various mathematical models used to simulate the electrical activity in the heart, starting with the monodomain model (done).
* Implement a finite element solver for a simple PDE in 2D (done, see [diffusion.py](diffusion/diffusion.py)).
* Familiarize myself with dolfinx by implementing a very similar PDE using the environment (done, see [diffusion_dolfinx.py](diffusion/diffusion_dolfinx.py)).
* Implement a simplified 2D monodomain solver using operator splitting and dolfinx (done, see [op_split.py](operator%20splitting/op_split.ipynb)).
* Analyse convergence of the simplified solver in both space and time (in progress, see [op_split_convergence.py](operator%20splitting/op_split_convergence.ipynb)).
* Implement a solver for the more complex Fitzhugh-Nagumo model using the ODE translator [gotranx](https://github.com/finsberg/gotranx).
* Expand the solver to accommodate 3D problems.
* Use my solver on the [Nieder benchmark](https://finsberg.github.io/fenics-beat/demos/niederer_benchmark.html).
