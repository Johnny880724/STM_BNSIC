# STM_BNSIC
Solving coefficient Poisson equation for binary neutron stars on irregular domain.

## Run a demo simulation
The simplest way to run the code is to run it directly in a python environment:
```bash
$ python3
```
The code requires libraries including `numpy`, `matplotlib`, `scipy`

To try a demo code, run the `main.py` file:
```python
python3 main.py
```

In the demo code, first create a inputconfig class that initializes all inputs:
```python
test_inputconfig = inputconfig()
```

Call the source term method in the `source_term_method.py` file
```python
stm.stm_coef_Neumann(test_inputconfig)
```

Use `plot1d_error` to make a plot of the relative error along the x-axis
or use `plot2d_error` to make a comparison plot of the result and the theory

## Changing the configuration
The configuration is initialized in the file `main.py` in the class `inputconfig`
The code requires
1. `N_grid` : the size of the grid
1. `maxIt_` : the maximum iteration allowed for the source term method
3. `it_multiple_` : the maximum iteration multiple (multiply by `N_grid ^ 2`) allowed for the jacobi iteration method.
4. `eta_` : the minimum convergence rate for termination
5. `rlx_` : the relaxation constant
6. `theory_` : the theoretical value for Phi
7. `boundary_` : the boundary condition for Phi_n
8. `S_zeta_` : the source (right hand side) for the coefficient Poisson equation
9. `rho_` : the density that defines the level set
10. `num_grid_dr` : the number of grid points for the separation
11. `zeta_` : the coefficient for the Poisson equation


## Details about the files
1. The file `main.py` contains an inputconfig class for configuring the input.
2. The file `source_term_method.py` contains the source term method that solves the coefficient Poisson equation.
3. The file `mesh_helper_functions.py` contains helper function of vector calculus and level set operations for `source_term_method.py`.
4. The file `mesh_helper_functions_3d.py` contains helper function of vector calculus and level set operations in 3d. It is currently only helpful for plotting purposes.