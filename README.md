# BNS_IC_CUR
Solving coefficient Poisson equation for Newtonian equation of neutron stars on irregular domain.

## Run a demo simulation
For me, I think the simplest way to run the code is to run it directly in a python environment:
```bash
$ python3
```
To try a demo code with grid size = 100 for test case 1, run the `main.py` file:
```python
exec(open("main.py").read()) 
```

After the solver is finished, output the result for the variable `u_result` (here is an example with matplotlib):
```python
import matplotlib.pyplot as plt
plt.matshow(u_result)
plt.colorbar()
plt.savefig("result.png")
```

Similarly, output the theoretical result using the variable `theory`
```python
plt.matshow(theory)
plt.colorbar()
plt.savefig("theory.png")
```

## Changing the variable
1. To try other grid sizes, change the list `grid_size_array` in the file `main.py`. For example, to generate a convergence test
    for grid size = 32, 64, 128, 256, change the variables in the list and run the file again.
```python
grid_size_array = [32,64,128,256]
```
Plot the error - grid size relation with
```python
import matplotlib.pyplot as plt
plt.plot(grid_size_array, result_array)
plt.savefig("convergence.png")
```
2. To try other test cases, change the funciton `test_case.setup_equations_1(max(0.01*grid_size*h,h))` in `line 38` to other functions in `test_case.py`.
    For example, to try test case 4, change the function on line 38 and run the file again:
```python
test_case.setup_equations_4(max(0.01*grid_size*h,h))
```

## Details about the files
1. The file `main.py` only serves as a demo environment for the solver.
2. The file `jacobi_newtonian_solver.py` contains the source term method that solves the Newtonian equation.
3. The file `mesh_helper_functions.py` contains helper function of vector calculus and level set operations for `jacobi_newtonian_solver.py`.
4. The file `test_cases.py` contains 5 test cases for testing purposes. The current demo in `main.py` uses test case 1.