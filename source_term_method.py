# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:09:48 2020

@author: Johnny Tsao
"""
"""
This file incorporates the source term method proposed by John Towers in his paper
A source term method for Poisson problems on irregular domains (2018)
https://doi.org/10.1016/j.jcp.2018.01.038

There are a few notation differences.
"""

import sys
import mesh_helper_functions as mhf
import mesh_helper_functions_3d as mhf3d
import numpy as np
import matplotlib.pyplot as plt

#adding to denominator to avoid 0/0 error
singular_null = mhf.singular_null

# Discretizations: source term method section 2.2

#discretization of Step, Delta functions
def I(phi):
    return mhf.D(phi)*phi

def J(phi):
    return 1./2 * mhf.D(phi)*phi**2

def K(phi):
    return 1./6 * mhf.D(phi)*phi**3

#Characteristic function of N1
##Chi is 1 if in N1
##       0 if not in N1
def Chi(phi):
    ret = np.zeros_like(phi)
    ret[:-1,:] += mhf.D(-phi[:-1,:]*phi[ 1:,:])
    ret[ 1:,:] += mhf.D(-phi[:-1,:]*phi[ 1:,:])
    ret[:,:-1] += mhf.D(-phi[:,:-1]*phi[ :,1:])
    ret[:, 1:] += mhf.D(-phi[:,:-1]*phi[ :,1:])
    return np.heaviside(ret,0)

# return the neighbor and the domain
def get_neighbor(domain):
    ret = np.zeros_like(domain)
    ret += domain
    ret[ 1:, :] += domain[:-1, :]
    ret[:-1, :] += domain[ 1:, :]
    ret[ :, 1:] += domain[ :,:-1]
    ret[ :,:-1] += domain[ :, 1:]
    return np.heaviside(ret,0)

# return N1
def get_N1(phi):
    return Chi(phi)

# return N2
def get_N2(phi):
    N1 = get_N1(phi)    
    return get_neighbor(N1)

#return N3
def get_N3(phi):
    N2 = get_N2(phi)
    return get_neighbor(N2)

# return the heaviside function discretized in the paper - formula 4
def H(phi_,h):
    J_mat = J(phi_)
    K_mat = K(phi_)
    first_term_1 = mhf.laplace(J_mat,h,h) / (mhf.abs_grad(phi_,h,h)**2 + singular_null)
    first_term_2 = -(mhf.laplace(K_mat,h,h) - J_mat*mhf.laplace(phi_,h,h))*mhf.laplace(phi_,h,h) / (mhf.abs_grad(phi_,h,h)**4 + singular_null)
    first_term = first_term_1 + first_term_2
    second_term = mhf.D(phi_)
    return Chi(phi_) * first_term + (1-Chi(phi_)) * second_term

# return the delta function discretized in the paper -formula 4
def delta(phi_,h):
    I_mat = I(phi_)
    J_mat = J(phi_)
    first_term = mhf.laplace(I_mat,h,h) / (mhf.abs_grad(phi_,h,h)**2 + singular_null)
    first_term -= (mhf.laplace(J_mat,h,h) - I_mat*mhf.laplace(phi_,h,h))*mhf.laplace(phi_,h,h) / (mhf.abs_grad(phi_,h,h)**4 + singular_null)
    return Chi(phi_) * first_term
    
# return the source term discretized in the paper
def get_source(a, b, phi_,f_mat_, h_):
    H_h_mat = H(phi_,h_)
    H_mat = mhf.D(phi_)
    term1 = mhf.laplace(b * H_mat,h_,h_)
    term2 = - H_h_mat * mhf.laplace(b, h_, h_)
    term3 = - (a - mhf.grad_n_n(b,phi_,h_,h_)) * delta(phi_, h_) * mhf.abs_grad(phi_,h_,h_)
    term4 = H_h_mat * f_mat_
    S_mat = term1 + term2 + term3 + term4
    # mhf3d.plot2d(term1,"1")
    return S_mat



#projection algorithm - source term method section 2.3
def projection(mesh_, phi_):
    xmesh, ymesh = mesh_
    h = xmesh[0,1]-xmesh[0,0]
    phi_abs_grad = mhf.abs_grad(phi_,h,h)
    grad_tup = mhf.grad(phi_,h,h)
    nx = -grad_tup[0] / (phi_abs_grad + singular_null)
    ny = -grad_tup[1] / (phi_abs_grad + singular_null)
    xp = xmesh + nx * phi_ / (phi_abs_grad + singular_null)
    yp = ymesh + ny * phi_ / (phi_abs_grad + singular_null)
    
    # A regularizing function to avoid the projection go out of bounds.
    def regularize(mesh_p, mesh):
        reg_min = np.min(mesh)
        reg_max = np.max(mesh)
        too_large = mhf3d.get_frame_n(mesh_p - reg_max)
        too_small = mhf3d.get_frame_n(reg_min - mesh_p)
        mesh_reg = mesh*(1-too_large)*(1-too_small) + too_large * reg_max + too_small * reg_min
        return mesh_reg
    
    xp_reg = regularize(xp,xmesh)
    yp_reg = regularize(yp,ymesh)
    
    return xp_reg, yp_reg

# quadrature extrapolation algorithm - source term method section 2.4
# (extrapolation may not work if the grid size is too small)
def extrapolation(val_, target_, eligible_):
    val_extpl = val_ * eligible_
    tau_0 = np.copy(target_)
    eps_0 = np.copy(eligible_)
    tau = np.copy(tau_0)
    eps = np.copy(eps_0)
    tau_cur = np.copy(tau)
    eps_cur = np.copy(eps)
    while(np.sum(tau) > 0):
        val_extpl_temp = np.copy(val_extpl)
        for i in range(len(val_)):
            for j in range(len(val_[i])):
                if(tau[i,j] == 1):
                    triplet_count = 0
                    triplet_sum = 0
                    # 2.9 is used to check if every element in the length-3 array is 1
                    if(np.sum(eps[i+1:i+4,j]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i+1,j] - 3*val_extpl[i+2,j] + val_extpl[i+3,j]
                    if(np.sum(eps[i-3:i,j]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i-1,j] - 3*val_extpl[i-2,j] + val_extpl[i-3,j]
                    if(np.sum(eps[i,j+1:j+4]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i,j+1] - 3*val_extpl[i,j+2] + val_extpl[i,j+3]
                    if(np.sum(eps[i,j-3:j]) > 2.9):
                        triplet_count += 1
                        triplet_sum += 3*val_extpl[i,j-1] - 3*val_extpl[i,j-2] + val_extpl[i,j-3]
                        
                    if(triplet_count > 0):
                        val_extpl_temp[i,j] = triplet_sum / triplet_count
                        tau_cur[i,j] = 0
                        eps_cur[i,j] = 1
                        
        tau = np.copy(tau_cur)
        eps = np.copy(eps_cur)
        val_extpl = np.copy(val_extpl_temp)
        
    return val_extpl

# Interpolation method for 2d grid using interpolate from scipy
# mesh (xmesh, ymesh) must be equal-distanced mesh grid
from scipy.interpolate import RegularGridInterpolator
def interpolation(mesh, mesh_p, fmesh):
    xmesh, ymesh = mesh
    xmesh_p, ymesh_p = mesh_p
    x = xmesh[0, :]
    y = ymesh[:, 0]
    f = RegularGridInterpolator((x, y), fmesh)
    fmesh_p = np.zeros_like(fmesh)
    rmesh = np.moveaxis(np.array([xmesh_p,ymesh_p]), 0, -1)
    
    fmesh_p = f(rmesh)
    fmesh_p = np.moveaxis(fmesh_p,0,1)
    
    return fmesh_p

## poisson solver function
## the result solution is subtracted by their average at every iteration
# u_init_          : (N*N np array) initial data
# maxIterNum_      : (scalar)       maximum iteration for Jacobi method
# mesh_            : (duple)        (xmesh, ymesh)
# phi_             : (N*N np array) level set
# source_          : (N*N np array) right hand side 
# print_option     : (bool)         switch to print the iteration progress
def poisson_jacobi_solver_zero(u_init_, maxIterNum_, source_, phi_,h_,print_option = True):
    u_prev = np.copy(u_init_)
    u      = np.copy(u_init_)
    isIn   = mhf.get_frame_n(phi_)
    numIn  = np.sum(isIn)
    for i in range(maxIterNum_):
        # enforce boundary condition
        u[ 0, :] = np.zeros_like(u[ 0, :])
        u[-1, :] = np.zeros_like(u[-1, :])
        u[ :, 0] = np.zeros_like(u[ :, 0])
        u[ :,-1] = np.zeros_like(u[ :,-1])
    
        u_new = np.copy(u)
    
        # update u according to Jacobi method formula
        # https://en.wikipedia.org/wiki/Jacobi_method
        
        del_u = u[1:-1,2:] + u[1:-1,0:-2] + u[2:,1:-1] + u[0:-2,1:-1]
        u_new[1:-1,1:-1] = -h_**2/4 * (source_[1:-1,1:-1] - del_u/h_**2)
        u = u_new
        
        # for Neumann condition: normalize the inside to mean = 0
        u -= (np.sum(u*isIn) / numIn)*isIn
        
        # check convergence and print process
        check_convergence_rate = 10**-5
        
        if(i % int(maxIterNum_*0.1) < 0.1):
            u_cur = np.copy(u)
            L2Dif = mhf.L_n_norm(np.abs(u_cur - u_prev)) / mhf.L_n_norm(u_cur)
            
            if(L2Dif < check_convergence_rate):
                break;
            else:
                u_prev = np.copy(u_cur)
            if(print_option):
                sys.stdout.write("\rJacobi Solver Progress: %4d iterations (max %4d)" % (i,maxIterNum_))
                sys.stdout.flush()
    if(print_option):
        print("")
    
    return u

## main coefficient poisson solver function
def stm_coef_Neumann(inputconfig):
    
    # making copies of the variables
    phi           = np.copy(inputconfig.phi_)
    rho           = np.copy(inputconfig.rho_)
    S_zeta        = np.copy(inputconfig.S_zeta_)
    theory        = np.copy(inputconfig.theory_)
    zeta          = np.copy(inputconfig.zeta_)
    boundary      = np.copy(inputconfig.boundary_)
    u_cur_result  = np.copy(inputconfig.u_init_)
    
    # iteration numbers
    maxIt         = inputconfig.maxIt_
    it_multiple   = inputconfig.it_multiple_
    
    # convergence parameters
    rlx_          = inputconfig.rlx_
    eta_          = inputconfig.eta_
    
    #mesh variables
    xmesh, ymesh  = inputconfig.mesh_
    h             = inputconfig.h_
    N             = inputconfig.N_grid
    
    # Level variables
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    Omega_m = mhf.D(-phi)
    Omega_p = mhf.D(phi)
    isIn = mhf.get_frame_n(phi)
    
    #1. Extend g(x,y) off of Gamma, define b throughout N2
    xmesh_p, ymesh_p = projection((xmesh,ymesh), phi)
    g_ext = interpolation((xmesh, ymesh), (xmesh_p, ymesh_p), boundary)
    
    a_mesh = g_ext * N2
    x = xmesh[0, :]
    y = ymesh[:, 0]
    
    #2. extrapolate f throughout N1 U Omega^+
    f_org = np.copy(S_zeta)
    eligible_0 = Omega_p * (1-N1)
    target_0 = N1 * (1 - eligible_0)
    f_extpl = extrapolation(f_org, target_0, eligible_0)
    
    #3. initialize a based on initial u throughout N2
    u_extpl = extrapolation(u_cur_result, target_0, eligible_0)  
    b_mesh = np.copy(u_extpl)
    
    #4. Find the source term for coefficient
    ux, uy = mhf.grad(u_cur_result, h, h)
    ux_extpl = extrapolation(ux, target_0, eligible_0)
    uy_extpl = extrapolation(uy, target_0, eligible_0)
    zetax, zetay = mhf.grad(zeta,h,h)
    extra = zetax * ux_extpl + zetay * uy_extpl
    f_use = (f_extpl - extra) / (zeta - singular_null)
    
    # termination array
    Q_array = np.zeros(maxIt)
    
    
    print_it = True
    for it in range(maxIt):
        # print iteration process
        if(print_it):
            print("Source term method grid size %d iteration %d :" % (N, it + 1))
        
        #A1-1 compute the source term
        source = get_source(a_mesh, b_mesh, phi, f_use, h)
        
        #A1-2 compute the source term with the addition of convergence term
        q = -0.75 * min(1, it*0.1)
        source += (q / h * u_cur_result) * (1-Omega_p) * N2
        
        #A2 call a Poisson solver resulting in u throughout Omega
        maxIterNum = it_multiple * N**2
        u_result = poisson_jacobi_solver_zero(u_cur_result, maxIterNum, source, phi, h,print_it)
        maxDif,L2Dif = mhf.get_error_N(u_result, theory, isIn)
        
        ## return nan if error goes too larges
        if(maxDif > 100):
            return np.nan * np.ones_like(u_result), it
        
        change = np.abs(u_result - u_cur_result)
        maxChange = np.max(change * isIn)
        
        # Adding relaxation
        q = rlx_
        u_cur_result = q * u_result + (1 - q) * u_cur_result
        
        #A3-1 Extrapolate u throughout N2
        eligible_0 = Omega_p * (1-N1)
        target_0 = N2 * (1-eligible_0)
        u_extpl = extrapolation(u_result, target_0, eligible_0)
        
        #A3-2 compute the new a throughout N2
        b_mesh = np.copy(u_extpl)
        
        #A3-3 compute the new source term f_use
        ux, uy = mhf.grad(u_cur_result, h, h)
        ux_extpl = extrapolation(ux, target_0, eligible_0)
        uy_extpl = extrapolation(uy, target_0, eligible_0)
        extra = zetax * ux_extpl + zetay * uy_extpl
        f_use = (f_extpl - extra) / (zeta - singular_null)
        
        #A4 check for termination
        Q_array[it] = np.max(u_result * (1-isIn) * N2)
        
        if(it > 5):
            hard_conergence_rate = eta_
            hard_convergence = maxChange / (np.max(np.abs(u_extpl)) + mhf.singular_null) < hard_conergence_rate
            if(hard_convergence):
                break
    u_result_org = np.copy(u_result)
    
    # Quadruple lagrange extrapolation to the full grid
    isIn_full = mhf.get_frame_n(rho)
    eligible_0 = Omega_p * (1-N1)
    target_0 = isIn_full * (1-eligible_0)  
    u_extpl_lagrange = extrapolation(u_result_org, target_0, eligible_0)
    
    
    inputconfig.result = np.copy(u_extpl_lagrange)
    inputconfig.endIt = it
    
    return u_extpl_lagrange, it


if(__name__ == "__main__"):
    plt.close("all")
    N_grid = 60
    u_init, (xmesh,ymesh), h = mhf.setup_grid(N_grid)
    r0 = 0.8
    r = mhf.XYtoR(xmesh,ymesh)
    rho = -r**2 + r0**2
    coef = np.copy(rho)
    phi = -r**2 + r0**2 - 0.0522
    param = 0.5
    theory = ymesh + param * np.sin(xmesh)
    boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
    rhs = -(2*(ymesh + param * xmesh* np.cos(xmesh)) - param*(r**2 - r0**2)*np.sin(xmesh))
    frame = mhf.get_frame_n(phi)
    frame_2 = mhf.get_frame_n(rho)
    mhf3d.plot2d_compare(frame,frame_2,frame_2)
    ux, uy = mhf.grad(theory,h,h)
    rhs_theory = mhf.div(rho*ux,rho*uy,h,h)
    
    #test
#    mhf3d.plot2d_compare(rhs,rhs_theory,frame)
    # maximum iteration number for the source term method
    maxIter = 100
    # the total iteration number N for Jacobi solver = it_multi * N_grid**2
    it_multi = 10
    u_extpl_result = coef_poisson_jacobi_source_term_Neumann_relativistic(u_init, it_multi, (xmesh,ymesh),phi,rho,rhs,coef,\
                                       theory*frame, boundary, maxIter)
#    mhf.plot2d_compare(u_extpl_result,theory,frame)
    