# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:55:40 2020

@author: Johnny Tsao
The same helper function file as poisson_helper_functions.py
This uses function with mesh type.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#adding to denominator to avoid 0/0 error
singular_null = 1.0e-30

###############################################

def setup_grid(N_grid_val = 100):
    # grid dimension
    grid_min = -1.
    grid_max = 1.
    
    N_grid = N_grid_val
    # grid spacing
    h = (grid_max - grid_min) / (N_grid) 
    
    # define arrays to hold the x and y coordinates
    xy = np.linspace(grid_min,grid_max,N_grid + 1)
    xmesh, ymesh = np.meshgrid(xy,xy)
    
    # solution
    u_init = np.zeros_like(xmesh)
    return u_init, (xmesh,ymesh), h


#### polar coordinates transformation ####

# return r(x,y)
def XYtoR(x,y):
    return np.sqrt(x**2+y**2)

# return theta(x,y)
def XYtoTheta(x,y):
    return np.arctan2(y,x)

# return normalized mesh
def norm_mesh(x,y):
    ret_l = np.sqrt(x**2 + y**2)
    ret_x = x / (ret_l + singular_null)
    ret_y = y / (ret_l + singular_null)
    return ret_x, ret_y

#### Level set functions ####
# Delta function (phi > 0)
def D(phi_):
    return np.heaviside(phi_,1)

# return the interior of the level set (phi < 0)
def get_frame(phi_):
    isOut = D(phi_)
    return 1 - isOut

# return the interior of the level set (phi < 0)
def get_frame_n(phi_):
    isOut = D(-phi_)
    return 1 - isOut

#### vector calculus helper functions ####
    
# gradient function
def grad(f,dx,dy):
    ret_y, ret_x = np.gradient(f,dx,dy)
    return ret_x, ret_y

# normalized gradient function
def grad_norm(f,dx,dy):
    ret_y, ret_x = np.gradient(f,dx,dy)
    ret_x_n, ret_y_n = norm_mesh(ret_x, ret_y)
    return ret_x_n, ret_y_n
    

# absolute gradient
def abs_grad(f,dx,dy):
    grad_y, grad_x = np.gradient(f,dx,dy)
    return np.sqrt(grad_x**2 + grad_y**2)

# Laplacian
def laplace(f,dx,dy):
    ret = np.zeros_like(f)
    ret[1:-1,1:-1] = (f[1:-1,2:] + f[1:-1,0:-2] + f[2:,1:-1] + f[0:-2,1:-1] - 4*f[1:-1,1:-1])/dx**2
    return ret

# Normalized normal gradient
def grad_n(u_, phi_, dx_, dy_):
    phi_x, phi_y = grad_norm(phi_,dx_,dy_)
    u_nx, u_ny = grad(u_, dx_, dy_)
    u_n = u_nx * phi_x + u_ny * phi_y
    return u_n

# Normalized normal gradient
def grad_n_n(u_, phi_, dx_, dy_):
    phi_x, phi_y = grad_norm(phi_,dx_,dy_)
    u_nx, u_ny = grad(u_, dx_, dy_)
    u_n = -(u_nx * phi_x + u_ny * phi_y)
    return u_n

# grad(a) dot grad(b)
def grad_dot_grad(a_mat, b_mat, h):
    ax,ay = grad(a_mat,h,h)
    bx,by = grad(b_mat,h,h)
    return ax*bx + ay*by

# divergence(fx,fy)
def div(fx,fy,dx,dy):
    ret_x = np.zeros_like(fx)
    ret_y = np.zeros_like(fy)
    ret_y[1:-1,:] = -(fy[:-2,:  ] - fy[2:, :])/(2*dy)
    ret_x[:,1:-1] = -(fx[:  ,:-2] - fx[: ,2:])/(2*dx)
    ret_y[0 ,:] = -(0 - fy[ 1,:])/dy
    ret_y[-1,:] = -(fy[-1,:] - 0)/dy
    ret_x[: , 0] = -(0 - fx[ :,1])/dx
    ret_x[: ,-1] = -(fx[:,-1] - 0)/dx
    return ret_x + ret_y



# gradient in the frame, boundary with ghost points
def grad_frame(u_, mesh_, phi_):
    xmesh, ymesh = mesh_
    isOut   = np.array(D(phi_)       ,dtype = int)
    isOutx1 = np.array(D(phi_[1:,:]) ,dtype = int)
    isOutx2 = np.array(D(phi_[:-1,:]),dtype = int)
    isOuty1 = np.array(D(phi_[:,1:]) ,dtype = int)
    isOuty2 = np.array(D(phi_[:,:-1]),dtype = int)
    
    # phi_x, phi_y
    dx = xmesh[0,1]-xmesh[0,0]
    dy = ymesh[1,0]-ymesh[0,0]
    phi_x, phi_y = grad_norm(phi_,dx,dy)
    
#    step is 1 if k+1 is out and k is in
#    step is -1 if k is out and k+1 is in
#    step is 0 if both out or in
    xstep = isOutx1 - isOutx2
    ystep = isOuty1 - isOuty2
    xstep_p = np.array(D( xstep),dtype = int)
    xstep_m = np.array(D(-xstep),dtype = int)
    ystep_p = np.array(D( ystep),dtype = int)
    ystep_m = np.array(D(-ystep),dtype = int)
    
    # ghost points for the boundary
    u_ghost_x = np.copy(u_) * (1-isOut)
    u_ghost_y = np.copy(u_) * (1-isOut)
    
    u_ghost_x[:-2,:] += -u_[ 2:,:]*xstep_m[:-1,:] + 2*u_[ 1:-1,:]*xstep_m[:-1,:]
    u_ghost_x[ 2:,:] += -u_[:-2,:]*xstep_p[ 1:,:] + 2*u_[ 1:-1,:]*xstep_p[ 1:,:]
    u_ghost_y[:,:-2] += -u_[:, 2:]*ystep_m[:,:-1] + 2*u_[:, 1:-1]*ystep_m[:,:-1]
    u_ghost_y[:, 2:] += -u_[:,:-2]*ystep_p[:, 1:] + 2*u_[:, 1:-1]*ystep_p[:, 1:]
    
    u_nx,temp = grad(u_ghost_y,dx,dy)
    temp,u_ny = grad(u_ghost_x,dx,dy)
    u_n = u_nx * phi_x + u_ny * phi_y
    
    return (u_nx, u_ny)

#### Error Analysis ####
    
# return Ln normal
def L_n_norm(error, n=2):
    error_n = np.power(error, n)
    average_n = np.sum(error_n) / len(error_n.flatten())
    average = np.power(average_n, 1./n)
    return average

# return Ln error in the frame 
def L_n_norm_frame(error,frame, n=2):
    num = np.sum(frame)
    error_n = np.power(error, n) * frame
    average_n = np.sum(error_n) / num
    average = np.power(average_n, 1./n)
    return average

# find absolute error, return maximum error, L2 error 
def get_error(u_result_, mesh_, frame_, sol_, print_option = True):
    xmesh,ymesh = mesh_
    dif = np.abs(u_result_ - sol_)
    L2Dif = L_n_norm_frame(dif,frame_,2)
    maxDif = np.max(dif*frame_)
    if(print_option):
        print("Max error : ", maxDif)
        print("L^2 error : ", L2Dif)
        print("")
    return maxDif, L2Dif


# find absolute relative error, return maximum relative error, L2 relative error 
def get_error_N(u_result_, u_theory_, frame_, option = (True,False)):
    w1,w2 = u_result_.shape
    u_result_0 = u_result_ - u_result_[int(w1/2),int(w2/2)]
    u_theory_0 = u_theory_ - u_theory_[int(w1/2),int(w2/2)]
    dif = np.abs(u_result_0 - u_theory_0)
    L2Dif = L_n_norm_frame(dif,frame_,2)
    maxDif = np.max(dif*frame_)
    if(option[0]):
        print("Max error : ", maxDif)
        print("L^2 error : ", L2Dif)
        print("")
    if(option[1]):
        plt.matshow(dif*frame_)
    return maxDif, L2Dif

# show the position of the maximum absolute in the grid
def show_max(u_):
    ret = np.zeros_like(u_)
    maxNum = np.max(np.abs(u_))
    ret = np.heaviside(np.abs(u_) - maxNum,1)
#    plt.matshow(ret)
    return ret
    
# plot the 3d error
def plot3d_all(u_result_, mesh_, sol_, fig_label_, toPlot_ = [True, True, True, True]):
    xmesh, ymesh = mesh_
    
    # 2D color plot of the max difference
    if(toPlot_[0]):
        test_mat = (sol_ - u_result_)/(sol_ + singular_null)
        plt.matshow(test_mat)
        plt.colorbar()
       
    #3D plot of the analytic solution
    if(toPlot_[1]):
        fig_an = plt.figure("poisson analytic solution %d" % fig_label_)
        ax_an = fig_an.gca(projection='3d')
        surf_an = ax_an.plot_surface(xmesh, ymesh, sol_, cmap=cm.coolwarm)
        fig_an.colorbar(surf_an)
        
    #3D plot of the numerical result
    if(toPlot_[2]):
        fig = plt.figure("poisson result %d" % fig_label_)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xmesh, ymesh, u_result_, cmap=cm.coolwarm)
        fig.colorbar(surf)
     
    #3D plot of the error
    if(toPlot_[3]):
        fig_dif = plt.figure("poisson difference %d" % fig_label_)
        ax_dif = fig_dif.gca(projection='3d')
        surf_dif = ax_dif.plot_surface(xmesh, ymesh, u_result_ - sol_, cmap=cm.coolwarm)
        fig_dif.colorbar(surf_dif)
    
    plt.show()

def plot2d(u_, theory_, frame_):
    plt.matshow(u_ * frame_)
    plt.colorbar()
    plt.title("reuslt")
    plt.matshow(theory_ * frame_)
    plt.colorbar()
    plt.title("theory")
    plt.matshow((u_ - theory_) * frame_)
    plt.colorbar()
    plt.title("error")

if(__name__ == "__main__"):
    print("Poisson solver mesh helper function file")
