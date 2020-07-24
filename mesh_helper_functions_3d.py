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
print("mhf3d called")

#adding to denominator to avoid 0/0 error
singular_null = 1.0e-30

###############################################


#### spherical coordinates transformation ####
# x = r sin(Theta) cos(Phi)
# y = r sin(Theta) sin(Phi)
# z = r cos(Theta)

# return r(x,y)
def XYZtoR(x,y,z):
    return np.sqrt(x**2+y**2+z**2)

# return theta(x,y)
def XYZtoTheta(x,y,z):
    return np.arctan2(np.sqrt(x**2 + y**2),z)

# return (x,y)
def XYZtoPhi(x,y,z):
    return np.arctan2(y,x)

# return normalized mesh
def norm_mesh(x,y,z):
    ret_l = np.sqrt(x**2 + y**2 + z**2)
    ret_x = x / (ret_l + singular_null)
    ret_y = y / (ret_l + singular_null)
    ret_z = z / (ret_l + singular_null)
    return ret_x, ret_y, ret_z

#### Level set functions ####
# Delta function (phi > 0)
def D(phi_):
    return np.heaviside(phi_,1)

# Delta function (phi > 0)
def not_D(phi_):
    return np.heaviside(-phi_,1)

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
def grad(f,dx,dy,dz):
    ret_y, ret_x, ret_z = np.gradient(f,dx,dy,dz)
    return ret_x, ret_y, ret_z

# normalized gradient function
def grad_norm(f,dx,dy,dz):
    ret_y, ret_x, ret_z = np.gradient(f,dx,dy,dz)
    ret_x_n, ret_y_n, ret_z_n = norm_mesh(ret_x, ret_y, ret_z)
    return ret_x_n, ret_y_n, ret_z_n
    

# absolute gradient
def abs_grad(f,dx,dy,dz):
    grad_y, grad_x, grad_z = np.gradient(f,dx,dy,dz)
    return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

# Laplacian
def laplace(f,dx,dy,dz):
    ret = np.zeros_like(f)
    ret[1:-1,1:-1,1:-1] = (f[1:-1,2:  ,1:-1] + f[1:-1,0:-2,1:-1] +\
                           f[2:  ,1:-1,1:-1] + f[0:-2,1:-1,1:-1] +\
                           f[1:-1,1:-1,2:  ] + f[1:-1,1:-1,0:-2] - 6*f[1:-1,1:-1,1:-1])/dx**2
    return ret

# Normalized normal gradient
def grad_n(u_, phi_, dx_, dy_, dz_):
    phi_x, phi_y, phi_z = grad_norm(phi_,dx_,dy_,dz_)
    u_nx, u_ny, u_nz = grad(u_, dx_, dy_,dz_)
    u_n = u_nx * phi_x + u_ny * phi_y + u_nz * phi_z
    return u_n

# Normalized normal gradient
def grad_n_n(u_, phi_, dx_, dy_, dz_):
    phi_x, phi_y, phi_z = grad_norm(phi_,dx_,dy_,dz_)
    u_nx, u_ny, u_nz = grad(u_, dx_, dy_,dz_)
    u_n = -(u_nx * phi_x + u_ny * phi_y + u_nz * phi_z)
    return u_n

# grad(a) dot grad(b)
def grad_dot_grad(a_mat, b_mat, dx,dy,dz):
    ax,ay,az = grad(a_mat,dx,dy,dz)
    bx,by,bz = grad(b_mat,dx,dy,dz)
    return ax*bx + ay*by + az*bz

def grad_dot(scalar, vec, dx,dy,dz):
    ax, ay, az = grad(scalar,dx,dy,dz)
    return vec[0] * ax + vec[1] * ay + vec[2] * az

# divergence(fx,fy,fz)
def div(fx,fy,fz,dx,dy,dz):
    ret_x = np.zeros_like(fx)
    ret_y = np.zeros_like(fy)
    ret_z = np.zeros_like(fz)
#    ret_x[1:-1, :  , :  ] = (fx[:-2,:  ,:  ] - fx[2:, :, :])/(2*dx)
#    ret_y[ :  ,1:-1, :  ] = (fy[:  ,:-2,:  ] - fy[: ,2:, :])/(2*dy)
#    ret_z[ :  , :  ,1:-1] = (fz[:  ,:  ,:-2] - fz[: , :,2:])/(2*dz)
#    ret_x[0 , :, :] = (0 - fx[ 1, :, :])/dx
#    ret_x[-1, :, :] = (fx[-1, :, :] - 0)/dx
#    ret_y[: , 0, :] = (0 - fy[ :, 1, :])/dy
#    ret_y[: ,-1, :] = (fy[ :,-1, :] - 0)/dy
#    ret_z[: , :, 0] = (0 - fz[ :, :, 1])/dz
#    ret_z[: , :,-1] = (fz[ :, :,-1] - 0)/dz
    ret_y[1:-1, :  , :  ] = (fy[2:,:  ,:  ] - fy[:-2, :, :])/(2*dy)
    ret_x[ :  ,1:-1, :  ] = (fx[:  ,2:,:  ] - fx[: ,:-2, :])/(2*dx)
    ret_z[ :  , :  ,1:-1] = (fz[:  ,:  ,2:] - fz[: , :,:-2])/(2*dz)
    ret_y[0 , :, :] = (fy[ 1, :, :] - 0)/dy
    ret_y[-1, :, :] = (0 - fy[-1, :, :])/dy
    ret_x[: , 0, :] = (fx[ :, 1, :] - 0)/dx
    ret_x[: ,-1, :] = (0 - fx[ :,-1, :])/dx
    ret_z[: , :, 0] = (fz[ :, :, 1] - 0)/dz
    ret_z[: , :,-1] = (0 - fz[ :, :,-1])/dz
    
    
    return ret_x + ret_y + ret_z



# gradient in the frame, boundary with ghost points
def grad_frame(u_, mesh_, phi_):
    xmesh, ymesh, zmesh = mesh_
    isOut   = np.array(D(phi_)             , dtype = int)
    isOutx1 = np.array(D(phi_[1 :,  :,  :]), dtype = int)
    isOutx2 = np.array(D(phi_[:-1,  :,  :]), dtype = int)
    isOuty1 = np.array(D(phi_[  :, 1:,  :]), dtype = int)
    isOuty2 = np.array(D(phi_[  :,:-1,  :]), dtype = int)
    isOutz1 = np.array(D(phi_[  :,  :, 1:]), dtype = int)
    isOutz2 = np.array(D(phi_[  :,  :,:-1]), dtype = int)
    
    # phi_x, phi_y
    dx = xmesh[0,1,0]-xmesh[0,0,0]
    dy = ymesh[1,0,0]-ymesh[0,0,0]
    dz = zmesh[0,0,1]-zmesh[0,0,0]
    phi_x, phi_y, phi_z = grad_norm(phi_,dx,dy,dz)
    
#    step is 1 if k+1 is out and k is in
#    step is -1 if k is out and k+1 is in
#    step is 0 if both out or in
    xstep = isOutx1 - isOutx2
    ystep = isOuty1 - isOuty2
    zstep = isOutz1 - isOutz2
    xstep_p = np.array(D( xstep),dtype = int)
    xstep_m = np.array(D(-xstep),dtype = int)
    ystep_p = np.array(D( ystep),dtype = int)
    ystep_m = np.array(D(-ystep),dtype = int)
    zstep_p = np.array(D( zstep),dtype = int)
    zstep_m = np.array(D(-zstep),dtype = int)
    
    # ghost points for the boundary
    u_ghost_x = np.copy(u_) * (1-isOut)
    u_ghost_y = np.copy(u_) * (1-isOut)
    u_ghost_z = np.copy(u_) * (1-isOut)
    
    u_ghost_x[:-2,:,:] += -u_[ 2:,:,:]*xstep_m[:-1,:,:] + 2*u_[ 1:-1,:,:]*xstep_m[:-1,:,:]
    u_ghost_x[ 2:,:,:] += -u_[:-2,:,:]*xstep_p[ 1:,:,:] + 2*u_[ 1:-1,:,:]*xstep_p[ 1:,:,:]
    u_ghost_y[:,:-2,:] += -u_[:, 2:,:]*ystep_m[:,:-1,:] + 2*u_[:, 1:-1,:]*ystep_m[:,:-1,:]
    u_ghost_y[:, 2:,:] += -u_[:,:-2,:]*ystep_p[:, 1:,:] + 2*u_[:, 1:-1,:]*ystep_p[:, 1:,:]
    u_ghost_z[:,:,:-2] += -u_[:,:, 2:]*zstep_m[:,:,:-1] + 2*u_[:,:, 1:-1]*zstep_m[:,:,:-1]
    u_ghost_z[:,:, 2:] += -u_[:,:,:-2]*zstep_p[:,:, 1:] + 2*u_[:,:, 1:-1]*zstep_p[:,:, 1:]
    
    u_nx,temp,temp = grad(u_ghost_y,dx,dy,dz)
    temp,u_ny,temp = grad(u_ghost_x,dx,dy,dz)
    temp,temp,u_nz = grad(u_ghost_z,dx,dy,dz)
    
    u_n = u_nx * phi_x + u_ny * phi_y + u_nz * phi_z
    
    return (u_nx* (1-isOut),u_ny* (1-isOut),u_nz* (1-isOut)) 

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

#
def map_01_11(arr):
    return np.array(2*(arr - 0.5),dtype = int)

def map_11_01(arr):
    return arr/2.0 + 0.5

# find absolute error, return maximum error, L2 error 
def get_error(u_result_, frame_, sol_, print_option = True):
    dif = np.abs(u_result_ - sol_)
    L2Dif = L_n_norm_frame(dif,frame_,2)
    maxDif = np.max(dif*frame_)
    if(print_option):
        print("Max error : ", maxDif)
        print("L^2 error : ", L2Dif)
        print("")
    return maxDif, L2Dif

# find absolute relative error, return maximum relative error, L2 relative error 
def get_error_N(u_result_, u_theory_, frame_, option = (True,True)):
#    plt.matshow(frame_[:,:,1])
    w1,w2,w3 = u_result_.shape
    u_result_0 = u_result_ - u_result_[int(w1/2),int(w2/2)]
    u_theory_0 = u_theory_ - u_theory_[int(w1/2),int(w2/2)]
    dif = np.abs(u_result_0 - u_theory_0)
    L2Dif = L_n_norm_frame(dif,frame_,2)
    maxDif = np.max(dif*frame_)
    if(option[0]):
        print("Max error : ", maxDif)
        print("L^2 error : ", L2Dif)
        print("")
#    if(option[1]):
#        plt.matshow(dif*frame_)
    return maxDif, L2Dif

# show the position of the maximum absolute in the grid
def show_max(u_):
    ret = np.zeros_like(u_)
    maxNum = np.max(np.abs(u_))
    ret = np.heaviside(np.abs(u_) - maxNum,1)
    plt.matshow(ret)
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


def setup_grid_3d(N_grid_val = 100):
    # grid dimension
    grid_min = -1.
    grid_max = 1.
    
    global N_grid
    N_grid = N_grid_val
    # grid spacing
    global h
    h = (grid_max - grid_min) / (N_grid) 
    
    # define arrays to hold the x and y coordinates
    xyz = np.linspace(grid_min,grid_max,N_grid + 1)
    global x, y, z
    x,y,z = xyz,xyz,xyz
    global xmesh, ymesh, zmesh
    xmesh, ymesh, zmesh = np.meshgrid(xyz,xyz,xyz)
    
    # solution
    global u_init
    u_init = np.zeros_like(xmesh)
    return u_init, (xmesh,ymesh), h

def log_frame(mat_, frame_):
    return np.log(mat_* frame_ + 1.0*(1-frame_))

def plot2d(mat_, title_="", *arg):
    frame = np.ones_like(mat_)
    if(arg):
        frame = arg[0]
    plt.matshow(mat_*frame)
    plt.colorbar()
    plt.title(title_)

    
def plot2d_compare(mat1, mat2, *arg):
    
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    frame = np.ones_like(mat1)
    this_cmap = cm.coolwarm
    if(arg):
        frame = arg[0]
        
    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    
    
    pt1 = ax[0,0].matshow(mat1*frame,cmap = this_cmap)
    ax[0,0].set_title("result")
    fig.colorbar(pt1,ax = ax[0,0], format=formatter)
    
    pt2 = ax[0,1].matshow(mat2*frame,cmap = this_cmap)
    ax[0,1].set_title("theory")
    fig.colorbar(pt2,ax = ax[0,1], format=formatter)
    
    pt3 = ax[1,0].matshow(np.abs(mat1 - mat2)*frame,cmap=this_cmap)
    ax[1,0].set_title("absolute error")
    fig.colorbar(pt3,ax = ax[1,0], format=formatter)
    
    frame_small = get_frame_n(np.abs(frame*mat2) - 1e-15*np.max(np.abs(frame*mat2)))
    pt4 = ax[1,1].matshow(np.abs(mat1 - mat2)/ (np.abs(mat2) + singular_null) * frame_small,cmap=this_cmap)
    ax[1,1].set_title("relative error")
    fig.colorbar(pt4,ax = ax[1,1], format=formatter)
    
    if(len(arg) > 1):
        fig.suptitle(arg[1])
    
    for i in range(2):
        for j in range(2):
            ax[i,j].xaxis.set_ticks_position('bottom')
            ax[i,j].yaxis.set_ticks_position('left')
            if(len(arg) > 2):
                ax[i,j].set_xlabel(arg[2][0])
                ax[i,j].set_ylabel(arg[2][1])
    return fig, ax
                
def plot2d_compare_zero(mat1, mat2, *arg):
    
    frame = np.ones_like(mat1)
    if(arg):
        frame = arg[0]
        
    mat1_zero = mat1 - frame * np.sum(mat1*frame) / np.sum(frame)
    mat2_zero = mat2 - frame * np.sum(mat2*frame) / np.sum(frame)
    ret = plot2d_compare(mat1_zero, mat2_zero, frame)
    return ret
    
    
if(__name__ == "__main__"):
    plt.close("all")
    print("Poisson solver mesh helper function 3d file")
#    grid_size = 100
#    setup_grid_3d(grid_size)
#    x0,y0,z0 = 0.5,0.2,0.3
#    f = (xmesh-x0)**2 + (ymesh-y0)**2 + (zmesh - z0)**2
#    gx, gy, gz = grad(f,h,h,h)
#    z_slice = int(grid_size / 2)
#    plt.matshow(gx[:,:,z_slice])
#    plt.colorbar()
#    plt.matshow(gy[:,:,z_slice])
#    plt.colorbar()
#    plt.matshow(gz[:,:,z_slice])
#    plt.colorbar()
#    plt.matshow(xmesh[:,:,z_slice])
#    plt.colorbar()
#    print(xmesh[0,:,0])
#    print(ymesh[:,0,0])
#    print(zmesh[0,0,:])
    
