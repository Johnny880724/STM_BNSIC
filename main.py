# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:06:21 2020

@author: Johnny Tsao
"""

import sys
import mesh_helper_functions as mhf
import mesh_helper_functions_3d as mhf3d
import numpy as np
import matplotlib.pyplot as plt
import source_term_method as stm

#adding to denominator to avoid 0/0 error
singular_null = mhf.singular_null

class inputconfig:
    def __init__(self):
        #############################################
        # This section allows customization by users
        #############################################
        
        self.N_grid = 64
        self.u_init_, (xmesh,ymesh), self.h_ = mhf.setup_grid(self.N_grid)
        self.mesh_ = (xmesh, ymesh)
        
        self.maxIt_ = 200
        self.it_multiple_ = 4
        self.eta_ = 1.0e-4
        self.rlx_ = 0.1
    
        r0 = np.sqrt(np.e/4)
        r = mhf.XYtoR(xmesh,ymesh)
        
        ## picking test cases
        power_dict = {"1":2.0, "2":1.0, "3":0.5}
        test_case = "2"
        power = power_dict[test_case]
        
        ## Here a test case with rho ~ r, Phi = y sin(x) is presented
        
        self.rho_ = -r**power + r0**power
        self.zeta_ = np.copy(self.rho_)
        
        #adding separation
        self.num_grid_dr = 2.0
        dr_by_r0 = self.num_grid_dr * self.h_ / r0
        sigma =  1 - (1 - dr_by_r0)**power
        rho_c = r0**power
        self.phi_ = self.rho_ - sigma * rho_c
        
        param = 1.0
        self.theory_ = ymesh + param * np.sin(xmesh)
        
        self.boundary_ = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
        self.S_zeta_ = -power*(ymesh + param * xmesh* np.cos(xmesh))*(r + mhf.singular_null)**(power-2) - param*self.rho_*np.sin(xmesh)
       
        #############################################  
        # end customization
        #############################################
        
        self.frame = mhf.get_frame_n(self.phi_)
        self.frame_full =mhf.get_frame_n(self.rho_)
        
        ## will be updated once the iterations are finished
        self.result = np.zeros_like(self.u_init_)
        self.endIt = 0
        
        
    # this test if the test case satisfy the coefficient Poisson equation
    def test(self, test_rhs, test_boundary):
        ux, uy = mhf.grad(self.theory_,self.h_,self.h_)
        rhs_theory = mhf.div(self.zeta_*ux,self.zeta_*uy,self.h_,self.h_)
        # test rhs
        if(test_rhs):
            mhf3d.plot2d_compare(self.S_zeta_,rhs_theory,self.frame_full,"rhs")
            
        # test boundary
        if(test_boundary):
            mhf3d.plot2d_compare(self.boundary_,mhf.grad_n_n(self.theory_,self.phi_,self.h_,self.h_),stm.get_N1(self.phi_),"boundary")
            
    # this shows the L2 error and max error for the result
    def show_error(self):
        maxDif_extpl, L2Dif_extpl = mhf.get_error_N(self.result, self.theory_, self.frame_full,(False,False))
        maxDif, L2Dif             = mhf.get_error_N(self.result, self.theory_, self.frame,(False,False))
        L2_rel_dif_extpl = L2Dif_extpl / np.sqrt(np.mean((self.theory_)**2))
        print("num grid " + str(self.num_grid_dr) + ": L2 extpl error " ,L2_rel_dif_extpl)

    def plot2d_error(self):
        mhf3d.plot2d_compare_zero(self.result, self.theory_, self.frame_full)
        
    def plot1d_error(self, plot):
        fig, ax = plot
        N_half = int(0.5*self.N_grid)
        rel_err = np.abs(((self.result - self.theory_)*self.frame_full)) / (np.abs(self.theory_) + mhf.singular_null)
        xmesh = self.mesh_[0]
        x_axis = np.delete(xmesh[N_half,:],[N_half,N_half+1])
        y_axis = np.delete(rel_err[N_half,:],[N_half,N_half+1])
        ax.plot(x_axis,y_axis)
        ax.set_title("relative error along the x axis")
        


if(__name__ == "__main__"):
    test_inputconfig = inputconfig()
    test_inputconfig.test(True,True)
    Phi_result, it = stm.stm_coef_Neumann(test_inputconfig)
    
    fig = plt.figure()
    ax = fig.gca()
    test_inputconfig.plot1d_error((fig,ax))
    test_inputconfig.plot2d_error()
    


