# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:49:41 2018

@author: XIN
"""

import numpy as np
import matplotlib.pyplot as plt

class dd_coherence(object):
    def __init__(self, N=5000, T=0.7, 
                 fx=200, fy=180, fz=30, 
                 mass=110, dipole=3.3, 
                 lattice=False, lattice_const=0.5, lattice_filling = 0.1,
                 dt = 0.01,):
        ### N - molecule number, T - sample temperature in uK, fx,fy,fz - trap frequency in Hz
        ### mass - molecular mass in AMU, dipole - molecular permanent dipole in Debye
        ### If lattice == True, trap = (fx,fy,fz), else molecule in lattice arrangement
        ### lattice_const - lattice constant in um
        ### dt - discrete evolution interval
        
        self.N = N
        self.T = T
        self.trap_fx = fx
        self.trap_fy = fy
        self.trap_fz = fz
        self.mass = mass
        self.d = dipole
        self.lattice = lattice
        self.lattice_const = lattice_const
        self.lattice_filling = lattice_filling
        self.dt = dt 
        
        if self.lattice == True:
            self.position_xs = self.lattice_const * np.random.choice(int((self.N/self.lattice_filling)**(1/3)), self.N)
            self.position_ys = self.lattice_const * np.random.choice(int((self.N/self.lattice_filling)**(1/3)), self.N)
            self.position_zs = self.lattice_const * np.random.choice(int((self.N/self.lattice_filling)**(1/3)), self.N)
        else:
            ### sigma = sqrt(kB*T/m)/(2pi*f), all length are in um unit
            self.position_xs = np.random.normal(
                loc=0.0,
                scale=14451.5*np.sqrt(self.T/self.mass)/self.trap_fx,
                size=self.N
                )
            self.position_ys = np.random.normal(
                loc=0.0,
                scale=14451.5*np.sqrt(self.T/self.mass)/self.trap_fy,
                size=self.N
                )        
            self.position_zs = np.random.normal(
                loc=0.0,
                scale=14451.5*np.sqrt(self.T/self.mass)/self.trap_fz,
                size=self.N
                )
            
        self.distances = self.get_dist()
        self.interaction = self.get_interaction()
        self.phase = np.zeros(self.N)
        
    def get_dist(self):
        
        xi_squared = np.reshape(np.repeat(self.position_xs ** 2, self.N), 
                                (self.N, self.N))
        xj_squared = np.transpose(np.reshape(np.repeat(self.position_xs ** 2, self.N), 
                                (self.N, self.N)))
        x_cross = np.outer(self.position_xs, self.position_xs)
        
        yi_squared = np.reshape(np.repeat(self.position_ys ** 2, self.N), 
                                (self.N, self.N))
        yj_squared = np.transpose(np.reshape(np.repeat(self.position_ys ** 2, self.N), 
                                (self.N, self.N)))
        y_cross = np.outer(self.position_ys, self.position_ys)

        zi_squared = np.reshape(np.repeat(self.position_zs ** 2, self.N), 
                                (self.N, self.N))
        zj_squared = np.transpose(np.reshape(np.repeat(self.position_zs ** 2, self.N), 
                                (self.N, self.N)))
        z_cross = np.outer(self.position_zs, self.position_zs)        
        
        return np.sqrt(xi_squared + xj_squared - 2*x_cross + yi_squared + yj_squared - 2*y_cross \
               + zi_squared + zj_squared - 2*z_cross)
        
    def get_interaction(self):
        ### interaction in phase accumulation per ms
        axial = np.where(self.distances == 0, 0, 0.316 * self.d**2 / (self.distances**3 + 2**-100))
        ### 0.316 coms from Debye^2/(um)^3/(4pi epsilon0 * hb)/3
        zi = np.reshape(np.repeat(self.position_zs, self.N), 
                                (self.N, self.N))
        zj = np.transpose(np.reshape(np.repeat(self.position_zs, self.N), 
                                (self.N, self.N)))
        angular = np.where(self.distances == 0, 0, 1-3*((zi-zj)/(self.distances + 2**-100))**2)
        return axial * angular
    
    def get_phase_rate(self):
        return np.sum(self.interaction, axis=0)
        
    def evolve(self):
        phase_rate = self.get_phase_rate()
        self.phase += phase_rate * self.dt
        
    def run(self, duration=1):
        t = 0
        while t <= duration - self.dt:
            self.evolve()
            t += self.dt
    
    def get_coherence(self):
        coherence_x = np.mean(np.cos(self.phase))
        coherence_y = np.mean(np.sin(self.phase))
        return np.sqrt(coherence_x**2 + coherence_y**2)
        
        
if __name__ == '__main__':
#    dd = dd_coherence(N=5000, T=0.7, 
#                      fx=200, fy=190, fz=30, 
#                      mass=110, dipole=3.3, 
#                      lattice=False, lattice_const=0.5, lattice_filling = 0.07,
#                      dt=0.01)
#    print(dd.position_xs)
#    print(dd.position_ys)
#    plt.scatter(dd.position_xs, dd.position_ys)
#    print(dd.position_zs)
#    print(dd.get_dist())
#    print(dd.phase)
#    print(dd.interaction)
#    print(dd.get_phase_rate())
#    print(dd.get_coherence())
    dd = dd_coherence(N=5000, T=0.7, 
                      fx=200, fy=190, fz=30, 
                      mass=127, dipole=0.57, 
                      lattice=True, lattice_const=1, lattice_filling = 0.1,
                      dt=0.1)
    t = []
    coh = []
    for i in range(50):
        dd.run(duration=1)
        t.append(i)
        coh.append(dd.get_coherence()) 
    plt.plot(t,coh)
#        
        
        
        