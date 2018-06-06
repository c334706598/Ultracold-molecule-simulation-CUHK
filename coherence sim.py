# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:49:41 2018

@author: XIN
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time

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
            self.position_xs = self.lattice_const * np.random.choice(int((self.N/self.lattice_filling)**(1/3))+1, self.N)
            self.position_ys = self.lattice_const * np.random.choice(int((self.N/self.lattice_filling)**(1/3))+1, self.N)
            self.position_zs = self.lattice_const * np.random.choice(int((self.N/self.lattice_filling)**(1/3))+1, self.N)
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
        
    def update(self):
        self.distances = self.get_dist()
        self.interaction = self.get_interaction()
               
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
        
    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.position_xs, self.position_ys, self.position_zs)
        plt.show()
        
    def get_interaction(self):
        ### interaction in phase accumulation per ms
        axial = np.where(self.distances == 0, 0, 0.158 * self.d**2 / (self.distances**3 + 2**-100))
        # axial part
        # 0.158 coms from Debye^2/(um)^3/(4pi epsilon0 * hb)/3/2
        
#        zi = np.reshape(np.repeat(self.position_zs, self.N), 
#                                (self.N, self.N))
#        zj = np.transpose(np.reshape(np.repeat(self.position_zs, self.N), 
#                                (self.N, self.N)))
#        angular = np.where(self.distances == 0, 0, 1-3*((zi-zj)/(self.distances + 2**-100))**2) # the quantization axis align to z
        
        
        xi = np.reshape(np.repeat(self.position_xs, self.N), 
                                (self.N, self.N))
        xj = np.transpose(np.reshape(np.repeat(self.position_xs, self.N), 
                                (self.N, self.N)))
        yi = np.reshape(np.repeat(self.position_ys, self.N), 
                                (self.N, self.N))
        yj = np.transpose(np.reshape(np.repeat(self.position_ys, self.N), 
                                (self.N, self.N)))
        angular = np.where(self.distances == 0, 0, 1-3*((xi-xj+yi-yj)/np.sqrt(2)/(self.distances + 2**-100))**2) # the quantization axis align to x+y
        
#        print(axial)
#        print(angular)
        # angular part
        return axial * angular
    
### Ignore this part    
#    def get_phase_rate(self):
#        return np.sum(self.interaction, axis=0)
#        
#    def evolve(self):
#        phase_rate = self.get_phase_rate()
#        self.phase += phase_rate * self.dt
#        
#    def run(self, duration=1):
#        t = 0
#        while t <= duration - self.dt:
#            self.evolve()
#            t += self.dt
#    
#    def get_coherence(self):
#        coherence_x = np.mean(np.cos(self.phase))
#        coherence_y = np.mean(np.sin(self.phase))
#        return np.sqrt(coherence_x**2 + coherence_y**2)
### Ignore this part    
        
    def find_neighbors(self, index=0, nn=6):
        ### find the indices of n neighbor that has the largest interaction
        return np.abs(self.interaction[index]).argsort()[-nn:][::-1]  
    
    def _Hamiltonian(self, spin_indices):
        l = len(spin_indices)
        h = np.zeros((2**l, 2**l), dtype=np.float32)
        for row in range(2**l):
            for column in range(2**l):
                if self.flip_condition(row, column, l):
                    # find the spin-exchange interaction
                    spin1_index, spin2_index = self.find_flip_pair(row, column)
                    spin1_index = spin_indices[spin1_index]
                    spin2_index = spin_indices[spin2_index]
                    h[row, column] = self.interaction[spin1_index, spin2_index]
                else:
                    pass
        return h
    
    def flip_condition(self, index1, index2, total_spin):
        ### spin is encoded in the binary form of indices, with 1 being spin up and 0 being spin down
        if index1 * index2 == 0 or index1 >= 2**total_spin - 1 or index2 >= 2**total_spin - 1:
        # if all spin up or all spin down, can not flip, or invalid index
            return False
        else:
            temp = index1^index2
            # get the different spins
            count = 0
            while temp != 0:
                count += temp % 2
                temp //= 2
            if count == 2:
                # if there are two spin states are different, can flip from one to another
                return True
            else:
                return False
    
    def find_flip_pair(self, index1, index2):
        temp = index1^index2
        res = []
        count = 0
        while temp != 0:
            if temp % 2 == 1:
                res.append(count)
            temp //= 2
            count += 1
        return res
                
    def _spin_evol(self, initial, hamiltonian, tmax):
        ## d psi = h * psi * (-i) * dt
        psi = initial
        t = 0
        while t <= tmax - self.dt:
            psi = psi / np.linalg.norm(psi)
            psi += np.matmul(hamiltonian, psi)*self.dt*(-1j)
            t += self.dt
        return psi
    
    def spin_evol(self, initial, hamiltonian, tmax, t_interval):
        
        psi_res = []
        psi = initial
        T = 0
        while T <= tmax - t_interval:
            psi = self._spin_evol(psi, hamiltonian, t_interval)
            T += t_interval
            psi_res.append(psi)
        return psi_res

    def ensemble_evol(self, n_of_neigh, tmax, t_interval):
        sx_res = []
        sy_res = []
        sz_res = []
        for i in range(self.N):
            neigh = dd.find_neighbors(i,n_of_neigh)
            neigh = np.insert(neigh,0,i)
            hamiltonian = dd._Hamiltonian(neigh)
            initial = np.ones(2**(n_of_neigh+1), dtype=np.complex128)
            temp = dd.spin_evol(initial, hamiltonian, tmax, t_interval)
            
            sx = []
            sy = []
            sz = []
            for j in range(len(temp)):
                psi_f = temp[j]
                sx_f = np.matmul(np.conjugate(psi_f),np.matmul(dd._Sx(n_of_neigh+1), psi_f))
                sy_f = np.matmul(np.conjugate(psi_f),np.matmul(dd._Sy(n_of_neigh+1), psi_f))
                sz_f = np.matmul(np.conjugate(psi_f),np.matmul(dd._Sz(n_of_neigh+1), psi_f))
                sx.append(sx_f)
                sy.append(sy_f)
                sz.append(sz_f)
                
            sx_res.append(sx)
            sy_res.append(sy)
            sz_res.append(sz)
            
        return np.array(sx_res), np.array(sy_res), np.array(sz_res)
        
    def _Sx(self, total_spin):
        ### get the sx operator for the single spin
        sx = np.zeros((2**total_spin, 2**total_spin), dtype=np.complex128)
        for row in range(2**total_spin):
            column = (row % 2)*(-2) + row + 1
            sx[row, column] = 1/2
        return sx

        
    def _Sy(self, total_spin):
        ### get the sy operator for the single spin
        sy = np.zeros((2**total_spin, 2**total_spin), dtype=np.complex128)
        for row in range(2**total_spin):
            column = (row % 2)*(-2) + row + 1
            if row % 2 == 0:
                sy[row, column] = -1j/2
            else:
                sy[row, column] = 1j/2
        return sy
                
    
    def _Sz(self, total_spin):
        ### get the sz operator for the single spin
        sz = np.zeros((2**total_spin, 2**total_spin), dtype=np.complex128)
        for row in range(2**total_spin):
            column = row
            if row % 2 == 0:
                sz[row, column] = 1/2
            else:
                sz[row, column] = -1/2
        return sz
        
        
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
                      lattice=True, lattice_const=0.532, lattice_filling=0.2,
                      dt=0.001)
#    dd.N = 3
#    dd.position_xs = np.array([0, 0, 0])
#    dd.position_ys = np.array([0, 0, 0])
#    dd.position_zs = np.array([0, 0.532, 1.064])
#    
#    dd.update()
    
    n_of_neigh = 4
    tmax = 40
    t_interval = 0.5
#    print(dd.distances)
    t1 = time.time()
    sx_evol, sy_evol, sz_evol = dd.ensemble_evol(n_of_neigh, tmax, t_interval)
    t2 = time.time()
#    print(sx_evol)
    y = np.mean(sx_evol, axis=0)
    y = np.insert(y,0,0.5)
#    print(y)
    x = np.linspace(0, tmax, (tmax-t_interval)//t_interval + 2)
#    print(x)
    plt.plot(x, y)
#    print(dd.position_xs)
#    print(dd.position_ys)
#    print(dd.position_zs)
#    print(dd.interaction)
#    print(dd._Hamiltonian([0,1,2]))
    print(t2-t1)
#    dd.plot_3d()
#    t = []
#    coh = []
#    for i in range(50):
#        dd.run(duration=1)
#        t.append(i)
#        coh.append(dd.get_coherence()) 
#    plt.plot(t,coh)
#        
#        
#    n_of_neigh = 1
#    neigh = dd.find_neighbors(0,n_of_neigh)
#    neigh = np.insert(neigh,0,0)
#    print(dd.find_neighbors(0,1))
#    print(neigh)
#    hamiltonian = dd._Hamiltonian(neigh)
#    print(hamiltonian)
#    initial = np.ones(2**(n_of_neigh+1), dtype=np.complex128)
#    print(initial)
#    sx_i = np.matmul(np.conjugate(initial),np.matmul(dd._Sx(n_of_neigh+1), initial))
#    sy_i = np.matmul(np.conjugate(initial),np.matmul(dd._Sy(n_of_neigh+1), initial))
#    sz_i = np.matmul(np.conjugate(initial),np.matmul(dd._Sz(n_of_neigh+1), initial))
#    print(sx_i)
#    print(sy_i)
#    print(sz_i)
#    final = dd._spin_evol(initial, hamiltonian, 200)
#    print(final)
#    sx_f = np.matmul(np.conjugate(final),np.matmul(dd._Sx(n_of_neigh+1), final))
#    sy_f = np.matmul(np.conjugate(final),np.matmul(dd._Sy(n_of_neigh+1), final))
#    sz_f = np.matmul(np.conjugate(final),np.matmul(dd._Sz(n_of_neigh+1), final))
#    print(sx_f)
#    print(sy_f)
#    print(sz_f)
    
    
#    x = []
#    y = []
#    for i in range(200):
#        final = dd._spin_evol(initial, hamiltonian, 1*i)
#        final = final / np.linalg.norm(final)
#        sx_f = np.matmul(np.conjugate(final),np.matmul(dd._Sx(n_of_neigh+1), final))
#        sy_f = np.matmul(np.conjugate(final),np.matmul(dd._Sy(n_of_neigh+1), final))
#        sz_f = np.matmul(np.conjugate(final),np.matmul(dd._Sz(n_of_neigh+1), final))
#        x.append(i)
#        y.append(sx_f)
#    plt.plot(x,y)
    
#    x = []
#    y = []
#    res = dd.spin_evol(initial, hamiltonian, 500, 0.1)
#    for i in range(len(res)):
#        psi_f = res[i]
#        sx_f = np.matmul(np.conjugate(psi_f),np.matmul(dd._Sx(n_of_neigh+1), psi_f))
#        x.append(i)
#        y.append(sx_f)
#    plt.plot(x,y)
    


    
#    print(dd.interaction)
#    print(dd.flip_condition(5,9,7))
    
#    print(dd._Sx(2))
#    print(dd._Sy(2))
#    print(dd._Sz(2))