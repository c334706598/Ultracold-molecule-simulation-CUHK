# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:37:34 2018

@author: XIN
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time

class dd_coherence(object):
    
    def __init__(self):
        ### N - molecule number, T - sample temperature in uK, fx,fy,fz - trap frequency in Hz
        ### mass - molecular mass in AMU, dipole - molecular permanent dipole in Debye
        ### If lattice == True, trap = (fx,fy,fz), else molecule in lattice arrangement
        ### lattice_const - lattice constant in um
        ### dt - discrete evolution interval        
        self.N = None
        self.T = None
        self.trap_fx = None
        self.trap_fy = None
        self.trap_fz = None
        self.mass = None
        self.dipole = None
        self.lattice = None
        self.lattice_const = None
        self.lattice_filling = None
        self.tmax = None
        self.interval = None
        self.dt = None
        self.position_xs = None
        self.position_ys = None
        self.position_zs = None
        self.quantization_axis = None
        self.distances = None
        self.interactions = None
        
    def print_out_properties(self):
        print('N,', self.N)
        print('T,', self.T)
        print('trap_f,', (self.trap_fx, self.trap_fy, self.trap_fz))
        print('mass,', self.mass)
        print('dipole,', self.dipole)
        print('lattice,', self.lattice)
        print('lattice_const,', self.lattice_const)
        print('lattice_filling,', self.lattice_filling)
        print('tmax,', self.tmax)
        print('interval,', self.interval)
        print('dt,', self.dt)
#        print('position_xs,', self.position_xs)
#        print('position_ys,', self.position_ys)
#        print('position_zs,', self.position_zs)
#        print('quantization_axis,', self.quantization_axis)
#        print('distances,', self.distances)
#        print('interactions,', self.interactions)
        
    def set_trap_freq(self, fx, fy, fz):
        self.trap_fx = fx
        self.trap_fy = fy
        self.trap_fz = fz
    
    def set_species(self, mass, dipole):
        self.mass = mass
        self.dipole = dipole
        
    def set_T(self, temperature):
        self.T = temperature
        
    def set_lattice(self, lattice=True, lattice_const=0.5, lattice_filling=0.1):
        self.lattice = lattice
        self.lattice_const = lattice_const
        self.lattice_filling = lattice_filling
        
    def set_quantization_axis(self, qx, qy, qz):
        self.quantization_axis = np.array([qx, qy, qz]) / (qx**2+qy**2+qz**2)**(1/2)
        
    def set_calculation(self, tmax, interval, dt):
        self.tmax = tmax
        self.interval = interval
        self.dt = dt
        
    def put_particle(self, x, y, z):
        if self.N == None:
            self.N = 1
            self.position_xs = np.array([x,])
            self.position_ys = np.array([y,])
            self.position_zs = np.array([z,])            
        else:
            self.N += 1
            self.position_xs = np.append(self.position_xs, x)
            self.position_ys = np.append(self.position_ys, y)
            self.position_zs = np.append(self.position_zs, z)
        
    def generate_N_particles(self, N):
        self.N = N
        
        if self.lattice == True:
            total_size = int((self.N/self.lattice_filling)**(1/3))+1
            coords = np.array(list(range(total_size)))
            self.position_xs = np.repeat(coords, total_size * total_size)
            self.position_ys = np.tile(np.repeat(coords, total_size), total_size)
            self.position_zs = np.tile(coords, total_size * total_size)
            choices = np.random.choice(total_size ** 3, size = self.N, replace=False)
            self.position_xs = self.position_xs[choices] * self.lattice_const
            self.position_ys = self.position_ys[choices] * self.lattice_const
            self.position_zs = self.position_zs[choices] * self.lattice_const
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

    def plot_3d(self):   # Plot spins in space in 3D
        fig = plt.figure(figsize=(20,20)).add
        ax = fig_subplot(111, projection='3d')
        ax.scatter(self.position_xs, self.position_ys, self.position_zs)
        ax.set_aspect('equal')
        plt.show()
        
    def update(self):
        self.get_distances()
        self.get_interactions()
        
    def get_distances(self):
        if self.N == None:
            self.distances = None
            return None
        else:
            xi_squared = np.reshape(np.repeat(self.position_xs ** 2, self.N), (self.N, self.N))
            xj_squared = np.reshape(np.tile(self.position_xs ** 2, self.N), (self.N, self.N))
            x_cross = np.outer(self.position_xs, self.position_xs)
            yi_squared = np.reshape(np.repeat(self.position_ys ** 2, self.N), (self.N, self.N))
            yj_squared = np.reshape(np.tile(self.position_ys ** 2, self.N), (self.N, self.N))
            y_cross = np.outer(self.position_ys, self.position_ys)
            zi_squared = np.reshape(np.repeat(self.position_zs ** 2, self.N), (self.N, self.N))
            zj_squared = np.reshape(np.tile(self.position_zs ** 2, self.N), (self.N, self.N))
            z_cross = np.outer(self.position_zs, self.position_zs)
            res = np.sqrt(xi_squared + xj_squared - 2*x_cross + yi_squared + yj_squared - 2*y_cross + zi_squared + zj_squared - 2*z_cross)
            
        self.distances = res              
        return res

    def get_interactions(self):
        if not np.any(self.quantization_axis):
            self.set_quantization_axis(0,0,1)
        if not np.any(self.distances):
            self.get_distances()
        ### interaction in phase accumulation per ms
        axial = np.where(self.distances == 0, 0, 0.316 * self.dipole**2 / (self.distances**3+2**-100)) # axial part, 0.316 coms from Debye^2/(um)^3/(4pi epsilon0 * hb)/3
        xi = np.reshape(np.repeat(self.position_xs, self.N), (self.N, self.N))
        xj = np.reshape(np.tile(self.position_xs, self.N), (self.N, self.N))
        yi = np.reshape(np.repeat(self.position_ys, self.N), (self.N, self.N))
        yj = np.reshape(np.tile(self.position_ys, self.N), (self.N, self.N))
        zi = np.reshape(np.repeat(self.position_zs, self.N), (self.N, self.N))
        zj = np.reshape(np.tile(self.position_zs, self.N), (self.N, self.N))
        cosine = np.where(self.distances == 0, 0, ((xi-xj)*self.quantization_axis[0] + (yi-yj)*self.quantization_axis[1] + (zi-zj)*self.quantization_axis[2]) / (self.distances+2**-100))        
        angular = 1-3*cosine**2
        res = axial * angular
        self.interactions = res
        return res

    def find_neighbors(self, index=0, nn=3):
        ### find the indices of n neighbor that has the largest interaction
        res = np.abs(self.interactions[index]).argsort()[-nn:][::-1]
        return res
    
    def cluster(self, index=0, nn=3):
        res = self.find_neighbors(index, nn)
        res = np.insert(res, 0, index)
        return res
    
    def flip_condition(self, index1, index2, total_spin):
        ### spin is encoded in the binary form of indices, with 1 being spin up and 0 being spin down
        if index1 * index2 == 0 or index1 >= 2**total_spin - 1 or index2 >= 2**total_spin - 1:
        # if all spin up or all spin down, can not flip, or invalid index
            return False
        else:
            temp = index1^index2
            # get the different spins
            count = 0
            count2 = 0
            while temp != 0:
                count += temp % 2
                if temp % 2 == 1:
                    count2 += index1 % 2
                temp //= 2
                index1 //= 2
            if count == 2 and count2 == 1:
                # Only if there are two spin are different, one spin up and one spin down
                return True
            else:
                return False
    
    def find_flip_pair(self, index1, index2, total_spin):
        if not self.flip_condition(index1, index2, total_spin):
            return None
        else:
            temp = index1^index2
            res = []
            count = 0
            while temp != 0:
                if temp % 2 == 1:
                    res.append(count)
                temp //= 2
                count += 1
            return res
    
    def hamiltonian(self, cluster):
        l = len(cluster)
        h = np.zeros((2**l, 2**l), dtype=np.float32)
        for row in range(2**l):
            for column in range(2**l):
                pair_found = self.find_flip_pair(row, column,l)
                if pair_found:
                    # find the spin-exchange interaction
                    spin1_index, spin2_index = pair_found
                    spin1_index = cluster[spin1_index]
                    spin2_index = cluster[spin2_index]
                    h[row, column] = self.interactions[spin1_index, spin2_index]
                else:
                    pass
        return h
    
    def Sx(self, total_spin):
        ### get the sx operator for the single spin
        sx = np.zeros((2**total_spin, 2**total_spin), dtype=np.complex128)
        for row in range(2**total_spin):
            column = (row % 2)*(-2) + row + 1
            sx[row, column] = 1/2
        return sx
        
    def Sy(self, total_spin):
        ### get the sy operator for the single spin
        sy = np.zeros((2**total_spin, 2**total_spin), dtype=np.complex128)
        for row in range(2**total_spin):
            column = (row % 2)*(-2) + row + 1
            if row % 2 == 0:
                sy[row, column] = -1j/2
            else:
                sy[row, column] = 1j/2
        return sy
                
    def Sz(self, total_spin):
        ### get the sz operator for the single spin
        sz = np.zeros((2**total_spin, 2**total_spin), dtype=np.complex128)
        for row in range(2**total_spin):
            column = row
            if row % 2 == 0:
                sz[row, column] = 1/2
            else:
                sz[row, column] = -1/2
        return sz

    def _spin_evol(self, initial, hamiltonian, tend):
        ## d psi = h * psi * (-i) * dt
        psi = initial
        t = 0
        while t <= tend - self.dt:
            psi = psi/np.linalg.norm(psi)
            psi += np.matmul(hamiltonian, psi)*self.dt*(-1j)
            psi = psi/np.linalg.norm(psi)
            t += self.dt
        return psi

    def spin_evol(self, initial, hamiltonian):        
        psi_res = []
        psi = initial
        t = 0
        while t <= self.tmax - self.interval:
            psi = self._spin_evol(psi, hamiltonian, self.interval)
            t += self.interval
            psi_res.append(psi)
        return psi_res
    
    def ensemble_evol(self, nn=3):
        sx_res = []
        sy_res = []
        sz_res = []
        for i in range(self.N):
            cluster = self.cluster(i, nn)
            hamiltonian = self.hamiltonian(cluster)
            initial = np.ones(2**(nn+1), dtype=np.complex128)
            temp = self.spin_evol(initial, hamiltonian)
            
            sx = []
            sy = []
            sz = []
            
            for j in range(len(temp)):
                psi_f = temp[j]
                sx_f = np.matmul(np.conjugate(psi_f),np.matmul(self.Sx(nn+1), psi_f))
                sy_f = np.matmul(np.conjugate(psi_f),np.matmul(self.Sy(nn+1), psi_f))
                sz_f = np.matmul(np.conjugate(psi_f),np.matmul(self.Sz(nn+1), psi_f))
                sx.append(sx_f)
                sy.append(sy_f)
                sz.append(sz_f)
                
            sx_res.append(sx)
            sy_res.append(sy)
            sz_res.append(sz)            
        return np.array(sx_res), np.array(sy_res), np.array(sz_res)
    
if __name__ == '__main__':
    
########################### Test put_particle() ########################################    
#    exp = dd_coherence()
#    exp.put_particle(1,0,0)
#    exp.put_particle(2,0,0)
#    exp.put_particle(3,0,0)
#    exp.put_particle(4,0,0)
#    exp.put_particle(5,0,0)
#    print(exp.N)
#    exp.plot_3d()
########################################################################################

########################## Test get_distances() ########################################
#    exp = dd_coherence()
#    exp.put_particle(1,0,0)
#    exp.put_particle(2,1,0)
#    exp.put_particle(3,0,1)
#    exp.put_particle(4,2,0)
#    exp.put_particle(5,0,1)
#    print(exp.get_distances())
########################################################################################
    
########################## Test set_quantization_axis() ################################
#    exp = dd_coherence()
#    exp.set_quantization_axis(1,2,3)
#    print(exp.quantization_axis)
########################################################################################

########################## Test get_interactions() #####################################
#    exp = dd_coherence()
#    exp.set_species(127, 0.53)
#    exp.put_particle(0,0,0)
#    exp.put_particle(0,1,0)
#    exp.put_particle(0,0,1)
#    exp.put_particle(1,1,0)
#    exp.put_particle(1,0,1)
#    exp.put_particle(1,-1,0)
#    exp.set_quantization_axis(1,1,0)
#    exp.update()
#    print(exp.interactions*100/(-4.43822))
########################################################################################

########################## Test find_neighbors() and cluster() #########################
#    exp = dd_coherence()
#    exp.set_species(127, 0.53)
#    exp.put_particle(0,0,0)
#    exp.put_particle(0,1,0)
#    exp.put_particle(0,0,1)
#    exp.put_particle(1,1,0)
#    exp.put_particle(1,0,1)
#    exp.put_particle(1,-1,0)
#    exp.set_quantization_axis(1,1,0)
#    exp.update()
#    print(exp.interactions*100/(-4.43822))    
#    print(exp.find_neighbors(4, 2))
#    print(exp.cluster(4, 2))
########################################################################################

########################## Test flip_condition() #######################################
#    exp = dd_coherence()
##    index1 = int('0b0001',2)
##    index2 = int('0b1101',2)
#    cond = exp.flip_condition(index1, index2, 4)
#    cond = np.zeros((16,16))
#    for index1 in range(16):
#        for index2 in range(16):
#            cond[index1, index2] = exp.flip_condition(index1, index2, 4)
#    print(cond)
########################################################################################

########################## Test find_flip_pair() #######################################
#    exp = dd_coherence()
#    index1 = int('0b0010101',2)
#    index2 = int('0b1010001',2)
#    pair = exp.find_flip_pair(index1, index2, 7)
#    print(pair)
########################################################################################

############################# Test hamiltonian() #######################################
#    exp = dd_coherence()
#    exp.set_species(127, 0.53)
#    exp.put_particle(0,0,0)
#    exp.put_particle(0,1,0)
#    exp.put_particle(0,0,1)
#    exp.put_particle(1,1,0)
#    exp.put_particle(1,0,1)
#    exp.put_particle(1,-1,0)
#    exp.set_quantization_axis(1,1,0)
#    exp.update()
#    cluster = exp.cluster(index=0, nn=3)
#    hamiltonian = exp.hamiltonian(cluster)
#    print(exp.interactions)
#    print(cluster)
#    print(hamiltonian)
########################################################################################

############################# Test Sx(), Sy(), Sz() ####################################
#    exp = dd_coherence()
#    Sx = exp.Sx(3)
#    Sy = exp.Sy(3)
#    Sz = exp.Sz(3)
#    print('-------------')
#    print(Sx)
#    print('-------------')
#    print(Sy)
#    print('-------------')
#    print(Sz)
########################################################################################                 

################################# Calculation ##########################################
    exp = dd_coherence()
    exp.set_lattice(False, 0.532, 0.028)
    exp.set_calculation(50,1,0.01)
    exp.set_quantization_axis(0,0,1)
    exp.set_species(110, 3.2)
    exp.set_trap_freq(200,30,190)
    exp.set_T(1)
#    exp.put_particle(0,0,0)
#    exp.put_particle(0,0,0.532)
#    exp.put_particle(2,0,0)
#    exp.put_particle(3,0,0)
#    exp.put_particle(4,0,0)
#    exp.put_particle(5,0,0)
#    exp.put_particle(0,1,0)
#    exp.put_particle(1,1,0)
#    exp.put_particle(2,1,0)
#    exp.put_particle(3,1,0)
#    exp.put_particle(4,1,0)
#    exp.put_particle(5,1,0)
    exp.generate_N_particles(2000)
    exp.update()
    
#    exp.plot_3d()
#    fig = plt.figure(figsize=(20,20))
#    ax = fig.add_subplot(1,1,1)  
#    ax.scatter(exp.position_xs, exp.position_ys)
#    ax.set_aspect('equal')
#    plt.show()
#    exp.print_out_properties()
#    
    for var in range(1,2,1):
        nn = 4
#        exp.set_lattice(True, 0.532, 0.02*3**var)
#        exp.generate_N_particles(1000)
#        exp.update()
#        print("filling", 0.02*3**var)
        t1 = time.time()
        sx_evol, sy_evol, sz_evol = exp.ensemble_evol(nn)
##        print(sz_evol)
##        print(sy_evol)
        t2 = time.time()
        y = np.mean(sx_evol, axis=0)
        y = np.insert(y,0,0.5) * 2
        x = np.linspace(0, exp.tmax, len(y))
        plt.plot(x, y)
        
#        temp = len(x)
#        fft = np.abs(np.fft.fft(y))
#        x = np.linspace(0, exp.tmax, len(fft))
#        plt.plot(x, fft)
#        np.savetxt("fft20180627.csv", fft, delimiter=",")
#        np.savetxt("curve20180627.csv", y, delimiter=",")
        print(t2-t1)

    
        
        