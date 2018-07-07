# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 09:27:34 2018

@author: XIN
"""
import numpy as np
import matplotlib.pyplot as plt

class internal_structure(object):
    def __init__(self):
        self.species_A = None
        self.species_B = None
        self.gI_A = None
        self.gI_B = None
        self.g_r = None
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.eqQ_A = None
        self.eqQ_B = None
        self.alpha_par = None
        self.alpha_per = None
        self.alpha_0 = None
        self.J = None
        self.I_A = None
        self.I_B = None
        self.Hamiltonian = None
        self.eigvals = None
        self.eigvecs = None
        self.mJ_list = None
        self.mI_A_list = None
        self.mI_B_list = None
        self.B_field = None
        self.beam_theta = None
        self.beam_intensity = None
        
    def print_out_attributes(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        
    def set_species(self, A='Na', B='Rb'):
        self.species_A = A
        self.species_B = B
    
    def set_rotation_quantum_number(self, J=1):
        self.J = J
        
    def set_hyperfine_quantum_number(self, I_A=1.5, I_B=1.5):
        self.I_A = I_A
        self.I_B = I_B
        
    def set_Lande_g(self, gI_A=1.484, gI_B=1.832, g_r=0.001):
        self.gI_A = gI_A
        self.gI_B = gI_B
        self.g_r = g_r
        
    def set_couplings(self, c1=60.7, c2=983.8, c3=259.3, c4=6560):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        
    def set_qradrupole(self, eqQ_A=-0.132*10**6, eqQ_B=-3.048*10**6):
        self.eqQ_A = eqQ_A
        self.eqQ_B = eqQ_B
        
    def set_polarizability(self, alpha_par=5.89553*10**-3, alpha_per=1.90617*10**-3):
        self.alpha_par = alpha_par
        self.alpha_per = alpha_per
        self.alpha_0 = (self.alpha_par + 2*self.alpha_per)/3
        
    def set_B_field(self, B=0.0336): 
        self.B_field = B

    def set_ODT_beam(self, intensity=13*10**7, theta=0):
        self.beam_intensity = intensity
        self.beam_theta = theta
                
    def ladder_p(self, j, m):
        if -j <= m <= j+1:
            return np.sqrt((j-m)*(j+m+1))
        else:
            return 0
    
    def ladder_m(self, j, m):
        if -j+1 <= m <= j:
            return np.sqrt((j+m)*(j-m+1))
        else:
            return 0
        
    def Initialize_Hamiltonian(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        self.Hamiltonian = np.zeros((size, size))
        self.mJ_list = [x-self.J for x in range(N_J)]
        self.mI_A_list = [x-self.I_A for x in range(N_I_A)]
        self.mI_B_list = [x-self.I_B for x in range(N_I_B)]
        
    def get_IA_J_interaction(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        H_IA_J = np.zeros((size, size))        
        for mJ in self.mJ_list:
            for mI_A in self.mI_A_list:
                for mI_B in self.mI_B_list:
                    index = self.qnum_to_index((mJ, mI_A, mI_B))                    
                    H_IA_J[index, index] = mJ * mI_A
                    
                    index2 = self.qnum_to_index((mJ+1, mI_A-1, mI_B))
                    index3 = self.qnum_to_index((mJ-1, mI_A+1, mI_B))
                    if index2:
                        H_IA_J[index2, index] = self.ladder_p(self.J, mJ)*self.ladder_m(self.I_A, mI_A)/2
                    if index3:
                        H_IA_J[index3, index] = self.ladder_m(self.J, mJ)*self.ladder_p(self.I_A, mI_A)/2
        return self.c1 * H_IA_J

    def get_IB_J_interaction(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        H_IB_J = np.zeros((size, size))        
        for mJ in self.mJ_list:
            for mI_A in self.mI_A_list:
                for mI_B in self.mI_B_list:
                    index = self.qnum_to_index((mJ, mI_A, mI_B))                    
                    H_IB_J[index, index] = mJ * mI_B
                    
                    index2 = self.qnum_to_index((mJ+1, mI_A, mI_B-1))
                    index3 = self.qnum_to_index((mJ-1, mI_A, mI_B+1))
                    if index2:
                        H_IB_J[index2, index] = self.ladder_p(self.J, mJ)*self.ladder_m(self.I_B, mI_B)/2
                    if index3:
                        H_IB_J[index3, index] = self.ladder_m(self.J, mJ)*self.ladder_p(self.I_B, mI_B)/2
        return self.c2 * H_IB_J                        
       
    def get_IA_IB_scalar_interaction(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        H_IA_IB = np.zeros((size, size))        
        for mJ in self.mJ_list:
            for mI_A in self.mI_A_list:
                for mI_B in self.mI_B_list:
                    index = self.qnum_to_index((mJ, mI_A, mI_B))                    
                    H_IA_IB[index, index] = mI_A * mI_B
                    
                    index2 = self.qnum_to_index((mJ, mI_A+1, mI_B-1))
                    index3 = self.qnum_to_index((mJ, mI_A-1, mI_B+1))
                    if index2:
                        H_IA_IB[index2, index] = self.ladder_p(self.I_A, mI_A)*self.ladder_m(self.I_B, mI_B)/2
                    if index3:
                        H_IA_IB[index3, index] = self.ladder_m(self.I_A, mI_A)*self.ladder_p(self.I_B, mI_B)/2
        return self.c4 * H_IA_IB
    
    def get_IA_IB_tensor_interaction(self):
        H_IA_J = self.get_IA_J_interaction() / self.c1
        H_IB_J = self.get_IB_J_interaction() / self.c2
        H_IA_IB = self.get_IA_IB_scalar_interaction() / self.c4        
        res = self.c3 * (3*np.matmul(H_IA_J, H_IB_J) + 3*np.matmul(H_IB_J, H_IA_J) - 2*self.J*(self.J+1)*H_IA_IB)
        res /= (2*self.J-1) * (2*self.J+3)
        return res

    def get_electric_quadrupole_A_interaction(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        H_IA_J = self.get_IA_J_interaction() / self.c1
        Identity = np.eye(size)
        factor = -self.eqQ_A / (2*self.I_A*(2*self.I_A-1)*(2*self.J-1)*(2*self.J+3))
        H_eqQ_A = 3*np.matmul(H_IA_J, H_IA_J) + 1.5*H_IA_J - self.I_A*(self.I_A+1)*self.J*(self.J+1)*Identity
        H_eqQ_A *= factor
        return H_eqQ_A
    
    def get_electric_quadrupole_B_interaction(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        H_IB_J = self.get_IB_J_interaction() / self.c2
        Identity = np.eye(size)
        factor = -self.eqQ_B / (2*self.I_B*(2*self.I_B-1)*(2*self.J-1)*(2*self.J+3))
        H_eqQ_B = 3*np.matmul(H_IB_J, H_IB_J) + 1.5*H_IB_J - self.I_B*(self.I_B+1)*self.J*(self.J+1)*Identity
        H_eqQ_B *= factor
        return H_eqQ_B
    
    def get_Zeeman_shift(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        H_Zee = np.zeros((size, size))
        for mJ in self.mJ_list:
            for mI_A in self.mI_A_list:
                for mI_B in self.mI_B_list:
                    index = self.qnum_to_index((mJ, mI_A, mI_B))                    
                    H_Zee[index, index] = self.gI_A * mI_A + self.gI_B * mI_B + self.g_r * mJ
                    
        factor = -7622590 * self.B_field
        H_Zee *= factor            
        return H_Zee
        
    def get_ac_Stark_shift(self):
        N_J, N_I_A, N_I_B, size = self.get_number_of_states()
        temp = N_I_A * N_I_B
        H_ac_Stark = np.zeros((size, size))
        factor_neg1_0 = -np.sqrt(2)/5 * self.beam_intensity * (self.alpha_par-self.alpha_per) * np.sin(self.beam_theta) * np.cos(self.beam_theta)
        factor_neg1_1 = -1/5 * self.beam_intensity * (-self.alpha_par + self.alpha_per) * np.sin(self.beam_theta)**2
        factor_0_1 = np.sqrt(2)/5 * self.beam_intensity * (self.alpha_par-self.alpha_per) * np.sin(self.beam_theta) * np.cos(self.beam_theta)
        factor_neg1_neg1 = factor_1_1 = -1/5 *self.beam_intensity * ((2*self.alpha_par + 3*self.alpha_per)*np.sin(self.beam_theta)**2\
                                                                     + (self.alpha_par + 4*self.alpha_per)*np.cos(self.beam_theta)**2)
        factor_0_0 = -1/5 * self.beam_intensity * ((self.alpha_par + 4*self.alpha_per)*np.sin(self.beam_theta)**2\
                                                   + (3*self.alpha_par + 2*self.alpha_per)*np.cos(self.beam_theta)**2)        
                                                             
        for i in range(temp):
            H_ac_Stark[i, i+temp] = H_ac_Stark[i+temp, i] = factor_neg1_0
            H_ac_Stark[i, i+2*temp] = H_ac_Stark[i+2*temp, i] = factor_neg1_1
            H_ac_Stark[i+temp, i+2*temp] = H_ac_Stark[i+2*temp, i+temp] = factor_0_1
            H_ac_Stark[i, i] = factor_neg1_neg1
            H_ac_Stark[i+2*temp, i+2*temp] = factor_1_1
            H_ac_Stark[i+temp, i+temp] = factor_0_0
        
        return H_ac_Stark
                
    def put_IA_J_interaction(self):
        self.Hamiltonian += self.get_IA_J_interaction()
        
    def put_IB_J_interaction(self):
        self.Hamiltonian += self.get_IB_J_interaction()
        
    def put_IA_IB_interaction(self):
        self.Hamiltonian += self.get_IA_IB_scalar_interaction() + self.get_IA_IB_tensor_interaction()
        
    def put_electric_quadrupole_interaction(self):
        self.Hamiltonian += self.get_electric_quadrupole_A_interaction() + self.get_electric_quadrupole_B_interaction()
    
    def put_Zeeman_shift(self):
        self.Hamiltonian += self.get_Zeeman_shift()
        
    def put_ac_Stark_shift(self):
        self.Hamiltonian += self.get_ac_Stark_shift()
    
    def index_to_qnum(self, index):
        N_J = 2*self.J+1
        N_I_A = 2*self.I_A+1
        N_I_B = 2*self.I_B+1
        size = N_J * N_I_A * N_I_B
        if index >= size or index < 0:
            return None
        else:
            mJ = index // (N_I_A * N_I_B) - self.J
            mI_A = (index % (N_I_A * N_I_B)) // N_I_B - self.I_A
            mI_B = (index % (N_I_A * N_I_B)) % N_I_B - self.I_B
            return (mJ, mI_A, mI_B)
        
    def qnum_to_index(self, qnum):
        N_I_A = 2*self.I_A+1
        N_I_B = 2*self.I_B+1
        mJ, mI_A, mI_B = qnum
        if -self.J <= mJ <= self.J and -self.I_A <= mI_A <= self.I_A and -self.I_B <= mI_B <= self.I_B:
            return int((mJ+self.J)*(N_I_A*N_I_B) + (mI_A+self.I_A)*(N_I_B) +(mI_B+self.I_B))
        else:
            return None
        
    def get_number_of_states(self):
        N_J = 2*self.J+1
        N_I_A = 2*self.I_A+1
        N_I_B = 2*self.I_B+1
        size = N_J * N_I_A * N_I_B
        return int(N_J), int(N_I_A), int(N_I_B), int(size)
    
    def solve_Hamiltonian(self):
        self.eigvals, self.eigvecs = np.linalg.eig(self.Hamiltonian)
        



if __name__ == '__main__':
    
    exp = internal_structure()
    exp.set_species()
    exp.set_B_field()
    exp.set_couplings()
    exp.set_rotation_quantum_number()
    exp.set_hyperfine_quantum_number()
    exp.set_Lande_g()
    exp.set_ODT_beam()
    exp.set_polarizability()
    exp.set_qradrupole()
    
    exp.Initialize_Hamiltonian()
    exp.put_IA_J_interaction()
    exp.put_IB_J_interaction()
    exp.put_IA_IB_interaction()
    exp.put_electric_quadrupole_interaction()
    exp.put_ac_Stark_shift()
    exp.put_Zeeman_shift()
    
    exp.solve_Hamiltonian()
    exp.print_out_attributes()
    plt.matshow(np.abs(exp.eigvecs))
    



