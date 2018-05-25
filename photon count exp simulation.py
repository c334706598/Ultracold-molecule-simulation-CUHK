# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:37:35 2018

@author: XIN
"""
import numpy as np
import matplotlib.pyplot as plt

class Experiment(object):
    def __init__(self, Ni, beta, reps, dt, tmin, tmax, switch_mode, switch_period, detector_dict):
        self.N = Ni
        self.beta = beta
        self.emit_ph = 0
        self.reps = reps
        self.dt = dt
        self.tmin = tmin
        self.tmax = tmax
        self.S_mode = switch_mode
        self.S_period = switch_period
        self.d_tuple = detector_dict

    def run(self):
        detector = Detector(collection_angle = detector_dict["collection_angle"] ,
                            quantum_efficiency = detector_dict["quantum_efficiency"] ,
                            other_reduction = detector_dict["other_reduction"] ,
                            dark_count = detector_dict["dark_count"] ,
                            reps = self.reps)
        
        N_data = []
        emit_ph_data = []
        
        t = tmin
        while t <= tmax:
            if self.S_mode and (t//self.S_period)%2 == 1: 
                N, emit_ph, reps = self.N, 0, self.reps
            else:
                N, emit_ph, reps = self.evol()
                
            detector.collect(emit_ph, self.dt)    
            N_data.append(N)
            emit_ph_data.append(emit_ph)
            t += dt
        
        return N_data, emit_ph_data, detector.data
            
    def evol(self):
        temp = int(self.beta * (self.N)**2 * self.dt)
#        print("N=",self.N)
        self.N -= temp
        self.emit_ph = temp // 2
        return self.N, self.emit_ph, self.reps
        
        
class Detector(object):
    def __init__(self, collection_angle, quantum_efficiency, other_reduction, dark_count, reps):
        self.col_angle = collection_angle
        self.q_effi = quantum_efficiency
        self.other_red = other_reduction
        self.dark_cnt = dark_count
        self.data = np.array([])
        self.reps = reps
        
    def collect(self, emit_ph, dt):
        dark_count = self.dark_cnt * dt
        bright_count = emit_ph * self.col_angle / (4*np.pi) * self.q_effi * self.other_red
        total_mean_count = dark_count + bright_count
        real_count = np.random.poisson(total_mean_count, (1, self.reps))
#        print(real_count)
        if not self.data.any():
            self.data = np.array(real_count)
#            print(type(self.data))
#            print(np.shape(self.data))
        else:
            self.data = np.concatenate((self.data, real_count))
#            print(np.shape(self.data))
#            print(self.data)

            
class Analyser(object):
    def __init__(self, data):
        self.data = data
        
    def TimeCorrelation(self, interval_min, interval_max, interval_step):
        size = np.shape(self.data)
        g2 = np.array([])
        for interval in range(interval_min, interval_max+1, interval_step):
            mean = np.mean(self.data, axis=0)
            mean = np.transpose(np.reshape(np.repeat(mean, size[0], axis=0),(-1,size[0])))
            padding = np.zeros((interval, size[1]), dtype=np.int32)
            array_1 = np.concatenate((padding, self.data-mean), axis=0)
            array_2 = np.concatenate((self.data-mean, padding), axis=0)
            cross_product = np.multiply(array_1, array_2)
            corr = np.mean(cross_product, axis=0) * size[0] / (size[0]-interval)
            if not g2.any():
                g2 = corr[np.newaxis,:]
#                print(np.shape(g2))
            else:
                
                g2 = np.concatenate((g2, corr[np.newaxis,:]))
#                print(np.shape(g2))
        return g2
        
Ni = 10000
beta = 0.01
reps = 1000
dt = 0.001
tmin = 0
tmax = 0.1
switch_mode = False
switch_period = 0.001
detector_dict = {"collection_angle":0.05,
                 "quantum_efficiency":0.65,
                 "other_reduction":1,
                 "dark_count":500}        
experiment_1 = Experiment(Ni, beta, reps, dt, tmin, tmax, 
                          switch_mode, switch_period, detector_dict)

NN_data, PH_data, DETECT_data = experiment_1.run()
#print(NN_data)
#x = np.linspace(0, 0.1, len(NN_data))
#plt.scatter(x, NN_data)
#plt.figure()
#plt.scatter(x, PH_data)
#plt.show()
#print(PH_data)
#print(DETECT_data)
a = Analyser(DETECT_data)
g = np.transpose(a.TimeCorrelation(1,30,1))
x = np.linspace(dt, dt*30, 30)
mean_g = np.mean(g,axis=0)
print(np.shape(mean_g))
#print(np.shape(g))
#print(np.shape(np.transpose(g)))
#print(np.shape(x))
plt.figure()
#for y in g:
#    plt.plot(x, y, '-')
plt.plot(x, mean_g, '-')
plt.show()

        