# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:03:56 2022

@author: hofer
"""

import numpy as np
import pandas as pd

class GaussianParameters():
    
    def __init__(self):
        pass
        
    def lab_widths(self, wx, wy, theta):
        wxx = ((wx * np.cos(theta))** 2 + 
                            (wy * np.sin(theta))**2)**.5
        wyy = ((wx * np.sin(theta))** 2 + 
                            (wy * np.cos(theta))**2)**.5
        return wxx, wyy
    
    
    def edge_mask(self, shape, wx, wy, theta, scale=2):
        height, width = shape
        wxx, wyy = self.lab_widths(wx, wy, theta)
        wxx_s, wyy_s = scale * wxx, scale * wyy
        box_coords = [wxx_s, wyy_s, width - wxx_s, height - wyy_s]
        box_coords = [int(round(val)) for val in box_coords]
        xmin, ymin, xmax, ymax = box_coords
        emask = np.zeros(shape, dtype = bool)
        emask[ymin:ymax, xmin:xmax] = True
        return emask
    
    
    def get_valid_random_coords(self, X, Y, wx, wy, theta, scale=2):
        img_dimensions = X.shape
        emask = self.edge_mask(img_dimensions, wx, wy, theta, scale)
        valid_xcoords = X[emask]
        valid_ycoords = Y[emask]   
        random_index = np.random.randint(len(valid_xcoords))
        x0 = valid_xcoords[random_index]
        y0 = valid_ycoords[random_index]
        return x0, y0
        
    
    def get_single_gaussian_data(self, X, Y, a_vals, w_vals, theta_vals, 
                             offset_vals, noise_vals, img_dims):
        
        w_vals = np.array(w_vals) * np.amin(img_dims)
        wx = w_vals[0] + w_vals[2] * np.random.random()
        wy = w_vals[0] + w_vals[2] * np.random.random()
        amplitude = a_vals[0] + a_vals[2] * np.random.random()
        theta = theta_vals[0] + theta_vals[2] * np.random.random()

        offset = offset_vals[0] + offset_vals[2] * np.random.random()
        x0, y0 = self.get_valid_random_coords(X, Y, wx, wy, theta)
        fits = [amplitude, x0, y0, wx, wy, theta, offset]
        noise_std = noise_vals[0] + noise_vals[2] * np.random.random()
        return fits, noise_std


class SimulationParameters():
    
    def __init__(self, sampling_params=None):
        self.define_simulation_parameters()
        if sampling_params is None:
            self.define_sampling_parameters()
        else:
            self.num_samples, self.dmin, self.dmax, self.dstep = sampling_params
    
    
    def define_simulation_parameters(self):
        self.noise_std_min = 0.09999
        self.noise_std_max = .1

        self.w_min = 1 / 10
        self.w_max = 1 / 5
        self.theta_min = -(np.pi / 2)
        self.theta_max = (np.pi / 2)
        self.a_min = .5
        self.a_max = 1
        self.offset_min = 0
        self.offset_max = .1
        self.seed_min = .7
        self.seed_max = 1.3
        
        
    def define_sampling_parameters(self):
        self.num_samples = 11
        self.dmin = 10 ** 3
        self.dmax = 1 * 10**5
        self.dstep = 10
        return self.num_samples, self.dmin, self.dmax, self.dstep
    
    
    def get_coordinates(self, width, height):
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        X, Y = np.meshgrid(x, y)
        return X, Y


    def get_random_float(self, low, high):
        delta = high - low
        return low + delta * np.random.random()


    def get_param_range(self, minv, maxv):
        span = maxv - minv
        return minv, maxv, span
        
    
    def get_simulation_parameters(self):
        w_vals = self.get_param_range(self.w_min, self.w_max)
        theta_vals = self.get_param_range(self.theta_min, self.theta_max)
        a_vals = self.get_param_range(self.a_min, self.a_max)
        offset_vals = self.get_param_range(self.offset_min, self.offset_max)
        noise_vals =  self.get_param_range(self.noise_std_min, self.noise_std_max)
        seed_vals =  [self.seed_min, self.seed_max]
        # self.get_param_range(self.seed_min, self.seed_max)
        
        return noise_vals, w_vals, theta_vals, a_vals, offset_vals, seed_vals


    def save_simulation_parameters(self, text_path):
        data_names = ['noise_std_min', 'noise_std_max', 'w_min', 'w_max', 
                      'theta_min', 'theta_max', 'a_min', 'a_max', 'offset_min', 
                      'offset_max', "seed_min", "seed_max", 'dmin', 'dmax', 'dstep']


        sim_data = [self.noise_std_min, self.noise_std_max, self.w_min, 
                    self.w_max, self.theta_min, self.theta_max, self.a_min, 
                    self.a_max, self.offset_min, self.offset_max, self.seed_min, 
                    self.seed_max, self.dmin, self.dmax, self.dstep]
        
        simulation_params = pd.DataFrame([sim_data], columns=data_names)
        simulation_params.to_json(text_path / 'simulation_paramters.json')
        
        
    def create_simulation_params(self):

        gbp = GaussianParameters()
        all_params = self.get_simulation_parameters()
        noise_vals, w_vals, theta_vals, a_vals, offset_vals, seed_vals = all_params
        num_samples, dmin, dmax, dstep = self.define_sampling_parameters()
        # sp.save_simulation_parameters(text_path)

        dlengths = np.logspace(np.log10(dmin), np.log10(dmax), dstep)
        lengths = np.ndarray.astype(dlengths**.5, dtype=int)
        row_datas = []
        counter = 0
        for length in lengths:
            img_dims = (length, length)
            XY_tuple = self.get_coordinates(*img_dims)
            X, Y = XY_tuple
            for i in range(num_samples):
                gparams, noise_std = gbp.get_single_gaussian_data(X, Y, a_vals, 
                                                                  w_vals, theta_vals, 
                                                                  offset_vals, 
                                                                  noise_vals, 
                                                                  img_dims)
                
                seed = [param * self.get_random_float(*seed_vals) for param in gparams]
                row_data = [img_dims[0], img_dims[1], gparams, seed, noise_std, 
                            str(counter) + '.npy']
                row_datas.append(row_data)
                counter += 1

        df_labels = ['height', 'width', 'parameters', 'seed', 'noise_std', 'file_name']
        gaussian_data = pd.DataFrame(row_datas, columns=df_labels)
        return gaussian_data