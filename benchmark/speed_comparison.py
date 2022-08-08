# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 14:45:53 2022

@author: hofer
"""
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

sys.path.append(r'../')
from jaxfit import CurveFit
    
sys.path.append(r'./timed_scipy_curve_fit')
from scipy_minpack import curve_fit

sys.path.append(r'./classes')
from gauss2d_jax import gaussian2d as gaussian2d_jax
from gauss2d_numpy import gaussian2d as gaussian2d_np
from random_gaussian_params import SimulationParameters
from comparison_utils import mod_container, extract_params, get_gpu_cpu_conversion


class PlotResults():
    
    def plot_overall_speed(self, df_dict, dlengths, savefig=True):
        fig, ax1 = plt.subplots(figsize=(5, 4))
        
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        markers = ['o', 's', '^', 'd', '+']
        lstyles = ['-', '--', ':']
        # dkeys = ['JAX_ut', 'JAX']
        # dkeys = ['JAX_ut', 'SciPy']
        dkeys = ['JAX', 'SciPy']
        
        label_dict = {'SciPy': 'SciPy', 'Gpufit':'Gpufit', 'JAX':'JAXFit', 'JAX_ut':'JAXFit'}
        
        for key, ls, color, marker in zip(dkeys, lstyles, colors, markers):
          tdict = df_dict[key]
          ydata = extract_params(tdict, 'fit_time')
          # ydata = [np.mean(val) for val in ydata]
          ax1.plot(dlengths, ydata, label=label_dict[key], 
                  linestyle=ls, color=color, marker=marker)
        
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlim(np.amin(dlengths), np.amax(dlengths))
        # ax1.set_xlim(, 10**7)
        ax1.set_ylabel('Fit Time (s)')
        ax1.set_xlabel('Data Length')
        
        # ax1.set_ylim(2 * 10**-4, 5)
        # ax1.set_xticklabels([])
        ax1.grid()
        ax1.legend()
        if savefig:
            plt.savefig('./results/comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        
    def plot_operation_speed(self, df_dict, dlengths, conversion_time, savefig=True):
        def order_labels(ax, olabels):
          handles, labels = ax.get_legend_handles_labels()
          handle_dict = dict(zip(labels, handles))
          ohandles = [handle_dict[label] for label in olabels]
          return ohandles
          
        all_time_keys = ['ftimes', 'jtimes', 'svd_times', 'gtimes', 'ptimes', 'ctimes']
        legend_labels = [r'$f(\mathbf{x})$', r'$J(\mathbf{x})$', 'SVD', r'$\nabla f(\mathbf{x})$', r'$m(\mathbf{w})$', 'Cost']
        legend_dict = dict(zip(all_time_keys, legend_labels))
        
        time_keys1 = ['ftimes', 'jtimes', 'svd_times']
        
        dkeys = ['JAX', 'SciPy']
        lstyles = ['-','--']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        markers = ['o', 's', '^', 'd', '+']
        fig, axes = plt.subplots(2, 1, figsize=(5, 8))
        
        ax1, ax2 = axes
        plabels = ['(a)', '(b)']
        for ax, plabel in zip(axes, plabels):
          ax.annotate(plabel, xy=(-.215, .975), 
                      xycoords='axes fraction', xytext=(0, 0), 
                      textcoords='offset points', ha='left', 
                      va='bottom', color='black', fontsize=12)
        
        for time_key, color, marker in zip(time_keys1, colors, markers):
          for key, ls in zip(dkeys, lstyles):
            tdict = df_dict[key]
            ydata = extract_params(tdict, time_key, ptype='time_param')
            ydata = [np.mean(val) for val in ydata]
            ax1.plot(dlengths, ydata, label=key + ' ' + legend_dict[time_key], 
                    linestyle=ls, color=color, marker=marker)
        
        ydata = extract_params(df_dict['JAX'], 'conversion_time')
        ax1.plot(dlengths, ydata, label='CPU to GPU', linestyle=':', color=colors[-2], 
                marker=markers[-2])
        
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlim(np.amin(dlengths), np.amax(dlengths))
        # ax1.set_xlim(10**4, 10**7)
        ax1.set_ylabel('Time (s)')
        # ax1.set_ylim(2 * 10**-4, 7)
        ax1.set_xticklabels([])
        ax1.grid()
        
        olabels1 = ['SciPy $f(\mathbf{x})$', 'SciPy $J(\mathbf{x})$', 'SciPy SVD', 'CPU to GPU', 
                       'JAX $f(\mathbf{x})$', 'JAX $J(\mathbf{x})$', 'JAX SVD']
        
        ohandles1 = order_labels(ax1, olabels1)
        ax1.legend(ohandles1, olabels1, ncol=2, frameon=True, loc='best', columnspacing=.25)
        
        
        time_keys2 = ['gtimes', 'ptimes']
        
        dkeys = ['JAX', 'SciPy']
        lstyles = ['-','--']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        markers = ['o', 's', '^', 'd', '+']
        # fig, ax = plt.subplots()
        
        for time_key, color, marker in zip(time_keys2, colors, markers):
          for key, ls in zip(dkeys, lstyles):
            tdict = df_dict[key]
            ydata = extract_params(tdict, time_key, ptype='time_param')
            ydata = [np.mean(val) for val in ydata]
            ax2.plot(dlengths, ydata, label=key + ' ' + legend_dict[time_key], 
                    linestyle=ls, color=color, marker=marker)
            
        ax2.plot(dlengths, conversion_time, label='GPU to CPU', 
                    linestyle=':', color=colors[3], marker=markers[3])
        
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_xlim(np.amin(dlengths), np.amax(dlengths))
        # ax2.set_xlim(10**4, 10**7)
        
        ax2.set_ylabel('Time (s)')
        ax2.set_xlabel('Data Length')
        ax2.grid()
        
        olabels2 = ['SciPy $\\nabla f(\mathbf{x})$', 'SciPy $m(\mathbf{w})$', 
                    'GPU to CPU', 
                    'JAX $\\nabla f(\mathbf{x})$', 'JAX $m(\mathbf{w})$']
        
        ohandles2 = order_labels(ax2, olabels2)
        ax2.legend(ohandles2, olabels2, ncol=2, frameon=True, loc='best', columnspacing=1)
        
        plt.subplots_adjust(hspace=0.05)
        
        if savefig:
            plt.savefig('./results/scipy_jax.png', dpi=300, bbox_inches='tight')
        plt.show()


def dict_results(results, index, fit_time):
    if len(results) == 5:
      iter_data_labels = ['original_index', 'fit_time', 'parameters', 'covariance', 
                          'fit_results', 'post_time', 'conversion_time']
    else:
      iter_data_labels = ['original_index', 'parameters', 'covariance']
    
    iter_data = [index, fit_time] + list(results)
    length_dict = dict(zip(iter_data_labels, iter_data))
    return length_dict

    
def get_row_data(row):
    params = row['parameters']
    seed = row['seed']
    img_dims = (row['height'], row['width'])
    noise_std = row['noise_std']
    return params, seed, img_dims, noise_std


def create_img_data(params, seed, img_dims, noise_std):
    XY_tuple = sp.get_coordinates(*img_dims)
    flat_XY = [coord.flatten() for coord in XY_tuple]
    flat_data = gaussian2d_np(flat_XY, *params)
    flat_data += np.random.normal(0, noise_std, flat_data.shape)
    return flat_XY, flat_data


length_samples = 11
length_min = 10 ** 3
length_max = 1 * 10**5
length_steps = 10
length_params = [length_samples, length_min, length_max, length_steps]
sp = SimulationParameters(length_params)
gaussian_data = sp.create_simulation_params()
jcf = CurveFit()
no_bounds = (-np.inf, np.inf)
jax_kwargs = {'bounds': no_bounds, 'method': 'trf', 'x_scale': 'jac', 'timeit': True}
scipy_kwargs = {'bounds': no_bounds, 'method': 'trf', 'x_scale': 'jac'}

lengths = gaussian_data['height'].unique()
num_lengths = len(lengths)
all_jax_dicts = []
all_scipy_dicts = []

for lindex, length in enumerate(lengths):
  print(length**2, lindex, 'of', num_lengths)
  mask = gaussian_data['height'] == length
  subdf = gaussian_data[mask]
  jax_length_dicts = []
  scipy_length_dicts = []

  slength = len(subdf)
  for index, row in subdf.iterrows():
      print(index, 'of', slength)
      params, seed, img_dims, noise_std = get_row_data(row)
      flat_XY, flat_data = create_img_data(params, seed, img_dims, noise_std)

      st = time.time()
      jax_results = jcf.curve_fit(gaussian2d_jax, flat_XY, flat_data, p0=seed, 
                                            **jax_kwargs) #call gpu fitting func
      st2 = time.time()
      scipy_results = curve_fit(gaussian2d_np, flat_XY, flat_data, p0=seed, 
                                           **scipy_kwargs) #call gpu fitting func
      scipy_fit_time = time.time() - st2
      jax_fit_time = st2 - st
      jax_length_dict = dict_results(jax_results, index, jax_fit_time)
      jax_length_dicts.append(jax_length_dict)
      scipy_length_dict = dict_results(scipy_results, index, scipy_fit_time)
      scipy_length_dicts.append(scipy_length_dict)

  all_jax_dicts.append(jax_length_dicts)
  all_scipy_dicts.append(scipy_length_dicts)

mod_container(all_jax_dicts)
mod_container(all_scipy_dicts)

df_dict = {'JAX': all_jax_dicts, 
           'SciPy': all_scipy_dicts}#, 'JAX_ut': jax_ut_df}
dlengths = lengths**2
conversion_time, _ = get_gpu_cpu_conversion(df_dict['JAX'])

#%%
pl = PlotResults()
pl.plot_operation_speed(df_dict, dlengths, conversion_time)
pl.plot_overall_speed(df_dict, dlengths)

        