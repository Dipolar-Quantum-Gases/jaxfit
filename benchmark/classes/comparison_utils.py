# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:11:17 2022

@author: hofer
"""
import numpy as np

def mod_container(elitem, container=None, ind=None):
  if isinstance(elitem, list):
    for index, item in enumerate(elitem):
      mod_container(item, elitem, index)
  elif isinstance(elitem, dict):
    for key, item in elitem.items():
      mod_container(item, elitem, key)
  elif isinstance(elitem, np.ndarray):
    container[ind] = elitem.tolist()
    

def extract_params(all_dicts, param, ptype='param', exclude_first=True, avg=True, only_first=False):
  if exclude_first or avg:
    only_first = False

  param_list_list = []
  for llist in all_dicts:
    param_list = []
    for index, idict in enumerate(llist):
      if ptype == 'param':
        val = idict[param]
      elif ptype == 'fit_param':
        val = idict['fit_results'][param]
      elif ptype == 'time_param':
        val = idict['fit_results']['all_times'][param]
      param_list.append(val)

    if exclude_first:
      param_list = param_list[1:]
    if only_first:
      param_list = param_list[0]
    if avg:
      if any(isinstance(el, list) for el in param_list):
        param_list = [item for sublist in param_list for item in sublist]
      param_list = np.mean(param_list)
    param_list_list.append(param_list)

  return param_list_list


def get_gpu_cpu_conversion(jax_dicts):
    time_keys = ['g_ctimes', 'svd_ctimes', 'c_ctimes',  'p_ctimes']
    conversion_times = []
    for time_key in time_keys:
      ydata = extract_params(jax_dicts, time_key, ptype='time_param')
      conversion_times.append(ydata)
      
    conversion_times = np.stack(conversion_times)
    conversion_time = np.sum(conversion_times, axis=0)
    return conversion_time, conversion_times