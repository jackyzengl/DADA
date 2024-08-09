import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))

def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)

def bool_flag(s):  # *
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def lineno():
    return str(inspect.currentframe().f_back.f_lineno)

def get_total_norm(parameters, 
                   norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm

@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))

def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem

# *
def get_dset_path(subset_name, subset_type, 
                  datasets_suffix=''):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'DLA', 
                        'datasets_dla'+datasets_suffix, 
                        subset_name, subset_type)

# *
def relative_to_abs(rel_traj,  # [20, nums, 2]
                    start_pos):  # [nums, 2]
    rel_traj = rel_traj.permute(1, 0, 2)  # [nums, 20, 2]
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)  # [nums, 1, 2]
    abs_traj = displacement + start_pos  # [nums, 20, 2]
    return abs_traj.permute(1, 0, 2)  # [20, nums, 2]

class new_zip(object):
    def __init__(self, 
                 a_loader, b_loader):
        self.a_loader = a_loader
        self.b_loader = b_loader
        self.len_a = len(a_loader)
        self.len_b = len(b_loader)
        self.len_max = max(self.len_a, self.len_b)
        self.index = 0
    
    def __next__(self):
        if self.index < self.len_max:
            a_batch = self.a_loader[self.index % self.len_a]
            b_batch = self.b_loader[self.index % self.len_b]
            self.index += 1
            return (a_batch, b_batch)
        else:
            raise StopIteration
    
    def __iter__(self):
        return self
    
"""
a = [torch.randn([3, 5, 2]), torch.randn([3, 4, 2]), torch.randn([3, 3, 2])]
b = [torch.randn([3, 2, 2]), torch.randn([3, 1, 2]), torch.randn([3, 7, 2]), 
     torch.randn([3, 6, 2]), torch.randn([3, 8, 2]), torch.randn([3, 9, 2]), torch.randn([3, 10, 2])]
for ba, bb in new_zip(a, b):
    print(ba.shape)
    print(bb.shape)
    print('**************')
"""