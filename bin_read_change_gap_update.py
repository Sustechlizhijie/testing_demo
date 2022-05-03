import os
import re
import h5py
import shutil
import numpy as np

# fPath: directory to store data
# prefix: "vel"
# step: 50000
# Nx: 64
# Ny: 64
# lz: N/nproc=64/32=2
# nvar: 3, u,v,w
# outN: 64
# nproc: 32
# fmt: '<f4'

fpath =  "./fDNS_G64_R3_phy"
prefix = "phy_vor"
# step =  50000
Nx= 64
Ny= 64
lz =2
nvar = 3
outN = 64
nproc = 32
# fmt: '<f4'
fmt = '<c8'  #左对齐8个字符


def read_sol_bin(fpath, prefix, step, lx1, ly, lz, nvar, outN, nproc, fmt):
    lx = lx1 - 1  
    lx2 = lx*2
    cdat = np.zeros((lx1, ly, lz, nvar), dtype=np.complex64)
    rdat = np.zeros((lx2, ly, lz, nvar), dtype=np.float32)
    data = np.zeros((outN, outN, outN, nvar), dtype=np.float32)

    oriN = ly
    skip = oriN//outN
    num  = lx1*ly*lz
    print("step={},lx1={},ly={},lz={},skip={},oriN={},outN={}".format(step,lx1,ly,lz,skip,oriN,outN))
    if skip <= lz: 
        nlz = lz//skip
        for i in np.arange(nproc): 
            fname = os.path.join(fpath, \
                "{}{:0>8d}.{:0>3d}".format(prefix, step, i))
            f = open(fname, 'rb')
            for j in np.arange(nvar):
                f.read(4)   
                #### read the Fortran-order array must read as       ####
                #### invert-sorted the shape and transpose the array ####     
                cdat[:,:,:,j] = np.fromfile(file=f, dtype=fmt, \
                    count=num).reshape((lz, ly, lx1)).transpose()
                f.read(4)
            f.close()
            rdat[0::2, :, :, :] = np.real(cdat[:lx, :, :, :])          
            rdat[1::2, :, :, :] = np.imag(cdat[:lx, :, :, :]) 
            data[:, :, i*nlz:(i+1)*nlz, :] = rdat[::skip, ::skip, ::skip, :]
    return data

# gaps = [20,50,100,500,800] # step interval 

sample_list = [] 
for j in range(0,1000,10):   #0-490间隔10，共50组样本

    step_list  = []
    for i in range(35):  #i=60个时刻
        step = 60000 + i*500 + 10+j  #起始60000，gap200，样本j
        print(step)
        data_sample = read_sol_bin(fpath, prefix, step, Nx//2+1, Ny, lz, nvar, outN, nproc, fmt)
        step_list.append(data_sample)

    sample_list.append(step_list)

np.save('3d_vor_100_35time_gap500_LES_64.npy',np.asarray(sample_list))

    
