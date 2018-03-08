# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:56:18 2017

@author: carlos.sato
"""
###############################################################################
import cv2
import os

def ViewImage(fname, Img, mult):
	np.savetxt(fname + '.txt', Img)
	Image = np.loadtxt(fname + '.txt', delimiter=' ')
	maxv = Image.max()
	Normalized = 255*np.sqrt((mult/maxv)*Image)
	cv2.imwrite(fname + '.png', Normalized)
	os.system("xdg-open " + fname + ".png")

from PIL import Image
im = Image.open('/home/sato/Documents/MATLAB/Models/Lena.tiff')
    
import numpy as np
import scipy

from scipy.fftpack import fftn, ifftn

Lena = np.array(im)[256:384,256:384]
del(im)
Lena =0.2989 * Lena[:,:,0] + 0.5870 * Lena[:,:,1] + 0.1140 * Lena[:,:,2]

overSampling = 3

Ov = np.array([overSampling,overSampling]).astype(np.float32)
Ov = ((Ov-1)/2)*(np.shape(Lena))
Ov = np.int16(Ov)
Lena = np.pad(Lena,Ov, 'constant', constant_values=(0,0)).astype(np.float32)

DifPad = scipy.absolute(scipy.fftpack.fft2(Lena))
DifPad = scipy.fftpack.fftshift(DifPad)

Mask = Lena > 0
#Mask = np.zeros(Lena.shape,dtype=np.bool)
#Mask[Lena.shape[0]//3:2*Lena.shape[0]//3,Lena.shape[1]//3:2*Lena.shape[1]//3] = True
#Mask[400:1000,400:1000] = True

###############################################################################

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import skcuda
import numpy as np
import scipy
from pycuda.gpuarray import if_positive as gpuif
from pycuda.gpuarray import sum as gpusum
from time import time

curtime = time()

iterations = 1000
beta = 0.9

##########  OSS  ##################

filtercount = 10 #iterations/100
imagSize = np.shape(DifPad);

R2D = []
for f in range(0,filtercount):
	R2D.append(gpuarray.zeros(imagSize,np.complex64))

toperrs = 1000*np.ones(filtercount).astype(np.float32)
filtnum = 0
store = 0
HIOfirst = 1
kfilter_gpu = gpuarray.zeros(imagSize,np.complex64)
ktemp_gpu = gpuarray.zeros(imagSize,np.complex64)
x = np.arange(-imagSize[1]//2 , imagSize[1]//2, 1)
y = np.arange(-imagSize[0]//2 , imagSize[0]//2, 1)
xx, yy = np.meshgrid(x, y, sparse=True,copy=True)

X = np.array(range(1,iterations+1))
sigma = (filtercount-np.ceil(X*filtercount/iterations))*np.ceil(iterations/filtercount)
sigma = ((sigma-np.ceil(iterations/filtercount))*(2*imagSize[0])/np.max(sigma))+(2*imagSize[0]/10)

lastUsedSigma = -1.0
#############################



phase_angle = np.random.rand(DifPad.shape[0],DifPad.shape[1]).astype(np.float32)*2.0*np.pi

#Define initial k, r space
initial_k = scipy.fftpack.ifftshift(DifPad).astype(np.float32)
k_space = initial_k * np.exp(1j*phase_angle)

DifPad_gpu = gpuarray.to_gpu(initial_k)
k_space_gpu = gpuarray.to_gpu(k_space)

plan_forward = cu_fft.Plan(Lena.shape, np.complex64, np.complex64)
plan_inverse = cu_fft.Plan(Lena.shape, np.complex64, np.complex64)

r_space_gpu = gpuarray.zeros(Lena.shape, np.complex64)
buffer_r_space = gpuarray.zeros(Lena.shape, np.complex64)
cu_fft.ifft(k_space_gpu, buffer_r_space, plan_inverse, True)
MaskBoolean = gpuarray.to_gpu(Mask)

sample = gpuarray.zeros(k_space.shape, np.complex64)

RfacF = np.zeros((iterations,1)).astype(np.float32);  
counter1=0; 
errorF=1;

iter = 1

buffer_r_space = buffer_r_space.real.astype(np.complex64)
ZeroVector = gpuarray.zeros(Lena.shape, np.float32)

print('Init time: ' + str(time()-curtime) + 's')
curtime = time()

from pycuda.elementwise import ElementwiseKernel
CuFFTShift = ElementwiseKernel(
"pycuda::complex<float> *x, pycuda::complex<float> *y",
"""
int k,l,idx,jdx;
k = i%384;
l = i/384;
if(k < 192) 
	idx = k+192;
else
	idx = k-192;
if(l < 192)
	jdx = l+192;
else
	jdx = l-192;
x[i] = y[jdx*384+idx];
""",
"CuFFTShift",
preamble="#include <pycuda-complex.hpp>",)

while (iter < iterations):
	cu_fft.ifft(k_space_gpu, r_space_gpu, plan_inverse, True)
	r_space_gpu = r_space_gpu.real.astype(np.complex64)
	    
	sample = gpuif(MaskBoolean, r_space_gpu, sample)
	r_space_gpu = buffer_r_space - beta * r_space_gpu
    
	sample = gpuif((sample.real < 0).astype(np.bool), r_space_gpu, sample)
	r_space_gpu = gpuif(MaskBoolean, sample, r_space_gpu)
    
	#### OSS ####

	if (HIOfirst == 0 or iter > np.ceil(iterations/filtercount)) and iter < np.ceil(iterations-iterations/filtercount):
		newsigma = sigma[(iter-1)]
		if lastUsedSigma != newsigma:
			print(str(iter) + ' changing filter to ' + str(newsigma))
			kfilter = np.exp(-(((np.sqrt((yy)**2+(xx)**2)**2))/(2*(newsigma)**2))).astype(np.complex64)
			kfilter_gpu.set(kfilter)
			temp_gpu = kfilter_gpu + 0.0
			cu_fft.fft(kfilter_gpu,temp_gpu,plan_forward)
			temp_gpu = temp_gpu/gpuarray.max(temp_gpu.real).get().astype(np.float32).astype(np.complex64)
			
			#CuFFTShift(kfilter_gpu,temp_gpu) # Desnecessauro	
			lastUsedSigma = newsigma

		cu_fft.fft(r_space_gpu, ktemp_gpu, plan_forward)
		ktemp_gpu = ktemp_gpu*kfilter_gpu
		cu_fft.ifft(ktemp_gpu, r_space_gpu, plan_inverse,True)
        
    		if np.mod(iterations,iter//filtercount)==0:
       			r_space_gpu = R2D[filtnum-1] + 0.0
   		else:
			r_space_gpu = gpuif(MaskBoolean,sample,r_space_gpu)

	##### ### ####

	buffer_r_space = r_space_gpu + 0.0;
    
	cu_fft.fft(r_space_gpu, k_space_gpu, plan_forward) 
	k_space_gpu = DifPad_gpu * (k_space_gpu / (k_space_gpu.__abs__()+1E-10))

	if gpusum(k_space_gpu.real).get() == 0:
		print(iter)
		print('EXITING FOR 0')
		iter = 999999999
            
    #### OSS ERRORS ####

	if iter < 9999999:
		cu_fft.fft(sample,ktemp_gpu,plan_forward)
		errorF = gpusum(gpuif(DifPad_gpu,ktemp_gpu.__abs__()-DifPad_gpu,ZeroVector))
		errorF = errorF / gpusum(gpuif(DifPad_gpu, DifPad_gpu, ZeroVector))
		errorF = errorF.get().astype(np.float32)
    		RfacF[counter1] = errorF

    		#Determine iterations with best error
    		filtnum = np.ceil(iter * filtercount / iterations).astype(np.int32);
      		
		if errorF <= toperrs[filtnum-1] and iter > store + 2:
        		toperrs[filtnum-1] = errorF
        		R2D[filtnum-1] = r_space_gpu + 0.0
        		store = iter

	counter1+=1;
	iter = iter + 1

print('Run in: ' + str(time()-curtime) + ' s')
ViewImage("buffer_r_space", np.real(r_space_gpu.get()),1)
#for iter_f in range(0,toperrs.shape[0]):
#	ViewImage("R2D" + str(toperrs[iter_f]), np.real(R2D[iter_f].get()),1)

