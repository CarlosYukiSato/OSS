# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:56:18 2017

@author: carlos.sato
"""
###############################################################################
import cv2
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

SAVE_FILES_PATH = 'OSS_Files/Asa/filter256_'
HDR_FILE_PATH = 'HDR/Asa256.txt'

iterations = 3000
beta = 0.9
filtercount = 20
bFORCE_REAL = False
IM_HALFSIZE = 480

from scipy.fftpack import fftn, ifftn

def SaveImage(fname, Image):
	phases = np.angle(Image).astype(np.float32)
	absol = np.absolute(Image).astype(np.float32)
	phases[absol==0] = 0
	np.savetxt(fname + '_amp.txt', absol)
	np.savetxt(fname + '_phase.txt', phases)

	maxv = absol.max()
	Normalized = (255*np.sqrt((1.0/maxv)*absol)).astype(np.uint8)
	cv2.imwrite(fname+'.png', Normalized)
	
im = np.loadtxt(HDR_FILE_PATH, delimiter=' ')[499-IM_HALFSIZE:499+IM_HALFSIZE,683-IM_HALFSIZE:683+IM_HALFSIZE]
im[im<0] = 0
DifPad = np.sqrt(im)

Mask = np.zeros(DifPad.shape,dtype=np.bool)
#Mask[200:254,200:254] = True
#ImgMask = np.asarray(cv2.imread('PinholeMask.png', cv2.IMREAD_GRAYSCALE))
#Mask[ImgMask>0] = True
masksize = IM_HALFSIZE//9
for j in range(-masksize,+masksize+1):
	for i in range(-masksize,+masksize+1):
		if i*i + j*j < masksize*masksize:
			Mask[Mask.shape[0]//2+j,Mask.shape[1]//2+i] = True
import scipy.ndimage

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
imagSize = np.shape(DifPad);

R2D = []
for f in range(0,filtercount):
	R2D.append(gpuarray.zeros(imagSize,np.complex64))

toperrs = 1E15*np.ones(filtercount).astype(np.float32)
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
#sigma = np.flip(sigma, 0)

lastUsedSigma = -1.0
#############################

phase_angle = np.random.rand(DifPad.shape[0],DifPad.shape[1]).astype(np.float32)*2.0*np.pi

#Define initial k, r space
initial_k = scipy.fftpack.ifftshift(DifPad).astype(np.float32)
k_space = initial_k * np.exp(1j*phase_angle)

DifPad_gpu = gpuarray.to_gpu(initial_k)
k_space_gpu = gpuarray.to_gpu(k_space)

plan_forward = cu_fft.Plan(DifPad.shape, np.complex64, np.complex64)
plan_inverse = cu_fft.Plan(DifPad.shape, np.complex64, np.complex64)

r_space_gpu = gpuarray.zeros(DifPad.shape, np.complex64)
buffer_r_space = gpuarray.zeros(DifPad.shape, np.complex64)
cu_fft.ifft(k_space_gpu, buffer_r_space, plan_inverse, True)
MaskBoolean = gpuarray.to_gpu(Mask)

sample = gpuarray.zeros(k_space.shape, np.complex64)  
RfacF = np.zeros((iterations,1)).astype(np.float32);  

counter1=0; 
errorF=1;

iter = 1

if bFORCE_REAL:
	buffer_r_space = buffer_r_space.real.astype(np.complex64)
ZeroVector = gpuarray.zeros(DifPad.shape, np.float32)

print('Init time: ' + str(time()-curtime) + 's')
curtime = time()

from pycuda.elementwise import ElementwiseKernel
CuFFTShift = ElementwiseKernel(
"pycuda::complex<float> *x, pycuda::complex<float> *y",
"""
const int halfsize = """ + str(IM_HALFSIZE) + """;
const int imagesize = 2*halfsize;

int k,l,idx,jdx;

k = i%imagesize;
l = i/imagesize;

if(k < halfsize) 
	idx = k+halfsize;
else
	idx = k-halfsize;
if(l < halfsize)
	jdx = l+halfsize;
else
	jdx = l-halfsize;

x[i] = y[jdx*imagesize+idx];
""",
"CuFFTShift",
preamble="#include <pycuda-complex.hpp>",)

while (iter < iterations):

	sample = gpuif(MaskBoolean, r_space_gpu, sample)
	r_space_gpu = buffer_r_space - beta * r_space_gpu
    
	sample = gpuif(sample.real < 0, r_space_gpu, sample)
	r_space_gpu = gpuif(MaskBoolean, sample, r_space_gpu)
    
	#### OSS ####

	if (HIOfirst == 0 or iter > np.ceil(iterations/filtercount)) and iter < np.ceil(iterations-iterations/filtercount):
		newsigma = sigma[(iter-1)]
		if lastUsedSigma != newsigma:
			print(str(iter) + ' changing filter to ' + str(newsigma))
			kfilter = np.exp( -0.5*(yy**2 + xx**2) / newsigma**2 ).astype(np.complex64)
			kfilter_gpu.set(kfilter)
			temp_gpu = kfilter_gpu + 0.0
			CuFFTShift(kfilter_gpu,temp_gpu) 
			lastUsedSigma = newsigma


		cu_fft.fft(r_space_gpu, ktemp_gpu, plan_forward)
		ktemp_gpu = ktemp_gpu*kfilter_gpu
		cu_fft.ifft(ktemp_gpu, r_space_gpu, plan_inverse,True)

    		if np.mod(iterations,iter//filtercount)==0 and filtnum > 1 and toperrs[filtnum-1] < 1:
       			r_space_gpu = R2D[filtnum-1] + 0.0
   		else:
			r_space_gpu = gpuif(MaskBoolean,sample,r_space_gpu)
	##### ### ####

	buffer_r_space = r_space_gpu + 0.0;
    
	cu_fft.fft(r_space_gpu, k_space_gpu, plan_forward) 
	k_space_gpu = DifPad_gpu * (k_space_gpu / (k_space_gpu.__abs__()+1E-20))

	cu_fft.ifft(k_space_gpu, r_space_gpu, plan_inverse, True)
	if bFORCE_REAL:
		r_space_gpu = r_space_gpu.real.astype(np.complex64)
            
    #### OSS ERRORS ####

	if True:
		cu_fft.fft(sample,ktemp_gpu,plan_forward)
		errorF = gpusum(gpuif(DifPad_gpu,(ktemp_gpu.__abs__()-DifPad_gpu).__abs__(),ZeroVector))
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

os.system('rm ' + SAVE_FILES_PATH + '*.txt')
os.system('rm ' + SAVE_FILES_PATH + '*.png')
for viewiter in range(0,filtnum):
	SaveImage(SAVE_FILES_PATH + str(viewiter), R2D[viewiter].get())


plt.subplot(1,1,1)
plt.plot(RfacF)
plt.ylabel('Fourier R-factor')
plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig("RFactF.png",dpi=50)
plt.show()

