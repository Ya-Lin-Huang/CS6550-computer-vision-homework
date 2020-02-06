# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:21:17 2019

@author: AndyLEE
"""
import cv2
import numpy as np
import math
from scipy import signal
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import color
import math

window_size = (3,30)
k = 0.04
thresh = 3e-6




#======================  FUNCTION  ======================#
def read_image(filename):
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    
    return img

def gaussian_smooth(img,sigma,ker):
    #filter images with Gaussian blur
    cx,cy =  (0+ker-1)/2,(0+ker-1)/2
    x,y =  np.meshgrid(np.arange(ker)-cx, np.arange(ker)-cy)
    gaussian_kernel = np.exp(-((x**2+y**2)/2*sigma**2)/(2*math.pi*(sigma**2)))
    gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
    img_gauss = np.zeros((img.shape[0],img.shape[1]))
    img_gauss= signal.convolve2d(img,gaussian_kernel,'same')
    return img_gauss

def sobel_edge_detection(dy,dx,h,w):

    mag = np.sqrt(dx**2+dy**2)  #edge strength
    theta =np.arctan2(dy,dx)
    hsv = np.zeros((h,w,3))
    hsv[:,:,0] = (theta + np.pi)/(2*np.pi) 
    hsv[:,:,1] = np.ones((h,w)) #saturation
#    hsv[:,:,0]= 1
    hsv[:,:,2] = (mag-mag.min())/(mag.max()-mag.min()) #magnitude normalization
    rgb = color.hsv2rgb(hsv)
    return mag,rgb

def structure_tensor(dy,dx,window_size):
    Ixx= dx**2
    Iyy= dy**2
    Ixy= dx*dy
    #==== gaussian====#
    Axx = gaussian_smooth(Ixx,5,window_size)
    Ayy = gaussian_smooth(Iyy,5,window_size)
    Axy = gaussian_smooth(Ixy,5,window_size)
    #==== mask 1 ====#
#    Axx = Ixx.sum()
#    Ayy = Iyy.sum()
#    Axy = Ixy.sum()
    
    return Axx,Ayy,Axy

def harris_oper(Axx , Ayy, Axy, k = 0.04):
     #A=[[Axx Axy],[Axy Ayy]]
    det = Axx*Ayy - Axy**2
    trace = Axx + Ayy
    R = det - k*(trace**2)
    return R

def nms(R,window_size=30):
    mask1 = (R>thresh)
    mask2 = (np.abs(ndi.maximum_filter(R,size = window_size))-R <1e-8) 
    mask = (mask1&mask2)
    return mask

def rotate(theta):
    
    return

def scale():
    
    return

# =============================================================================
#  gaussian_smooth
# =============================================================================
img1 = read_image("./original.jpg") 
for i in range(5,15,5):
    img_gauss = gaussian_smooth(img1,5,i)
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(img_gauss,cmap ='gray')
    plt.savefig('./results/gaussian_smooth_ker-{}'.format(i),dpi=300)
# =============================================================================
#  sobel_edge_detection
# =============================================================================
    dy, dx = np.gradient(img_gauss)
    h,w = img_gauss.shape
    
    mag,sobel_rgb = sobel_edge_detection(dy,dx,h,w)
    plt.axis('off')
    plt.imshow(mag,cmap = 'gray' ) # mag
    plt.savefig('./results/spbel_mag_ker-{}'.format(i),dpi=300)
    plt.show()
    
    plt.axis('off')
    plt.imshow(sobel_rgb)
    plt.savefig('./results/spbel_rgb_ker-{}'.format(i),dpi=300)
    plt.show()

# =============================================================================
#  Structure tensor + Harris operater + NMS
# =============================================================================
for win in window_size:
    Axx,Ayy,Axy = structure_tensor(dy,dx,win)
    R = harris_oper(Axx , Ayy, Axy, k = 0.04 )
    mask = nms(R,window_size=win)
    mask=np.nonzero(mask)
    mask = np.array(mask)
#    mask2 = np.zeros(mask.shape)
    
#    for i,row in enumerate(mask):
#        for j,boo in enumerate(row):
#            if j == True:
#                mask2[i,j] = 1
            
                
    fig3,ax = plt.subplots()    
    plt.axis('off')
    plt.imshow(img_gauss,cmap ='gray')
    plt.plot(mask,color='r',markersize=3)
    plt.show()

##==== VISUALIZATION ====##

        
    
#img2 = cv2.imread("./original.jpg")
#plt.imshow(img2)








