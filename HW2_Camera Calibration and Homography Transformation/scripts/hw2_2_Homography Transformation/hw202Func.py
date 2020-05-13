# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:47:16 2019

@author: Lily
"""

import numpy as np
from scipy import linalg, matrix
import matplotlib.pyplot as plt
import copy

def homography(left,right):
    '''
    Inputs:
        left: 
        right:
    Outputs:
        H: the homography matrix 
    '''
    A = []
    for i in range(len(left)):
        # left points
        x = left[i,0]
        y = left[i,1]
        # right points
        u = right[i,0]
        v = right[i,1]
        
        tmp = [[x, y, 1, 0, 0, 0, -x*u, -y*u],
               [0, 0, 0, x, y, 1, -x*v, -y*v]]
        A.append(tmp)
    # compute eigenvector corresponding to smallest eigenvalue 
    A= np.vstack(A)
    A = np.matrix(A)
    b = right.reshape(-1,1)
    h = np.dot(linalg.inv(A),b)
    h = np.append(h,1)
    H = h.reshape(3,3)
    
#    A = np.matrix(A)
#    ATA = A.T*A
#    value, vector = linalg.eig(ATA)
#    H = vector[np.argmin(value)]
#    H=H.reshape(3,3)
    return H

def forward(pts_in ,pts_out, in_img, out_img):
    '''
    Inputs:
        pts_in: the points of the image to be transforomed
        pts_out: the points of the output image location 
        input_img: the image corresponding to pts_in 
        canvas: the image corresponding to pts_out
    Output:
        canvas: the image after forward warping
    '''
    canvas = copy.deepcopy(out_img)
    H = homography(pts_in,pts_out)
    # find the iteration range
    X_max = pts_in[:,0].max()
    X_min = pts_in[:,0].min()
    Y_max = pts_in[:,1].max()
    Y_min = pts_in[:,1].min()

    for x in range(X_max-X_min):
        for y in range(Y_max-Y_min):
            loc_in = [X_min+x, Y_min+y, 1]
            loc_out = np.dot(H,np.array(loc_in).T)
#            u,v = np.ceil(loc_out[0]/loc_out[2]).astype(int) , np.ceil(loc_out[1]/loc_out[2]).astype(int)
            u,v = int(loc_out[0]/loc_out[2]) , int(loc_out[1]/loc_out[2])
            canvas[v][u] = in_img[Y_min+y][X_min+x]
    return canvas


def interpolation(img, new_x, new_y):
    fx = round(new_x - int(new_x), 3)
    fy = round(new_y - int(new_y), 3)
    tmp = np.zeros(3)
    tmp += (1 - fx) * (1 - fy) * img[int(new_y), int(new_x)] 
    tmp += (1 - fx) * fy * img[int(new_y) + 1, int(new_x)]
    tmp += fx * (1 - fy) * img[int(new_y), int(new_x) + 1]
    tmp += fx * fy * img[int(new_y) + 1, int(new_x) + 1] 
    pixel = tmp
    return pixel
'''
def interpolation2(img,x,y):
    #calculate weights
    w = np.ceil(x)-np.floor(x)
    h= np.ceil(y)-np.floor(y)
    left_w = (np.ceil(x)-x)/w
    right_w = (x-np.floor(x))/w
    up_w =(np.ceil(y)-y)/h
    down_w = (y-np.floor(y))/h
    # interpolation
    top_left_color = img[np.floor(y).astype(int), np.floor(x).astype(int),:]
    top_right_color = img[np.floor(y).astype(int), np.ceil(x).astype(int),:]
    bottom_left_color =img[np.ceil(y).astype(int), np.floor(x).astype(int),:]
    bottom_right_color = img[np.ceil(y).astype(int),np.ceil(y).astype(int),:]
    up_weigh_color = left_w*top_left_color+right_w*top_right_color
    bottom_weigh_color = left_w*bottom_left_color+right_w*bottom_right_color
    
    pixel = up_w*up_weigh_color+down_w*bottom_weigh_color
    
    return pixel
'''
def checkShape(in_img,out_img):
    '''
    觀察到大圖放到小圖會有IndexError(超出小圖的邊界)，故讓大圖乘上一個縮小的倍數
    Inputs:
    '''
    shape1 = np.shape(in_img)
    shape2 = np.shape(out_img)
    # X-axis
    Nx = shape1[1]/shape2[1]
    # Y-axis
    Ny = shape1[0]/shape2[0]   
    adj = 1  #     
    re=1
    if Nx > 1.2:
        re1 = 1/Nx #resize
        if re > re1:
            re=re1
    elif Ny > 1.2:
        re1 = 1/Ny
        if re > re1:
            re=re1
    else:
        re = re 
    return re*adj

def backward(pts_in ,pts_out, in_img, out_img):
    canvas = copy.deepcopy(out_img)
    H = homography(pts_out,pts_in)
    # find the iteration range
    X_max = pts_out[:,0].max()
    X_min = pts_out[:,0].min()
    Y_max = pts_out[:,1].max()
    Y_min = pts_out[:,1].min()
#    re = checkShape(in_img, out_img)
    for x in range(X_max-X_min):
        for y in range(Y_max-Y_min):
            loc_in = [(X_min+x), (Y_min+y),1 ]     #out_img上的位置
            loc_new = np.dot(H,np.array(loc_in).T) #對應in_img圖上的位置
            u_,v_ = int(loc_new[0]/loc_new[2]) , int(loc_new[1]/loc_new[2])
            pixel = interpolation(in_img,u_,v_)
            
            v,u = int((Y_min+y)), int((X_min+x))
            canvas[v][u] = pixel
#    plt.imshow(canvas)
#    plt.show()        
    return canvas



