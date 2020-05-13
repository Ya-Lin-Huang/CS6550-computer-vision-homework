# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:57:46 2019

@author: Lily
"""

import numpy as np
import pandas as pd
from scipy import linalg, matrix
import scipy
import cv2
import matplotlib.pyplot as plt
#%% 1-a
def normalization(dim,vec):
    '''
    Normalize the coordinates of pts_2d & pts_3d
    center
    Inputs:
        dim: dimensions ( 2 or 3 )
        vec: the vec to be normalized
    Outputs:
        Tvec: the transformation matrix (will used for latter re-projection to image)
        vec2: the vector after transformation
    '''

    vec = np.asarray(vec)
    mean, std = np.mean(vec,0), np.std(vec )
    if dim==2:
        Tr = np.matrix([[std, 0, mean[0]],[0, std, mean[1]],[0, 0  , 1]])
    else:
        Tr = np.matrix([[std, 0, 0, mean[0]], [0, std, 0, mean[1]], [0, 0, std, mean[2]], [0, 0, 0, 1]])
    
    Tvec = np.linalg.inv(Tr)
    vec2 = np.dot( Tvec, np.concatenate( (vec.T, np.ones((1,vec.shape[0]))) ) )
    vec2 = vec2[0:dim,:].T
    return Tvec, vec2

def computePmatrix1(loc_2d,loc_3d,norT_2d, norT_3d,method=0):
    '''
    gernerate A matrix : Ap = 0
    Inputs:
        loc_2d: 2d data normalize 
        loc_3d: 3d data normalize
        method: 0-> P from eigenvector of (A.T*A)
                1-> P from SVD of A
    Outputs:
        P: P matrix
    '''
    def Amaker(length):
        A_all=[]
        for i in range(length): 
            x = loc_2d[i,0]
            y = loc_2d[i,1]
            X = loc_3d[i,0]
            Y = loc_3d[i,1]
            Z = loc_3d[i,2]
            A = np.vstack(([X,Y,Z,1,0,0,0,0,-x*X,-x*Y,-x*Z,-x],
                           [0,0,0,0,X,Y,Z,1,-y*X,-y*Y,-y*Z,-y]))
            A_all.append(A)
        return np.vstack(A_all) 
    # P matrix
    if method == 0:
        # method 1: eigen vector of A.T*A
        A = Amaker(len(loc_2d))
        ATA = np.dot(np.matrix(A).T, A)
        value, vector = linalg.eig(ATA)
        P0 = vector[np.argmin(value)]
    else:
        # method 2: SVD
        A = Amaker(len(loc_2d))
        U,sigma,VT=linalg.svd(A)
#        P0=VT.T[np.argmin(sigma)]   
        P0 = VT[-1,:]
    
    P0 = P0.reshape(3,4)   #re-build projection matrix, P by P vector
    #Denormalization:
    P = np.dot( np.dot( np.linalg.inv(norT_2d), P0 ), norT_3d )

    return P

def project(mat,loc_3d,loc_2d):
    '''
    project 3d points to 2d points
    Inputs:
        mat: projection matrix.
        loc_3d: original 3d_points.
        loc_2d: original 2d_points.
        norT: the matrix T used in normalization, 
        and will denormalize the projected points to the coordinate of the orignal 2d points. 
    Outputs:
        proj_2ds: 2d points projected from 3d points
        real_2ds: 2d points got from clicker.py
    '''
    proj_2ds=[]
    real_2ds = []
    loc_3d = np.array(loc_3d)
    loc_2d = np.array(loc_2d)
    for i in range(len(loc_3d)):
        vec = np.append(loc_3d[i],1)
        real =np.append(loc_2d[i],1)
        proj_2d =  np.dot(mat,vec)
        proj_2d = proj_2d/proj_2d.reshape(-1,1)[2]
        proj_2ds.append(proj_2d)
        real_2ds.append(real)
        
    return proj_2ds, real_2ds
#%% 1-b
'''
def KRT(estP):
    r, q = linalg.rq(estP[:,:-1])
    Rhead = r           
    Qhead = q           
    D = np.matrix([[np.sign(Rhead[0,0]), 0, 0],
                   [0,np.sign(Rhead[1,1]),0],
                   [0,0,np.sign(Rhead[2,2])]])
    K = np.dot(Rhead,D) # 上三角 #positive focal length
    R = np.dot(D,Qhead) # 'R' represent for rotation matrix   
    K = K/K[-1,-1]  #scale K such that K33 = 1
    T = np.dot( np.linalg.inv(K), estP[:,-1])
    return K,R,T
'''
def KRT(estP):
    tmp = estP[2,:3]
    nor = np.linalg.norm(tmp)
    estP=estP/nor
    r, q = linalg.rq(estP[:,:-1])
    Rhead = r           
    Qhead = q           
    D = np.matrix([[np.sign(Rhead[0,0]), 0, 0],
                   [0,np.sign(Rhead[1,1]),0],
                   [0,0,np.sign(Rhead[2,2])]])
    K = np.dot(Rhead,D) # 上三角 #positive focal length
    R = np.dot(D,Qhead) # 'R' represent for rotation matrix   
    K = K/K[-1,-1]  #scale K such that K33 = 1
    T = np.dot( np.linalg.inv(K), estP[:,-1])
    return K,R,T
#%% 1-c
def plot2img(click_pts, proj_pts, path,num):
    click_pts = np.asarray(click_pts).squeeze()
    proj_pts = np.asarray(proj_pts).squeeze()
    X=click_pts[:,0]
    Y=click_pts[:,1]
    
    U=proj_pts[:,0]
    V=proj_pts[:,1]
    
    imge = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)
    plt.imshow(imge)
    for x,y,u,v in zip(X,Y,U,V):
        plt.scatter(x,y,s = 10,c = 'yellow',alpha = 0.8)
        plt.scatter(u,v,s = 3,c = 'red',alpha = 0.8)
#        plt.savefig('./results/re-project2d_P{}.png'.format(num+1))
    plt.show()
#%% 1-d

#%% =======================================================================#
#     OTHER FUNCTION                                                     #
# =======================================================================#
def rmse(predictions, targets):
    predictions = np.array(predictions).squeeze()
    targets = np.array(targets).squeeze()
    return np.sqrt(((predictions - targets) ** 2).mean())

def norm(M): 
    return np.sqrt(float(np.sum(list(map(lambda x: x*x.T, M)))))

