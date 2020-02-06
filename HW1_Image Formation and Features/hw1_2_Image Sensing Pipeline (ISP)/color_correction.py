
import cv2
import skimage
import numpy as np

def BGR2RGB(img):
    '''
    Input:
        img: H*W*3, input BGR image
    Output:
        output: H*W*3, output RGB image
    '''
    b, g, r = cv2.split(img)
    output = cv2.merge([r, g, b])
    return output

def RGB2BGR(img):
    '''
    Input:
        img: H*W*3, input RGB image
    Output:
        output: H*W*3, output BGR image
    '''
    r, g, b = cv2.split(img)
    output = cv2.merge([b, g, r])
    return output

def RGB2XYZ(img):
    '''
    Input:
        img: H*W*3, input RGB image
    Output:
        output: H*W*3, output CIE XYZ image
    '''
    output = skimage.color.rgb2xyz(img)
    return output

def XYZ2RGB(img):
    '''
    Input:
        img: H*W*3, input CIE XYZ image
    Output:
        output: H*W*3, output RGB image
    '''
    output = skimage.color.xyz2rgb(img)
    return output


def color_correction(img, ccm):
    '''
    Input:
        img: H*W*3 numpy array, input image
        ccm: 3*3 numpy array, color correction matrix 
    Output:
        output: H*W*3 numpy array, output image after color correction
    '''
    ########################################################################
    # TODO:                                                                #
    #   Following the p.22 of hw1_tutorial.pdf to get P as output.         #
    #                                                                      #
    ########################################################################
    output = np.zeros((img.shape[0],img.shape[1],3))
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            original_space = np.zeros((3,1))
            original_space[0] = img[i,j,0]
            original_space[1] = img[i,j,1]
            original_space[2] = img[i,j,2]
            color_space = np.dot(ccm.T,original_space)

            output[i,j,0] = color_space[0]
            output[i,j,1] = color_space[1]
            output[i,j,2] = color_space[2]
    
    '''H=img.shape[0]
    W=img.shape[1]
    
    
    img_r=img.reshape(-1,3)
    output=np.dot(img_r,ccm)
    output=output.reshape(H,W,img.shape[2])'''
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    
    #### Prevent the value larger than 1 or less than 0
    output = np.clip(output, 0, 1)
    return output
