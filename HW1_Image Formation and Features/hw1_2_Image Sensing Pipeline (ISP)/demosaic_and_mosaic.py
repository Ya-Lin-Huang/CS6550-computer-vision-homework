
import numpy as np

from demosaic_2004 import demosaicing_CFA_Bayer_Malvar2004

def mosaic(img, pattern):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W numpy array, output image after mosaic.
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create the H*W output numpy array.                              #   
    #   2. Discard other two channels from input 3-channel image according #
    #      to given Bayer pattern.                                         #
    #                                                                      #
    #   e.g. If Bayer pattern now is BGGR, for the upper left pixel from   #
    #        each four-pixel square, we should discard R and G channel     #
    #        and keep B channel of input image.                            #     
    #        (since upper left pixel is B in BGGR bayer pattern)           #
    ########################################################################
    
    '''output = np.zeros((img.shape[0],img.shape[1]))
    
    if pattern == 'GRBG' or pattern == 'GBRG':
       output = img[:,:,1]
    
    if pattern == 'RGGB':
       output = img[:,:,0]
    
    if pattern =='BGGR':
        output = img[:,:,2]'''
        
    H=img.shape[0]
    W=img.shape[1]
    array_1=np.array(([1,0],[0,1]))
    array_1=np.tile(array_1,(int(H/2),int(W/2)))
    
    array_2=np.array(([0,1],[0,0]))
    array_2=np.tile(array_2,(int(H/2),int(W/2)))
    
    array_3=np.array(([0,0],[1,0]))
    array_3=np.tile(array_3,(int(H/2),int(W/2)))

    array_4=np.array(([0,1],[1,0]))
    array_4=np.tile(array_4,(int(H/2),int(W/2)))   
    
    array_5=np.array(([1,0],[0,0]))
    array_5=np.tile(array_5,(int(H/2),int(W/2)))
    
    array_6=np.array(([0,0],[0,1]))
    array_6=np.tile(array_6,(int(H/2),int(W/2)))
   
    
    if pattern == 'GRBG' :
        output=img[:,:,0]*array_2+img[:,:,1]*array_1+img[:,:,2]*array_3    
        
    elif pattern == 'RGGB':
        output=img[:,:,0]*array_5+img[:,:,1]*array_4+img[:,:,2]*array_6    
        
    elif pattern == 'GBRG':
        output=img[:,:,0]*array_3+img[:,:,1]*array_1+img[:,:,2]*array_2  
        
    elif pattern == 'BGGR':
        output=img[:,:,0]*array_6+img[:,:,1]*array_4+img[:,:,2]*array_5   
    
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################

    return output


def demosaic(img, pattern):
    '''
    Input:
        img: H*W numpy array, input RAW image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W*3 numpy array, output de-mosaic image.
    '''
    #### Using Python colour_demosaicing library
    #### You can write your own version, too
    output = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
    output = np.clip(output, 0, 1)

    return output

