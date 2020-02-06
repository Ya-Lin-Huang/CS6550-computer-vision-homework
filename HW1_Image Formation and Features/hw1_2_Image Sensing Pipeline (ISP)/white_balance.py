
import numpy as np

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################
    mask = np.zeros((img.shape[0],img.shape[1]))
   
    if pattern == 'GRBG':
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                if i % 2 == 0:
                    if j%2 == 0:
                        mask[i,j] = 1
                    else:
                        mask[i,j] = fr
                else:
                    if j%2 == 0:
                        mask[i,j] = fb
                    else:
                        mask[i,j] = 1
    
    if pattern == 'RGGB':
         for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                if i % 2 == 0:
                    if j%2 == 0:
                        mask[i,j] = fr
                    else:
                        mask[i,j] = 1
                else:
                    if j%2 == 0:
                        mask[i,j] = 1
                    else:
                        mask[i,j] = fb
                        
    
    if pattern =='GBRG':
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                if i % 2 == 0:
                    if j%2 == 0:
                        mask[i,j] = 1
                    else:
                        mask[i,j] = fb
                else:
                    if j%2 == 0:
                        mask[i,j] = fr
                    else:
                        mask[i,j] = 1
        
        
    if pattern =='BGGR':
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                if i % 2 == 0:
                    if j%2 == 0:
                        mask[i,j] = fb
                    else:
                        mask[i,j] = 1
                else:
                    if j%2 == 0:
                        mask[i,j] = 1
                    else:
                        mask[i,j] = fr 
        


    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return mask