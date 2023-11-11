import os
from PIL import Image
from numpy import array, tile, zeros, ones, empty
from numpy import concatenate as c_
from numpy.random import uniform
from numpy.linalg import norm
from scipy import spatial
import numpy as np
from numpy.random import randint



def find(name, path):
    '''
    Arguments: 
    name: Name of file.
    path: the path of a folder to search it from
    Return:
    path of the first matching file in the folder.
    If there is none matching file, it return empty.
    '''
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return(None)

def getPixel(image):
    '''
    Arguments:
    image: an image.
    Return:
    A 2-dimensional numpy array A[x*y,5] detailing:
    x*y: Image pixel count
    z: The rgb value of the image,plus coordinates
    '''
    
    h,w = image.size
    totalsize = w*h
    pix = array(image.getdata())
    k,t =pix.shape
    print(k,t)
    #Create width matrix:
    wm = array(range(w))
    wm = tile(wm,(h,1))
    wm = wm.flatten()
    wm = array(wm)[np.newaxis]
    wm = wm.T
    #print(wm)
    hm = array(range(h))[np.newaxis]
    hm = tile(hm.T, (1,w))
    hm = hm.flatten()
    hm = array(hm)[np.newaxis]
    hm = hm.T
    #print(hm)
    result = c_((wm,hm,pix), axis=1)
    return result



def arrtoPixel(Y,w,h):
    #print(Y)
    tada = Y[:,2:]
    canvas = tada.reshape(w,h,-1).astype(np.uint8)
    img = Image.fromarray(canvas)
    return img

        


class ThatsNotImage(Exception):
    pass



class MyKMean:
    ''' 
    This class only used in this specific case - image segmentation. Its output can't be used anywhere else
    Parameters:
    K = K count.
    Thresold: a number between 0 and 1, default 0.01, control accuracy
    maxIter: How long will we run if there is no cinvergence?
    '''
    def __init__(self, K, thresold= 0.01, maxIter=300):
        self.K = K
        self.theta = None
        self.thresold = thresold
        self.maxIter = maxIter



    def save_Y(self, X, minid):
        n,d = X.shape
        Y = X
        #len = []
        self.theta = np.floor(self.theta)
        
        for i in range(n):
            tu = minid[i]
            # Replace rgb values only
            r = self.theta[tu,2:]
            Y[i,2:] = r
        #print(max(len))
        return Y

    def fit(self, XAI):
        
        # init theta:
        n,d = XAI.shape
        X = XAI
        self.theta = empty((self.K, d))
        #init Y
        
        # random value of theta:
        for i in range (d):
            xd = X[:,i]
            maxrng = np.amax(xd)
            minrng = np.amin(xd)
            werng = uniform(low=minrng,high=maxrng,size=self.K)
            self.theta[:,i] = werng
        
        #converge = True
        count = 0
        while True:
            tree = spatial.KDTree(self.theta,copy_data=True)
            mindist, minid = tree.query(X)
            
            unique, counts = np.unique(minid, return_counts=True)
            for i in range(self.K):
                if (i in unique) == False:
                    nid = randint(0,n-1)
                    minid[nid]=i



            #Calculate distances
            nbcount = zeros(shape=(self.K,d))
            nbrows = zeros(shape =(self.K,d))
            for i in range (n):
                tu = minid[i]
                temp = nbcount[tu,:]
                temp = temp + X[i,:]
                trow = nbrows[tu,:]
                trow = trow + 1
                nbcount[tu,:] = temp
                nbrows[tu,:] = trow
            newtheta = nbcount/nbrows
            count = count+1
            #Now we check convergence
            if count>self.maxIter:
                break
               
                
            err = self.theta-newtheta
            len = 0
            for i in range(self.K):
                vt = norm(err[i,:])/norm(self.theta[i,:])
                len = len+vt
            if len<self.thresold:
                break #
        Y = MyKMean.save_Y(self,X=X, minid=minid)
        return Y

        
    
    
    