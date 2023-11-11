import sys
from numpy import array, loadtxt
import numpy as np
from PIL import Image
import os

from MyRandonFunctions import find,getPixel,MyKMean,arrtoPixel


if __name__=='__main__':
    argument = sys.argv
    #argument = [1,2,'cat2.png','cat2seg.png']
    ## From description:
    # arg0 is the file name.
    # arg1 is K
    # arg2 is input file.
    #arg3 is output file
    K = argument[1]
    #Check for troll. Can't knows if troll exist.
    try:
        int(K)
    except:
        print(K," is not an integer")
    K = int(K)
    path = ''
    inputFile = argument[2]
    outputFile = argument[3]
    #truePath = find(inputFile,path)
    #Troll is everywhere.
    try:
        open(inputFile, mode='r')
    except:
        print(inputFile,' cannot be opened')
    
    image = Image.open(inputFile)
    w,h = image.size
    pix = getPixel(image)
    Kmean = MyKMean(K=K)
    Y = Kmean.fit(pix)
    outImage = arrtoPixel(Y,h,w)

    outPath = os.path.join(path,outputFile)
    outImage.save(outPath)
    outImage.show()


    

    


