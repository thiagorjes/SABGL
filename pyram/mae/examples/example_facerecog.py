import cv2
import numpy as np
from example_facerecog_config import params

def loadFaceFeatures(path, filename):
    return np.fromregex(path + filename + '.txt', r'.*=(\d+) (\d+)\n', dtype=int)

def loadFaceImage(path, filename, width, height, channels):
    g = np.empty((height, width, channels), dtype=np.uint8)
    f = np.fromfile(path + filename + '.raw', dtype=np.uint8,count=height*width*channels)
    i = 0
    for y in xrange(height):
        for x in xrange(width):
            g[y,x,0] = f[i + height * width * 2]
            g[y,x,1] = f[i + height * width]
            g[y,x,2] = f[i]
            i = i + 1
    return g

def loadDataset(path, filename):
    file_list = np.genfromtxt(path + filename, dtype=(int, 'S1', int, int))
    file_name = '%c-%03d-%d' % (file_list[0][1], file_list[0][2], file_list[0][3])
    return loadFaceImage(path, file_name, params['input']['width'], params['input']['height'], params['input']['channels'])


im0 = loadDataset(params['dataset']['path'], params['dataset']['train'][0])

cv2.imshow("", im0)
cv2.waitKey()
cv2.destroyAllWindows()