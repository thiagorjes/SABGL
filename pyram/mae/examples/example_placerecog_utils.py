import numpy as np
from pyram.mae.vgram.vgram_image import ImageProcProxy
from example_placerecog_config import params
from random import uniform

def LoadDataset(filename, imagepath):
    file_list = np.genfromtxt(filename, delimiter=',', names=True, dtype=np.dtype([('timestamp', object), ('x', float), ('y', float), ('label', int)]))

    input_images = ImageProcProxy.emptyDataset(file_list.shape[0], 
                                               params['input']['width'] * 2, 
                                               params['input']['height'])
    for sample in xrange(file_list.shape[0]):
        image_file          = ImageProcProxy.readImageColor(imagepath + file_list['timestamp'][sample] + '.bb08.l.png')
        image_crop          = ImageProcProxy.cropImage(image_file, 0, 0, 
                                                       params['input']['width'], 
                                                       params['input']['height'])
        image_gaus          = ImageProcProxy.applyGaussian(image_crop, 
                                    params['filter']['gaussian_radius'], 
                                    params['filter']['gaussian_sigma'])
        image_crop_int      = ImageProcProxy.convertBGR2INT(image_crop)
        image_gaus_int      = ImageProcProxy.convertBGR2INT(image_gaus)
        image_crop_vec      = ImageProcProxy.flattenImage(image_crop_int)
        image_gaus_vec      = ImageProcProxy.flattenImage(image_gaus_int)
        input_images[sample]= ImageProcProxy.concatImages(image_crop_vec, image_gaus_vec)
    return input_images, file_list['label'], file_list['x'], file_list['y']

def LoadDatasetAndCropImages(filename, imagepath):
    file_list = np.genfromtxt(filename, delimiter=',', names=True, dtype=np.dtype([('timestamp', object), ('x', float), ('y', float), ('label', int)]))

    input_images = np.zeros((file_list.shape[0], params['input']['height'], params['input']['width'], 3), dtype=np.uint8)
    for sample in xrange(file_list.shape[0]):
        image_file          = ImageProcProxy.readImageColor(imagepath + file_list['timestamp'][sample] + '.bb08.l.png')
        image_crop          = ImageProcProxy.cropImage(image_file, 0, 0, 
                                                       params['input']['width'], 
                                                       params['input']['height'])
        input_images[sample]= image_crop
        
    return input_images, file_list['label']

def ProcessImage(image_crop, pixel_range=0):
    x = uniform(-pixel_range, pixel_range)
    y = uniform(-pixel_range, pixel_range)
    image_trans         = ImageProcProxy.translateImage(image_crop, x, y)
    #ImageProcProxy.showImageBGR(image_trans)
    image_gaus          = ImageProcProxy.applyGaussian(image_trans, 
                            params['filter']['gaussian_radius'], 
                            params['filter']['gaussian_sigma'])
    image_crop_int      = ImageProcProxy.convertBGR2INT(image_crop)
    image_gaus_int      = ImageProcProxy.convertBGR2INT(image_gaus)
    image_crop_vec      = ImageProcProxy.flattenImage(image_crop_int)
    image_gaus_vec      = ImageProcProxy.flattenImage(image_gaus_int)
    input_image         = ImageProcProxy.concatImages(image_crop_vec, image_gaus_vec)
    
    return input_image

def TruePositive(sample, predicted, expected, max_number_of_frames=5):
    samples = expected.shape[0]
    for sample_index in xrange(max(0, sample-max_number_of_frames), min(samples, sample+max_number_of_frames+1)):
        if ( predicted[sample] == expected[sample_index] and predicted[sample] != -1
             or (predicted[sample] == -1 and expected[sample] == -1)):
            return True
    return False

def EvaluateOutput(predicted, expected, max_number_of_frames=5):
    hit = 0.0
    samples = expected.shape[0]
    for sample in xrange(samples):
        if TruePositive(sample, predicted, expected, max_number_of_frames):
            hit = hit + 1.0
    return hit / float(samples)

def ClearFalseNegatives(train_label, test_label, test_data):
    ItemsToClear = np.setdiff1d(test_label, train_label)
    ItemsToClearMask = np.in1d(test_label.ravel(), ItemsToClear).reshape(test_label.shape)
    #test_label[ItemsToClearMask] = -1
    return test_data[~ItemsToClearMask,...], test_label[~ItemsToClearMask,...]

def ClearLowConfidence(predicted, confidence, threshold=.5):
    predicted[confidence < threshold] = -1
    
