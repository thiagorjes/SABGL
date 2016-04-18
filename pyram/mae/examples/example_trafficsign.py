import numpy as np
from pyram.mae.vgram.vgram_core import VGRAM
from pyram.mae.vgram.vgram_output import NetworkOutput
from pyram.mae.vgram.vgram_image import ImageProcProxy
from pyram.mae.vgram.vgram_synapse import ConnectionLogPolar, ConnectionInput
from example_trafficsign_config import params

def LoadDataset(filename, imagepath):
    file_list = np.genfromtxt(filename, dtype=np.dtype([('Filename','S21'), ('Width', int), ('Height', int), ('roi_x1', int), ('roi_y1', int), ('roi_x2', int), ('roi_y2', int), ('classId', int)]))

    ImageProcProxy.createCLAHE(clipLimit=1.4, tileGridSize=(2,2))
    input_images = ImageProcProxy.emptyDataset(file_list.shape[0], params['input']['width'], params['input']['height'])
    for sample in xrange(file_list.shape[0]):
        image_file          = ImageProcProxy.readImageColor(imagepath + file_list['Filename'][sample])
        image_roi           = ImageProcProxy.cropImage(image_file, 
                                                       file_list['roi_x1'][sample], file_list['roi_y1'][sample], 
                                                       file_list['roi_x2'][sample], file_list['roi_y2'][sample])
        image_clahe         = ImageProcProxy.emptyLike(image_roi)
        image_clahe[:,:,0]  = ImageProcProxy.applyCLAHE(image_roi[:,:,0])
        image_clahe[:,:,1]  = ImageProcProxy.applyCLAHE(image_roi[:,:,1])
        image_clahe[:,:,2]  = ImageProcProxy.applyCLAHE(image_roi[:,:,2])
        image_resized       = ImageProcProxy.resizeInterLinear(image_clahe, 
                                    params['input']['width'], 
                                    params['input']['height'])
        image_gaussian      = ImageProcProxy.applyGaussian(image_resized, 
                                    params['filter']['gaussian_radius'], 
                                    params['filter']['gaussian_sigma'])
        image_int           = ImageProcProxy.convertBGR2INT(image_gaussian)
        image_vector        = ImageProcProxy.flattenImage(image_int)
        input_images[sample]= image_vector
        
    return input_images, file_list['classId']

if __name__ == '__main__':
    network = VGRAM(params['output']['width'], params['output']['height'])
    
    network.connections = [ConnectionLogPolar(input_layer=ConnectionInput(params['input']['width'], params['input']['height']),
                                              synapses=params['connection']['synapses'], 
                                              radius=params['connection']['radius'], 
                                              factor=params['connection']['factor'])]

    input_data, class_data = LoadDataset(params['dataset']['train']['file'], params['dataset']['train']['path'])
    network.train(input_data, class_data)

    input_data, class_data = LoadDataset(params['dataset']['test']['file'], params['dataset']['train']['path'])
    output_data = network.test(input_data, class_data)
    output_value= NetworkOutput.MajorityVoteMean(output_data, class_data.flatten()) * 100.0
    print "Percentage correct =", output_value, "%"
