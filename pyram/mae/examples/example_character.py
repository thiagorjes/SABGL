import numpy as np
from example_character_config import params
from pyram.mae.vgram.vgram_synapse import ConnectionGaussian, ConnectionInput
from pyram.mae.vgram.vgram_output import NetworkOutput
from pyram.mae.vgram.vgram_image import ImageProcProxy
from pyram.mae.vgram.vgram_core import VGRAM

def LoadDataset(imagefile, labelfile, finalsize):
    loaded = np.fromfile(file=open(imagefile),dtype=np.uint8)
    images = loaded[16:].reshape((-1,28,28))
    
    loaded = np.fromfile(file=open(labelfile),dtype=np.uint8)
    labels = loaded[8:].reshape((images.shape[0]))
    
    input_images = ImageProcProxy.emptyDataset(images.shape[0], params['input']['width'], params['input']['height'])
    for sample in xrange(images.shape[0]):
        image_resized       = ImageProcProxy.resizeInterLinear(images[sample], params['input']['width'], params['input']['height'])
        image_gaussian      = ImageProcProxy.applyGaussian(image_resized, 
                                    params['filter']['gaussian_radius'], 
                                    params['filter']['gaussian_sigma'])
        image_vector        = ImageProcProxy.flattenImage(image_gaussian)
        input_images[sample]= image_vector
        
    return input_images[:finalsize], labels[:finalsize]


if __name__ == '__main__':
    network = VGRAM(params['output']['width'], params['output']['height'])
    
    network.connections = [ConnectionGaussian(input_layer=ConnectionInput(params['input']['width'], params['input']['height']),
                                              synapses=params['connection']['synapses'], 
                                              radius=params['connection']['radius'])]

    input_data, class_data = LoadDataset(params['dataset']['train']['images'], 
                                              params['dataset']['train']['labels'],
                                              params['dataset']['train']['total'])
    network.train(input_data, class_data)

    input_data, class_data = LoadDataset(params['dataset']['test']['images'], 
                                              params['dataset']['test']['labels'],
                                              params['dataset']['test']['total'])
    output_data = network.test(input_data, class_data)
    print "Percentage correct =", NetworkOutput.MajorityVoteMean(output_data, class_data.flatten()) * 100.0, "%"
