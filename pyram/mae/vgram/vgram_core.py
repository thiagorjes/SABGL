import numpy as np
from vgram_wrapper import VG_RAM_WNN, DATA_SET
from datetime import datetime
from pyram.mae.vgram.vgram_image import ImageProcProxy
from pyram.mae.examples.example_placerecog_config import params
import cv2
from random import uniform

class VGRAM(object):
    connectionList = []
    
    def __init__(self, width, height):
        self.neural_network = VG_RAM_WNN(width, height)
        self.neural_network.InitializeNetwork()

    @property
    def connections(self):
        return self.connectionList
    
    @connections.setter
    def connections(self, connections):
        self.connectionList = connections
        
        self.neural_network.number_of_synapses_per_neuron = sum(item.synapses for item in connections)
        
        self.neural_network.AllocateNetworkSynapses() 
        
        input_offset = 0
        synapse_offset = 0
        for connection in connections:
            connection.connect(self.neural_network, synapse_offset, input_offset * connection.input_layer.offset)
            input_offset = input_offset + connection.input_layer.offset_size()
            synapse_offset = synapse_offset + connection.synapses

    def train(self, memory_size, input_size, num_samples, filename,stereo_mode):
        self.neural_network.memory_size = memory_size   
        self.neural_network.input_size = input_size        
        self.neural_network.AllocateNetworkMemories() 
        file_list = np.genfromtxt(filename, delimiter=',', names=True, dtype=np.dtype([('image',object), 
                                                                                            ('label', int), 
                                                                                            ('x', float), 
                                                                                            ('y', float), 
                                                                                            ('z',float),
                                                                                            ('rx', float), 
                                                                                            ('ry', float), 
                                                                                            ('rz',float), 
                                                                                            ('timestamp', object)]))

        for sample in xrange(file_list.shape[0]):
            print "iterate:",sample
            if not stereo_mode :
                orig_image_file     = ImageProcProxy.readImageColor(file_list['image'][sample])
                image_file          = cv2.resize(orig_image_file, (params['input']['width'],params['input']['height']))
                image_crop          = image_file 
            else:
                image_file          = ImageProcProxy.readImageColor(file_list['image'][sample])
                image_crop          = ImageProcProxy.cropImage(image_file, 0, 0, params['input']['width'], params['input']['height'])
            image_gaus          = ImageProcProxy.applyGaussian(image_crop, 
                                        params['filter']['gaussian_radius'], 
                                        params['filter']['gaussian_sigma'])
            image_crop_int      = ImageProcProxy.convertBGR2INT(image_crop)
            image_gaus_int      = ImageProcProxy.convertBGR2INT(image_gaus)
            image_crop_vec      = ImageProcProxy.flattenImage(image_crop_int)
            image_gaus_vec      = ImageProcProxy.flattenImage(image_gaus_int)
            input_image = ImageProcProxy.concatImages(image_crop_vec, image_gaus_vec)
            input_class = file_list['label'][sample]
            self.neural_network.Train( input_image, input_class, sample)
        

    def test(self,  memory_size, input_size, num_samples, filename,stereo_mode):
        self.neural_network.test_size = memory_size
        self.neural_network.input_size = input_size
        self.neural_network.AllocateNetworkOutput(self.neural_network.test_size * self.neural_network.number_of_neurons)
        file_list = np.genfromtxt(filename, delimiter=',', names=True, dtype=np.dtype([('image',object), 
                                                                                            ('label', int), 
                                                                                            ('x', float), 
                                                                                            ('y', float), 
                                                                                            ('z',float),
                                                                                            ('rx', float), 
                                                                                            ('ry', float), 
                                                                                            ('rz',float), 
                                                                                            ('timestamp', object)]))

        for sample in xrange(file_list.shape[0]):
            print "iterate:",sample
            if not stereo_mode :
                orig_image_file     = ImageProcProxy.readImageColor(file_list['image'][sample])
                image_file          = cv2.resize(orig_image_file, (params['input']['width'],params['input']['height']))
                image_crop          = image_file 
            else:
                image_file          = ImageProcProxy.readImageColor(file_list['image'][sample])
                image_crop          = ImageProcProxy.cropImage(image_file, 0, 0, params['input']['width'], params['input']['height'])
            image_gaus          = ImageProcProxy.applyGaussian(image_crop, 
                                        params['filter']['gaussian_radius'], 
                                        params['filter']['gaussian_sigma'])
            image_crop_int      = ImageProcProxy.convertBGR2INT(image_crop)
            image_gaus_int      = ImageProcProxy.convertBGR2INT(image_gaus)
            image_crop_vec      = ImageProcProxy.flattenImage(image_crop_int)
            image_gaus_vec      = ImageProcProxy.flattenImage(image_gaus_int)
            input_image = ImageProcProxy.concatImages(image_crop_vec, image_gaus_vec)
            input_class = file_list['label'][sample]
            output = self.neural_network.Test(input_image, input_class, sample)

        self.neural_network.DeallocateNetworkOutput()
        return output, file_list['label']
    
    def testSequence(self, input_data, step_list):
        self.neural_network.test_size = input_data.shape[0] 
        self.neural_network.input_size = input_data.shape[1]
        
        self.testing_data = DATA_SET(self.neural_network.test_size, self.neural_network.input_size)
        self.testing_data.InitializeDataSet()
        self.testing_data.AllocateDataSet()
        self.testing_data.CopyDataSetFrom(input_data, None)
        
        self.neural_network.AllocateNetworkOutput(self.neural_network.test_size, -1)
        
        output = self.neural_network.TestSequence(self.testing_data, step_list)
        
        self.neural_network.DeallocateNetworkOutput()
        self.testing_data.DeallocateDataSet()
        del(self.testing_data)
        
        return output
    
    def unload(self, filename='memory.pickle'):
        self.neural_network.Unload(filename)
    
    def reload(self, filename='memory.pickle'):
        self.neural_network.Reload(filename)
    
    def __del__(self):
        self.neural_network.DeallocateNetworkSynapses() 
        self.neural_network.DeallocateNetworkMemories() 
        self.neural_network.DeallocateNetworkOutput()
        if 'training_data' in self.__dict__:
            self.training_data.DeallocateDataSet()
        if 'testing_data' in self.__dict__:
            self.testing_data.DeallocateDataSet()
