import numpy as np
import ctypes
import pickle
import os.path
from ctypes_utils import ctypes2numpy, array_1d_int
from ctypes_utils import numpy2ctypes

basepath = os.path.abspath(os.path.dirname(__file__))
try:
    #to debug uncomment following line
    #libvgram = ctypes.cdll.LoadLibrary(os.path.join(basepath, "/Users/avelino/Sources/deepslam/pyram_build/src/vgram.so"))
    libvgram = ctypes.cdll.LoadLibrary(os.path.join(basepath, "vgram.so"))
except:
    raise OSError,"Could not load VGRAM dynamic library"

class VG_RAM_WNN(ctypes.Structure):
    _fields_ = [
                ("network_layer_width", ctypes.c_int),
                ("network_layer_height", ctypes.c_int),
                ("number_of_neurons", ctypes.c_int),
                ("number_of_synapses_per_neuron",ctypes.c_int),
                ("input_size",ctypes.c_int),
                ("memory_bit_group_size",ctypes.c_int),
                ("memory_size",ctypes.c_int),
                ("test_size",ctypes.c_int),
                ("network_output",ctypes.POINTER(ctypes.c_int)),
                ("synapses",ctypes.POINTER(ctypes.c_int)),
                ("memories",ctypes.POINTER(ctypes.c_int)),
                ("d_network_output",ctypes.POINTER(ctypes.c_int)),
                ("d_synapses",ctypes.POINTER(ctypes.c_int)),
                ("d_memories",ctypes.POINTER(ctypes.c_int))
                ]
    
    def AllocateNetworkMemories(self):
        cAllocateNetworkMemories(ctypes.byref(self))
        
    def DeallocateNetworkMemories(self):
        cDeallocateNetworkMemories(ctypes.byref(self))
        
    def AllocateNetworkSynapses(self):
        cAllocateNetworkSynapses(ctypes.byref(self))
    
    def DeallocateNetworkSynapses(self):
        cDeallocateNetworkSynapses(ctypes.byref(self))
    
    def AllocateNetworkOutput(self, size, value=0):
        cAllocateNetworkOutput(ctypes.byref(self), size, value)
    
    def DeallocateNetworkOutput(self):
        cDeallocateNetworkOutput(ctypes.byref(self))
    
    def InitializeNetwork(self):
        cInitializeNetwork(ctypes.byref(self))
    
    def CreateSynapsesRandom(self, synapses, offset, input_offset, input_width, input_height, output_width, output_height):
        cCreateSynapsesRandom(ctypes.byref(self), synapses, offset, input_offset, input_width, input_height, output_width, output_height)
    
    def CreateSynapsesLogPolar(self, synapses, offset, input_offset, input_width, input_height, output_width, output_height,
                               gaussian_radius, log_factor, same_interconnection_pattern):
        cCreateSynapsesLogPolar(ctypes.byref(self), synapses, offset, input_offset, input_width, input_height, output_width, output_height,
                               gaussian_radius, log_factor, same_interconnection_pattern)
    
    def CreateSynapsesGaussian(self, synapses, offset, input_offset, input_width, input_height, output_width, output_height,
                               gaussian_radius, same_interconnection_pattern):
        cCreateSynapsesGaussian(ctypes.byref(self), synapses, offset, input_offset, input_width, input_height, output_width, output_height,
                               gaussian_radius, same_interconnection_pattern)
    
    def Train(self, input_image, input_class, sample):
        inputDataTemp = np.ascontiguousarray(input_image, dtype=np.int32)
        cTrain(ctypes.byref(self), inputDataTemp, input_class, sample)        
    
    def Unload(self, filename):
        network_memory = ctypes2numpy(self.memories, self.number_of_neurons * self.memory_size * (self.memory_bit_group_size + 1) )
        
        network_metadata = {
                            'network_layer_width': self.network_layer_width,
                            'network_layer_height': self.network_layer_height,
                            'number_of_neurons': self.number_of_neurons,
                            'number_of_synapses_per_neuron': self.number_of_synapses_per_neuron,
                            'input_size': self.input_size, 
                            'memory_bit_group_size': self.memory_bit_group_size,
                            'memory_size': self.memory_size, 
                            'memory': network_memory
                            }
        
        network_file = open(filename, 'wb')
        pickle.dump(network_metadata, network_file)
        network_file.close()
    
    def Reload(self, filename):
        network_file = open(filename, 'rb')
        network_metadata = pickle.load(network_file)
        network_file.close()
        
        self.network_layer_width = network_metadata['network_layer_width']
        self.network_layer_height = network_metadata['network_layer_height']
        self.number_of_neurons = network_metadata['number_of_neurons']
        self.number_of_synapses_per_neuron = network_metadata['number_of_synapses_per_neuron']
        self.input_size = network_metadata['input_size']
        self.memory_bit_group_size = network_metadata['memory_bit_group_size']
        self.memory_size = network_metadata['memory_size']
        network_memory = network_metadata['memory']
        
        self.DeallocateNetworkMemories() 
        self.AllocateNetworkMemories() 
        
        numpy2ctypes(self.memories, np.ascontiguousarray(network_memory, dtype=np.int32))
    
    def Test(self, input_image, input_class, sample):
        inputDataTemp = np.ascontiguousarray(input_image, dtype=np.int32)
        cTest(ctypes.byref(self), inputDataTemp, input_class, sample)  
        return ctypes2numpy(self.network_output, self.number_of_neurons * self.test_size)
    
    def TestSequence(self, testing_data, step_list):
        step_array_size = len(step_list)
        step_array = np.array(step_list)
        step_array = np.reshape(step_array, step_array.size)
        step_array = np.ascontiguousarray(step_array, dtype=np.int32)
        cTestSequence(ctypes.byref(self), ctypes.byref(testing_data), step_array, step_array_size)
        return ctypes2numpy(self.network_output, self.test_size)

class DATA_SET(ctypes.Structure):
    _fields_ = [
                ("num_samples", ctypes.c_int),
                ("num_inputs", ctypes.c_int),
                ("sample_class",ctypes.POINTER(ctypes.c_int)),
                ("sample",ctypes.POINTER(ctypes.c_int)),
                ("d_sample_class",ctypes.POINTER(ctypes.c_int)),
                ("d_sample",ctypes.POINTER(ctypes.c_int))
                ]
    
    def DeallocateDataSet(self):
        cDeallocateDataSet(ctypes.byref(self))

    def AllocateDataSet(self):
        cAllocateDataSet(ctypes.byref(self))
        
    def InitializeDataSet(self):
        cInitializeDataSet(ctypes.byref(self))

    def CopyDataSetFrom(self, input_data, class_data):
        inputDataTemp = np.reshape(input_data, input_data.size)
        inputDataTemp = np.ascontiguousarray(inputDataTemp, dtype=np.int32)
        if (class_data is None):
            cCopyDataSet(ctypes.byref(self), inputDataTemp, None)
        else:
            classDataTemp = np.reshape(class_data, class_data.size)
            classDataTemp = np.ascontiguousarray(classDataTemp, dtype=np.int32)
            cCopyDataSet(ctypes.byref(self), inputDataTemp, classDataTemp)

cInitializeNetwork                  = libvgram.InitializeNetwork
cInitializeNetwork.argtypes         = [ctypes.POINTER(VG_RAM_WNN)]

cAllocateNetworkSynapses            = libvgram.AllocateNetworkSynapses
cAllocateNetworkSynapses.argtypes   = [ctypes.POINTER(VG_RAM_WNN)]

cDeallocateNetworkSynapses          = libvgram.DeallocateNetworkSynapses
cDeallocateNetworkSynapses.argtypes = [ctypes.POINTER(VG_RAM_WNN)]

cAllocateNetworkMemories            = libvgram.AllocateNetworkMemories
cAllocateNetworkMemories.argtypes   = [ctypes.POINTER(VG_RAM_WNN)]

cDeallocateNetworkMemories          = libvgram.DeallocateNetworkMemories
cDeallocateNetworkMemories.argtypes = [ctypes.POINTER(VG_RAM_WNN)]

cAllocateNetworkOutput              = libvgram.AllocateNetworkOutput
cAllocateNetworkOutput.argtypes     = [ctypes.POINTER(VG_RAM_WNN), ctypes.c_int, ctypes.c_int]

cDeallocateNetworkOutput            = libvgram.DeallocateNetworkOutput
cDeallocateNetworkOutput.argtypes   = [ctypes.POINTER(VG_RAM_WNN)]

cInitializeDataSet                  = libvgram.InitializeDataSet
cInitializeDataSet.argtypes         = [ctypes.POINTER(DATA_SET)]

cAllocateDataSet                    = libvgram.AllocateDataSet
cAllocateDataSet.argtypes           = [ctypes.POINTER(DATA_SET)]

cDeallocateDataSet                  = libvgram.DeallocateDataSet
cDeallocateDataSet.argtypes         = [ctypes.POINTER(DATA_SET)]

cCopyDataSet                        = libvgram.CopyDataSet
cCopyDataSet.argtypes               = [ctypes.POINTER(DATA_SET), array_1d_int, array_1d_int]

cCreateSynapsesRandom               = libvgram.CreateSynapsesRandom
cCreateSynapsesRandom.argtypes      = [ctypes.POINTER(VG_RAM_WNN), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

cCreateSynapsesLogPolar             = libvgram.CreateSynapsesLogPolar
cCreateSynapsesLogPolar.argtypes    = [ctypes.POINTER(VG_RAM_WNN), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                       ctypes.c_double, ctypes.c_double, ctypes.c_bool]

cCreateSynapsesGaussian             = libvgram.CreateSynapsesGaussian
cCreateSynapsesGaussian.argtypes    = [ctypes.POINTER(VG_RAM_WNN), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                       ctypes.c_double, ctypes.c_bool]

cTrain                              = libvgram.Train
cTrain.argtypes                     = [ctypes.POINTER(VG_RAM_WNN), array_1d_int, ctypes.c_int, ctypes.c_int]

cTest                               = libvgram.Test
cTest.argtypes                      = [ctypes.POINTER(VG_RAM_WNN), array_1d_int, ctypes.c_int, ctypes.c_int]

cTestSequence                       = libvgram.TestSequence
cTestSequence.argtypes              = [ctypes.POINTER(VG_RAM_WNN), ctypes.POINTER(DATA_SET), array_1d_int, ctypes.c_int]

