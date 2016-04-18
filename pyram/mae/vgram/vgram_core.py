from vgram_wrapper import VG_RAM_WNN, DATA_SET

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

    def train(self, input_data, class_data):
        self.neural_network.memory_size = input_data.shape[0] 
        self.neural_network.input_size = input_data.shape[1]
        self.training_data = DATA_SET(self.neural_network.memory_size, self.neural_network.input_size)
        self.training_data.InitializeDataSet()
        self.training_data.AllocateDataSet()
        self.training_data.CopyDataSetFrom(input_data, class_data)

        self.neural_network.AllocateNetworkMemories() 
        self.neural_network.Train(self.training_data)

        self.training_data.DeallocateDataSet()
        del(self.training_data)

    def test(self, input_data, class_data):
        self.neural_network.test_size = input_data.shape[0] 
        self.neural_network.input_size = input_data.shape[1]
        
        self.testing_data = DATA_SET(self.neural_network.test_size, self.neural_network.input_size)
        self.testing_data.InitializeDataSet()
        self.testing_data.AllocateDataSet()
        self.testing_data.CopyDataSetFrom(input_data, class_data)
        
        self.neural_network.AllocateNetworkOutput(self.neural_network.test_size * self.neural_network.number_of_neurons)
        
        output = self.neural_network.Test(self.testing_data)
        
        self.neural_network.DeallocateNetworkOutput()
        self.testing_data.DeallocateDataSet()
        del(self.testing_data)
        
        return output
    
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
