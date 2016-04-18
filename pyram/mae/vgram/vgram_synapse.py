class ConnectionInput(object):
    def __init__(self, width, height, offset=0):
        self.width = width
        self.height = height
        self.offset = offset
    
    def offset_size(self):
        return self.width * self.height

class Connection(object):
    def __init__(self, input_layer, synapses):
        self.synapses = synapses
        self.input_layer = input_layer
    
    def connect(self, neural_network, synapse_offset, input_offset):
        pass

class ConnectionRandom(Connection):
    def connect(self, neural_network, synapse_offset, input_offset):
        neural_network.CreateSynapsesRandom(self.synapses, synapse_offset, input_offset, 
                                            self.input_layer.width, self.input_layer.height, 
                                            neural_network.network_layer_width, neural_network.network_layer_height)


class ConnectionGaussian(Connection):
    def __init__(self, input_layer, synapses, radius, same_interconnection_pattern=False):
        Connection.__init__(self, input_layer, synapses)
        self.same_interconnection_pattern = same_interconnection_pattern
        self.radius = radius
    
    def connect(self, neural_network, synapse_offset, input_offset):
        neural_network.CreateSynapsesGaussian(self.synapses, synapse_offset, input_offset, 
                                              self.input_layer.width, self.input_layer.height, 
                                              neural_network.network_layer_width, neural_network.network_layer_height, 
                                              self.radius, self.same_interconnection_pattern)

class ConnectionLogPolar(ConnectionGaussian):
    def __init__(self, input_layer, synapses, radius, factor, same_interconnection_pattern=False):
        ConnectionGaussian.__init__(self, input_layer, synapses, radius, same_interconnection_pattern)
        self.factor = factor
    
    def connect(self, neural_network, synapse_offset, input_offset):
        neural_network.CreateSynapsesLogPolar(self.synapses, synapse_offset, input_offset, 
                                              self.input_layer.width, self.input_layer.height, 
                                              neural_network.network_layer_width, neural_network.network_layer_height, 
                                              self.radius, self.factor, self.same_interconnection_pattern)

