from pyram.mae.vgram.vgram_core import VGRAM
from pyram.mae.vgram.vgram_output import NetworkOutput
from pyram.mae.vgram.vgram_synapse import ConnectionGaussian, ConnectionRandom, ConnectionInput
from example_placerecog_utils import LoadDataset, ProcessImage, ClearFalseNegatives, EvaluateOutput
from example_placerecog_utils import LoadDatasetAndCropImages
from example_placerecog_config import params
import numpy as np

if __name__ == '__main__':
    network = VGRAM(params['output']['width'], params['output']['height'])
    
    network.connections = [
                           ConnectionRandom(input_layer=ConnectionInput(params['input']['width'], params['input']['height'], 0),
                                            synapses=params['connection_rand']['synapses']),
                           ConnectionGaussian(input_layer=ConnectionInput(params['input']['width'], params['input']['height'], 1),
                                              synapses=params['connection_gaus']['synapses'], 
                                              radius=params['connection_gaus']['radius'])
                           ]
    train_data, train_label, _, _ = LoadDataset(params['dataset']['train']['file'], 
                                              params['dataset']['train']['path'])
    network.train(train_data, train_label)
    #network.unload()
    #network.reload()
    test_data, test_label = LoadDatasetAndCropImages(params['dataset']['test']['file'], 
                                              params['dataset']['test']['path'])
    
    test_data, test_label = ClearFalseNegatives(train_label, test_label, test_data)
    
    confidence = np.zeros(test_label.shape[0], dtype=float)
    predicted = np.zeros(test_label.shape[0], dtype=int)
    for i in xrange(test_label.shape[0]):
        max_confidence = 0.0
        for j in xrange(1):
            test_image = ProcessImage(test_data[i,...], 0)
            test_image = np.reshape(test_image, (1,-1))
            output_data = network.test(test_image, test_label[i])
            pred, conf = NetworkOutput.MajorityVoteAndConfidence(output_data, 1)
            if conf >= max_confidence:
                predicted[i] = pred
                confidence[i] = conf
                max_confidence = conf
    
    #ClearLowConfidence(predicted, confidence, 0.1)
    
    result_file = open(params['result_file'], 'w')
    for frame_radius in xrange(19):
        result = EvaluateOutput(predicted, test_label.flatten(), max_number_of_frames=frame_radius)
        print>>result_file, (result * 100.0)
    
