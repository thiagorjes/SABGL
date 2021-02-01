from pyram.mae.vgram.vgram_core import VGRAM
from pyram.mae.vgram.vgram_output import NetworkOutput
from pyram.mae.vgram.vgram_synapse import ConnectionGaussian, ConnectionRandom, ConnectionInput
from example_placerecog_utils import LoadDataset, ClearFalseNegatives, EvaluateOutput
from example_placerecog_config import params
import numpy as np

def run_vgram(train_filename, train_path, test_filename, test_path, 
              result_filename, output_filename, coords_filename,
              batch_size):
    #print "entrou"
    network = VGRAM(params['output']['width'], params['output']['height'])
    #print "carregou parametros"
    network.connections = [
                           ConnectionRandom(input_layer=ConnectionInput(params['input']['width'], params['input']['height'], 0),
                                            synapses=params['connection_rand']['synapses']),
                           ConnectionGaussian(input_layer=ConnectionInput(params['input']['width'], params['input']['height'], 1),
                                              synapses=params['connection_gaus']['synapses'], 
                                              radius=params['connection_gaus']['radius'])
                           ]

    memory_size, input_size, num_samples = LoadDataset(train_filename, train_path)
    print 'Train size:', memory_size, input_size, num_samples, train_filename  

    network.train(memory_size, input_size, num_samples, train_filename,True)

    memory_size, input_size, num_samples  = LoadDataset(test_filename, test_path)
    print 'Test size:', memory_size, input_size, num_samples, test_filename  
    
    output_data, test_label = network.test(memory_size, input_size, num_samples, test_filename,True)
    
    
    '''the output computed below takes the closest of top 3 most voted, which is closest to the previous'''
    #pred_label = NetworkOutput.MajorityVoteClosestToPrevious(output_data, test_label.shape[0])
    #pred_label = NetworkOutput.MajorityVoteClosestToPrevious2(output_data, test_label.shape[0], test_label[0])
    '''the output computed below takes the closest of top 3 most voted, which is closest to ground truth'''
    #pred_label = NetworkOutput.MajorityVoteClosestToExpected(output_data, test_label)
    '''the output computed below takes the most voted and doesn't take into consideration the confidence'''
    pred_label, confidence = NetworkOutput.MajorityVoteAndConfidence(output_data, num_samples)
    #ClearLowConfidence(pred_label, confidence, 0.1)
    #pred_label[0:batch_size] = test_label[0:batch_size] #disregard first results
    
    result_file = open(result_filename, 'w')
    for frame_radius in xrange(19):
        result = EvaluateOutput(pred_label, test_label.flatten(), max_number_of_frames=frame_radius)
        print>>result_file, (result * 100.0)
    
    np.savetxt(output_filename, [pred_label, test_label])
    np.savetxt(coords_filename, [x, y])

if __name__ == '__main__':
    run_vgram(params['dataset']['train']['file'],
              params['dataset']['train']['path'],
              params['dataset']['test']['file'],
              params['dataset']['test']['path'],
              params['result_file'],
              params['output_file'],
              params['coords_file'],
              batch_size=5)
