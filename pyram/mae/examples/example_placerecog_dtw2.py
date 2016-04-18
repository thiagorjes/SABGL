from pyram.mae.vgram.vgram_core import VGRAM
from pyram.mae.vgram.vgram_synapse import ConnectionGaussian, ConnectionRandom, ConnectionInput
from example_placerecog_config import params
from example_placerecog_utils import LoadDataset, ClearFalseNegatives, EvaluateOutput
import numpy as np

def step_list(step_size):
    return [(step_size,1), (1,step_size), (1,1)]

def run_ensemble2(train_filename, train_path, test_filename, test_path, 
                  result_filename, output_filename, coords_filename,
                  batch_size, step_size):
    
    network = VGRAM(params['output']['width'], params['output']['height'])
    
    network.connections = [
                           ConnectionRandom(input_layer=ConnectionInput(params['input']['width'], params['input']['height'], 0),
                                            synapses=params['connection_rand']['synapses']),
                           ConnectionGaussian(input_layer=ConnectionInput(params['input']['width'], params['input']['height'], 1),
                                              synapses=params['connection_gaus']['synapses'], 
                                              radius=params['connection_gaus']['radius'])
                           ]
    train_data, train_label, _, _ = LoadDataset(train_filename, train_path)
    
    network.train(train_data, train_label)
    
    test_data, test_label, x, y = LoadDataset(test_filename, test_path)
    
    test_data, test_label = ClearFalseNegatives(train_label, test_label, test_data)
    
    print 'Train size:', train_label.shape[0]
    print 'Test size:', test_label.shape[0]
    
    test_size = test_label.shape[0]
    pred_label = -1 * np.ones((test_size,), dtype=int)
    for i in xrange((batch_size-1), test_size):
        test_data_i = test_data[i-(batch_size-1):i+1,:]
        predicted = network.testSequence(test_data_i, step_list(step_size))
        pred_label[i] = predicted[batch_size-1]
        print pred_label[i], test_label[i], predicted
    pred_label[0:batch_size] = test_label[0:batch_size] #disregard first results
    
    result_file = open(result_filename, 'w')
    for frame_radius in xrange(19):
        result = EvaluateOutput(pred_label, test_label.flatten(), max_number_of_frames=frame_radius)
        print>>result_file, (result * 100.0)
    
    np.savetxt(output_filename, [pred_label, test_label])
    np.savetxt(coords_filename, [x, y])

if __name__ == '__main__':
    run_ensemble2(params['dataset']['train']['file'],
                  params['dataset']['train']['path'],
                  params['dataset']['test']['file'],
                  params['dataset']['test']['path'],
                  params['result_file'],
                  params['output_file'],
                  params['coords_file'],
                  batch_size=5,
                  step_size=0)