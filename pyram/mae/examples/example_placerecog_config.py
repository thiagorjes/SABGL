params = {
    'input': 
    {
        'width' :   640,
        'height':   364
    },
    'filter':
    {
    'gaussian_radius' : 33,
    'gaussian_sigma'  : 11,
    },
    'connection_gaus':
    {
        'synapses'  : 64, 
        'radius'    : 30.0,
    },
    'connection_rand': 
    {
        'synapses' : 64
    },
    'output':
    {
        'width' :    96,
        'height':    54,
     },
    'dataset':
    {
        'train':
        {
            'path': '/dados/UFES/GPS/2012/',
            'file': '/dados/UFES/GPS/UFES-2012-30-train.csv'
         },
        'test':
        {
            'path': '/dados/UFES/GPS/2014/',
            'file': '/dados/UFES/GPS/UFES-2014-30-test.csv'
         }
     },
    'output_file': 'output.txt',
    'result_file': 'result.txt',
    'coords_file': 'coords.txt'
}
