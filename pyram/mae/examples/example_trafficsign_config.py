
params = {
    'input': 
    {
        'width' :   70,
        'height':   70
    },
    'filter':
    {
        'gaussian_radius' : 7,
        'gaussian_sigma'  : 1.0
    },
    'connection':
    {
        'synapses'  :256, 
        'radius'    :7.0, 
        'factor'    :2.0
    },
    'output':
    {
        'width' :   3*17,
        'height':   3*9,
     },
    'dataset':
    {
        'train':
        {
            'path': '/dados/GTSRB/_scripts/training_set_raw/',
            'file': '/dados/GTSRB/desc_training_set_rand_860.csv'
         },
        'test':
        {
            'path': '/dados/GTSRB/_scripts/testing_set_raw/',
            'file': '/dados/GTSRB/desc_testing_set_rand_430.csv'
         }
     }
}