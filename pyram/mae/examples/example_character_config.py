
params = {
    'input': 
    {
        'width' :   64,
        'height':   64,
    },
    'filter':
    {
        'gaussian_radius' : 5,
        'gaussian_sigma'  : 1.0
     },
    'connection':
    {
        'synapses'  :128, 
        'radius'    :10.0
    },
    'output':
    {
        'width' :   32,
        'height':   32,
     },
    'dataset':
    {
        'train':
        {
            'total' : 60000,
            'images': '/dados/MNIST/train-images-idx3-ubyte',
            'labels': '/dados/MNIST/train-labels-idx1-ubyte'
         },
        'test':
        {
            'total' : 10000,
            'images': '/dados/MNIST/t10k-images-idx3-ubyte',
            'labels': '/dados/MNIST/t10k-labels-idx1-ubyte'
         }
     }
}