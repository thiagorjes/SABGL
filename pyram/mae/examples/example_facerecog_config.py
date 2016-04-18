params = {
    'input': 
    {
        'width' :   768,
        'height':   576,
        'channels': 3
    },
    'reshape':
    {
        'width' :   128,
        'height':   200,
        'offset_x': 0.0,
        'offset_y': 15.0,
        'baseline_factor': 0.45,
        'gaussian_radius': 3,
        'gaussian_sigma' : 3.0
    },
    'filter':
    {
    'gaussian_radius' : 5,
    'gaussian_sigma'  : 1.0
    },
    'connection_gaus':
    {
        'synapses'  : 128, 
        'radius'    : 10.0,
    },
    'output':
    {
        'width' :   32,
        'height':   32,
     },
    'dataset':
    {
        'path' : '/dados/faces/ARPhotoDataBase/',
        'train': ['random_faces_t.txt'],
        'test' : [
                  'faces_te_all_side_lights.txt',
                  'faces_te_anger.txt',
                  'faces_te_glasses.txt',
                  'faces_te_left_light.txt',
                  'faces_te_right_light.txt',
                  'faces_te_scarf.txt',
                  'faces_te_scream.txt',
                  'faces_te_smile.txt'
                  ]
     }
}