from interpolator import *
from dataio import *

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Traditional light field')

    parser.add_argument('--t', type=str, 
    choices=['interpolation', 'undersampled', 'variable_focal_plane',
    'variable_aperture_size', 'expand_field_of_view', '1', '2', '3','4', '5'],
    help='task to perform')

    parser.add_argument('--i', type=str,
    choices=['bilinear', 'quadra-linear', 'b','q'],
    help='interpolator to perform')

    arg = parser.parse_args()
    return arg

if __name__ == '__main__':
    path = 'data/'
    data = Dataset(path)
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    arg = argparse()

    if arg.i == "b":
        arg.i = "bilinear"
    elif arg.i == "q":
        arg.i = "quadra-linear"

    if arg.t == "1":
        arg.t = "interpolation"
    elif arg.t == "2":
        arg.t = "undersampled"
    elif arg.t == "3":
        arg.t = "variable_focal_plane"
    elif arg.t == "4":
        arg.t = "variable_aperture_size"
    elif arg.t == "5":
        arg.t = "expand_field_of_view"
    
    print(arg)

    task(data, arg.t, arg.i)