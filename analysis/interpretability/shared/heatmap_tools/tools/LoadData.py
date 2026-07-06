import numpy as np
import torch
import h5py
import os

def load_pth(load_dir,device='cpu'):
    """
    Load data from a .pth file
    """
    return torch.load(load_dir, map_location=device)

def load_npy(load_dir):
    """
    Load data from a .npy file
    """
    return np.load(load_dir)

#========== load_npz: interactively load arrays from an .npz file ==========
def load_npz(load_dir):

    """
    Interactively load arrays from an .npz file
    """
    # load the npz file
    npz_file = np.load(load_dir)
    array_names = list(npz_file.files)
    
    # if there is only one array, load it directly
    if len(array_names) == 1:
        name = array_names[0]
        result = npz_file[name]
        return result
    
    # multiple arrays: prompt the user to choose
    print(f"Found {len(array_names)} arrays:")
    for i, name in enumerate(array_names):
        shape = npz_file[name].shape
        print(f"  {i+1}. {name} - shape: {shape}")

    print(f"  0. Load all")
    while True:
        try:
            choice = input(f"Select a number: ")
            choice = int(choice)
            
            if choice == 0:
                # load all arrays
                result = {name: npz_file[name] for name in array_names}
                print(f"All arrays loaded")
                return result
                
            elif 1 <= choice <= len(array_names):
                # load a single array
                selected_name = array_names[choice-1]
                result = npz_file[selected_name]
                print(f"Loaded array '{selected_name}', shape: {result.shape}")
                return result
                
            else:
                print("Invalid choice, please try again")
                
        except ValueError:
            print("Please enter a number")
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return None
#========== load_h5: load all data from an H5 file ==========
def load_h5(file_path):
    """
    H5 file data is a nested dict; load all data

    Args:
        file_path (str): path to the H5 file

    Returns:
        dict: data dictionary
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        data = {key: {subkey: f[key][subkey][()] for subkey in f[key].keys()} 
                    if isinstance(f[key], h5py.Group) 
                    else f[key][()] for key in f.keys()}
    return data


def resolve_h5_path(root_dir, sample_id, h5_name="HE.h5"):
    """Resolve a sample H5 path, accepting both HE.h5 and HE_noskip.h5 layouts."""
    candidates = []
    if h5_name:
        candidates.append(os.path.join(root_dir, sample_id, h5_name))
    candidates.extend([
        os.path.join(root_dir, sample_id, "HE.h5"),
        os.path.join(root_dir, sample_id, "HE_noskip.h5"),
    ])
    for path in dict.fromkeys(candidates):
        if os.path.exists(path):
            return path
    return candidates[0]


def get_locations_and_lens(h5_data):
    """Extract tile coordinates and valid tile count from common H5 schemas."""
    coord_key = None
    for key in ("locations", "locations_5x_in_20x", "coords", "coordinates"):
        if key in h5_data:
            coord_key = key
            break
    if coord_key is None:
        raise KeyError("H5 file is missing locations/locations_5x_in_20x/coords/coordinates.")

    locations = h5_data[coord_key]
    if "lens" in h5_data:
        lens = int(np.asarray(h5_data["lens"]).reshape(-1)[0])
    else:
        lens = int(locations.shape[0])
    return locations[:lens], lens
    
def load_sample_list_from_txt(txt_path):
    '''
    Description: read a sample list from a .txt file
    Args:
        txt_path: path to the .txt file, one sample ID per line
    Returns:
        list of sample IDs
    '''
    with open(txt_path, 'r', encoding='utf-8') as f:
        sample_list = [line.strip() for line in f if line.strip()]
    return sample_list
