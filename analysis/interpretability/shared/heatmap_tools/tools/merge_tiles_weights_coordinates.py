'''
    Description: single-sample heatmap data preprocessing
'''
import numpy as np
import os
import h5py

def merge_tiles_weights_coordinates(tile_path, weight_path, save_dir=None):
    """
    Merge the tiles weight/coordinate data for a single sample and return the merged data dict

    Args:
        tile_path (str): path to tiles data (a single sample's tiles data)
            - format: sample_id/HE.h5(HE_images, locations)
        weight_path (str): path to the weight data file (single-sample weights, list, each of shape [actual_len])
            - format: sample_id.npy(weights)
        save_dir (str): output directory path
    Returns:
        None

    Returned data structure:
        {
            'sample_id': {
                'coordinates': numpy.array([[x1, y1], [x2, y2], ...]),
                'scores': numpy.array([score1, score2, ...])
        }
    """
    print("=== Merge tiles weight/coordinate data for a single sample ===")
    
    # 1. load tiles data
    print("Loading tiles data...")
    tiles_data = load_one_tile_h5(tile_path)
    print(f"tiles data keys: {list(tiles_data.keys())}")
    
    # 2. load weight data
    print("Loading weight data...")
    if  os.path.exists(weight_path):
        weight_data = np.load(weight_path)
    else:
        print(f"{weight_path} does not exist")
        return
    print(f"weight data length: {len(weight_data)}")
    
    if 'locations' not in tiles_data.keys():
        print(f"  tiles data is missing the 'locations' key")
        print(f"  available keys: {list(tiles_data.keys())}")
        return
    # 3. get the sample length
    sample_len=tiles_data['locations'].shape[0]
    
    # 4. check data consistency
    if sample_len != len(weight_data):
        print(f"Warning: tiles data length({len(tiles_data)}) != weight data length({len(weight_data)})")
        return
    
    # 5. use the weight file's sample name as ID
    sample_id = os.path.basename(os.path.dirname(tile_path))
    print(f"\nMerging sample: {sample_id}")
    
    # check required data
    if 'locations' not in tiles_data.keys():
        print(f"  tiles data is missing the 'locations' key")
        print(f"  available keys: {list(tiles_data.keys())}")
        return
    
    merged_data_dict={}
    
    # 6. get coordinates and weight data
    coordinates = tiles_data['locations']
    scores = weight_data
    
    # 7. merge weights and coordinates
    merged_data_dict[sample_id] = {
        'coordinates': coordinates,
        'scores': scores
    }


    # 8. save results
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{sample_id}.h5')
        with h5py.File(save_path, 'w') as f:
            sample_group = f.create_group(sample_id)
            sample_group['coordinates'] = coordinates  
            sample_group['scores'] = scores
        print(f"  Sample merge done, saved to {save_path}")
        del tiles_data, weight_data, coordinates, scores
        return
    else:
        print(f"  Sample merge done, not saved")
        del tiles_data, weight_data, coordinates, scores
        return merged_data_dict


def load_one_tile_h5(file_path):
    """
    Load a single sample's tiles data from an H5 file

    Args:
        file_path (str): path to the H5 file

    Returns:
        dict: tiles data dictionary
    """
    tiles_data = {}
    with h5py.File(file_path, 'r') as f:
        tiles_data = {key: {subkey: f[key][subkey][()] for subkey in f[key].keys()} 
                    if isinstance(f[key], h5py.Group) 
                    else f[key][()] for key in f.keys()}
    return tiles_data


def test():
    '''
===== Example 1: single-sample multi-head attention weight reshape + merge tiles data =====

# reshape weights to actual length
weights_path = "path/to/TCGA-A6-5660-01.npz"
lens_path = "path/to/sample_len_dict.pth"
save_dir = "path/to/weights_reshape"
multi_weights_to_len(weights_path, lens_path, save_dir)

# merge tiles data
tile_path = "path/to/HE.h5"
# the reshaped weight file
weight_path = "path/to/TCGA-A6-5660-01.npy"
save_dir = "path/to/layer"
merge_tiles_weights_coordinates(tile_path, weight_path, save_dir)

===== Example 2: multi-sample multi-head attention weight reshape + merge tiles data =====
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "heatmap_tools", "tools"))
from preprocess_heatmap import multi_weights_to_len, merge_tiles_weights_coordinates
import os
from tqdm import tqdm
sample_list_path = "path/to/samples_list_568.pt"
tile_dir = "path/to/SVS"
weight_dir = "path/to/attn_npys"
lens_path = "path/to/sample_len_dict.pth"

save_dir_1 = "path/to/weights_reshape"
save_dir_2 = "path/to/merge_weights_locations"

sample_list = torch.load(sample_list_path, map_location='cpu')

# multi-sample processing
for i, sample_id in enumerate(tqdm(sample_list, desc="Processing samples")):
    if 1:
        if os.path.exists(os.path.join(save_dir_1,'layer', f"{sample_id}.npy")) and os.path.exists(os.path.join(save_dir_1,'heads', f"{sample_id}.npy")):
            print(f"{sample_id}.npy already exists")
            continue
        weight_path_1 = os.path.join(weight_dir, f"{sample_id}.npz")
        multi_weights_to_len(weight_path_1, lens_path, save_dir_1)
    if 0:
        if os.path.exists(os.path.join(save_dir_2,'layer', f"{sample_id}.h5")):
            print(f"{sample_id}.h5 already exists")
            continue
        tile_path = os.path.join(tile_dir, sample_id, "HE.h5")
        weight_path_layer = os.path.join(save_dir_1, "layer",f"{sample_id}.npy")
        save_dir_layer = os.path.join(save_dir_2, "layer")
        merge_tiles_weights_coordinates(tile_path, weight_path_layer, save_dir_layer)
        if 0:
            weight_path_head = os.path.join(save_dir_1, "heads",f"{sample_id}.npy")
            save_dir_head = os.path.join(save_dir_2, "heads")
            merge_tiles_weights_coordinates(tile_path, weight_path_head, save_dir_head)
'''
    pass

