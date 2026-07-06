import torch
import numpy as np
import os
import h5py

#====== Reshape single-sample multi-head self-attention weights to actual length ======#
def score_shaping_selfAttn(weights_path, lens_path, save_dir, device='cpu', layer_or_head=None):
    """
            Reshape single-sample multi-head self-attention weights (Transformer) to actual length,
            handling layer and head automatically, and save to save_dir
            Args:
                weights_path: path to the weight file
                    - format: sample_id.npz(heads, num_instances, num_instances)
                lens: actual length
                    - format: .pth(dict, key: sample_id, value: len)
                save_dir: output directory
                    - format 1: layer/sample_id.npy(actual_len)
                    - format 2: heads/sample_id.npy(heads,actual_len)
            Returns:
                None
        """
    lens=torch.load(lens_path, map_location=device)
    # if the weight file for this sample exists, load it; otherwise skip this iteration
    if  os.path.exists(weights_path):
        weights = np.load(weights_path)
    else:
        print(f"{weights_path} does not exist")
        return
    weights = weights['attn']
    
    # first average over heads
    avg_heads = weights.mean(axis=0) 
    
    # then average over queries: how much each key is attended to on average by all queries
    avg_query = avg_heads.mean(axis=0)
    
    # use the weight file name as sample_id
    sample_id=weights_path.split('/')[-1].split('.')[0]
    sample_len = lens[sample_id]
    num_instances = weights.shape[1]
    
    # take weights up to sample_len
    if sample_len <= num_instances:
        weights_layer = avg_query[:sample_len]
        weights_head = avg_heads[:, :sample_len]
    else:
        print(f"{sample_id} actual length exceeds num_instances, invalid sample")
        return
        
    if layer_or_head == 'layer':
        os.makedirs(os.path.join(save_dir, 'layer'), exist_ok=True)
        save_path_layer = os.path.join(save_dir, 'layer',f'{sample_id}.npy')
        np.save(save_path_layer, weights_layer)
        print(f"{sample_id} weights reshaped to actual length and saved to: {save_path_layer}")

    elif layer_or_head == 'head':
        os.makedirs(os.path.join(save_dir, 'heads'), exist_ok=True)
        save_path_head = os.path.join(save_dir, 'heads',f'{sample_id}.npy')
        np.save(save_path_head, weights_head)
        print(f"{sample_id} weights reshaped to actual length and saved to: {save_path_head}")

    elif layer_or_head is None:
        os.makedirs(os.path.join(save_dir, 'layer'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'heads'), exist_ok=True)
        
        save_path_layer = os.path.join(save_dir, 'layer',f'{sample_id}.npy')
        save_path_head = os.path.join(save_dir, 'heads',f'{sample_id}.npy')
        
        np.save(save_path_layer, weights_layer)
        np.save(save_path_head, weights_head)
        print(f"{sample_id} weights reshaped to actual length and saved to: {save_path_layer} and {save_path_head}")

    else:
        print(f"{sample_id} invalid layer_or_head")
        return
    
    del weights, sample_id, avg_heads, avg_query, weights_layer, weights_head
    return
#====== Reshape multi-sample TanhAttn attention weights to actual length ======#
def score_shaping_TanhAttn(weights_path, lens_path, sample_list_path, save_dir, device='cpu'):
    """ 
            Reshape multi-sample TanhAttn attention weights (AdaptableMil) to actual length, saved to save_dir
            Args:
                weights_path: path to the weight file
                    - format: raw_attns.pth(torch.Size([num_samples, num_instances, 1])
                lens: actual length
                    - format: .pth(list[num_samples])
                sample_list_path: path to the sample list file
                    - format: .pt(list[num_samples])
                save_dir: output directory
                    - format: save_dir/processed_attns.npy(dict[num_samples, actual_len])
                device: compute device
    """
    lens=torch.load(lens_path, map_location=device)
    sample_list=torch.load(sample_list_path, map_location=device)
    weights = torch.load(weights_path, map_location=device)
    # drop the last dimension of weights
    weights = weights.squeeze(-1)
    # check that sample_list, lens, and weights have consistent lengths
    if len(sample_list) != len(lens) or len(sample_list) != weights.shape[0]:
        print(f"sample_list, lens, and weights lengths are inconsistent")
        return
    
    processed_weights={} 
    lens_dict={}
    num_instances = weights.shape[1]
    for idx, sample_id in enumerate(sample_list):
        sample_len = lens[idx]
        lens_dict[sample_id] = sample_len
        if sample_len <= num_instances:
            processed_weights[sample_id] = weights[idx][:sample_len]
        else:
            print(f"{sample_id} actual length exceeds num_instances, invalid sample")
            return
    # verify the processed weights, lens_dict, and sample_list
    print(f"Checking processed weights and lens_dict")
    print(f"First 10 lens entries:")
    print(f"---total length: {len(lens_dict)}")
    print(f"---keys: {list(lens_dict.keys())[1:10]}")
    print(f"First 10 processed_weights entries:")
    print(f"---total length: {len(processed_weights)}")
    print(f"---keys: {list(processed_weights.keys())[1:10]}")
    print("Length consistency check between len and weight of the first sample:")
    print(f"---len: {lens_dict[list(lens_dict.keys())[0]]}")
    print(f"---weight: {len(processed_weights[list(processed_weights.keys())[0]])}")
  
    # save the processed weights
    os.makedirs(save_dir, exist_ok=True)
    lens_save_path = os.path.join(save_dir, 'processed_lens.npy')
    np.save(lens_save_path, lens_dict)
    print(f"lens saved to: {lens_save_path}")
    weights_save_path = os.path.join(save_dir, 'processed_attns.npy')
    np.save(weights_save_path, processed_weights)
    print(f"weights reshaped to actual length and saved to: {weights_save_path}")
    return

