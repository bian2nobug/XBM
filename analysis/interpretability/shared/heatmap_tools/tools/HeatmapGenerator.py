'''
    Description: multi-step orchestration script, including merging tiles weight/coordinate data and heatmap generation
    Adapted to the PatchBasedHeatmapGenerator calling convention
'''
import sys
import os

# get the directory of the current script and add it to the path dynamically
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import numpy as np
from scipy import stats
from PatchBasedHeatmapGenerator import PatchBasedHeatmapGenerator

#=== Main heatmap generation function - auto-analyze data distribution and pick the best normalization ===#
def heatmap_main(wsi_path, coordinates, scores, patch_size, patch_level=0, thumbnail_dir='.', heatmap_dir='.', heatmap_params=None, save_thumbail=True):
    '''
    Description: main heatmap generation function - auto-analyze data distribution and pick the best normalization

    Args:
        wsi_path (str)        : path to the WSI file
                               e.g. "path/to/HE.svs"
        coordinates (ndarray) : patch coordinate array, shape: (n_patches, 2)
                               e.g. np.array([[x1, y1], [x2, y2], ...])
        scores (ndarray)      : patch score array, shape: (n_patches,)
                               e.g. np.array([0.1, 0.8, 0.3, ...])
        patch_size (tuple)    : patch size, e.g. (256, 256)
        patch_level (int)     : patch extraction level (0=5x, 1=10x, 2=20x, 3=40x), default 0 (5x)
        thumbnail_dir (str)   : thumbnail output directory
                               e.g. "/.../attention_heatmap_result/test/thumbnail"
        heatmap_dir (str)     : heatmap output directory
                               e.g. "/.../attention_heatmap_result/test/heatmap"
        heatmap_params (dict) : heatmap generation params, e.g. None (use defaults)
        save_thumbail (bool)  : whether to save the thumbnail, default True

    Returns:
        None

    Features:
        - automatically reads the scan magnification from the WSI
        - automatically analyzes distribution characteristics (skewness, kurtosis, CV, etc.)
        - intelligently selects the most suitable normalization method
        - generates the main heatmap and comparison heatmaps
        - outputs a detailed data analysis report
    '''
    # check that the WSI file exists
    assert os.path.isfile(wsi_path), f"WSI file not found: {wsi_path}"

    # derive the WSI sample ID
    wsi_name = os.path.basename(os.path.dirname(wsi_path))

    # append the sample ID to heatmap_dir and thumbnail_dir
    heatmap_dir = os.path.join(heatmap_dir)
    thumbnail_dir = os.path.join(thumbnail_dir)

    # ensure output directories exist
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(thumbnail_dir, exist_ok=True)

    print(f'Starting heatmap and thumbnail generation for sample {wsi_name}')

    # ============================================================
    #        Data distribution analysis & normalization selection
    # ============================================================
    print(f"\nAnalyzing distribution characteristics of sample {wsi_name}...")
    best_method, analysis_info = analyze_data_distribution(scores)

    if heatmap_params is None:
        heatmap_params = {
            'thumbnail_size_scale': (0.125, 0.125), 
            'style': 'JET', 
            'alpha': None, 
            'normalize_method': best_method,  # or use best_method
            'smooth': True,
            'wsi_name': wsi_name,  # or use wsi_name
            'add_colorbar': True,
            'add_title': True
        }

    # generate the heatmap (scan magnification is read automatically from the WSI)
    heatmap_generator = PatchBasedHeatmapGenerator(wsi_path, coordinates, scores, patch_size, patch_level=patch_level)

    #                   generate the heatmap using the recommended method
    # ============================================================

    normalize_method = heatmap_params['normalize_method']
    print(f"\nGenerating heatmap with normalization method '{normalize_method}'...")
    # generate the main heatmap (using the recommended method, with colorbar and title)
    thumbnail_best, heatmap_best = heatmap_generator.generate_heatmap(
        thumbnail_size_scale=heatmap_params['thumbnail_size_scale'],
        style=heatmap_params['style'],
        alpha=heatmap_params['alpha'],
        normalize_method=normalize_method,
        smooth=heatmap_params['smooth'],
        wsi_name=heatmap_params['wsi_name'],
        add_colorbar=heatmap_params['add_colorbar'],
        add_title=heatmap_params['add_title']
    )
    heatmap_best.save(os.path.join(heatmap_dir, f'heatmap_{normalize_method}_auto.jpg'), 
                  quality=65, optimize=True)
    if save_thumbail:
        thumbnail_best.save(os.path.join(thumbnail_dir, f'thumbnail.jpg'), 
                        quality=70, optimize=True)
    print(f"[{normalize_method.upper()}] heatmap saved at: {os.path.join(heatmap_dir, f'heatmap_{normalize_method}_auto.png')}")
    print(f"[{normalize_method.upper()}] thumbnail saved at: {os.path.join(thumbnail_dir, f'thumbnail.png')}")
    del heatmap_best
    del thumbnail_best

    # ============================================================
    #        Comparison test: generate heatmaps with other methods
    # ============================================================

    # check whether best_method is rank
    is_rank = normalize_method == 'rank'
    if is_rank:
        compare_methods = None
        print(f"best_method is rank, skipping comparison")
    else:
        all_methods = ['rank', 'zscore', 'log', 'longtail', 'minmax']
        compare_methods = [method for method in all_methods if method != normalize_method]

    if compare_methods is not None:
        print(f"\nGenerating comparison heatmaps to test other normalization methods...")
        for method in compare_methods:
            try:
                thumbnail_compare, heatmap_compare = heatmap_generator.generate_heatmap(
                thumbnail_size_scale=heatmap_params['thumbnail_size_scale'],
                style=heatmap_params['style'],
                alpha=heatmap_params['alpha'],
                normalize_method=method,
                smooth=heatmap_params['smooth'],
                wsi_name=heatmap_params['wsi_name'],
                add_colorbar=heatmap_params['add_colorbar'],
                add_title=heatmap_params['add_title']
                )
                heatmap_compare.save(os.path.join(heatmap_dir, f'heatmap_{method}_compare.jpg'),quality=75, optimize=True)
                print(f"  [{method.upper()}] comparison heatmap generated")
                del heatmap_compare
                del thumbnail_compare
            except Exception as e:
                print(f"  [{method.upper()}] generation failed: {e}")

    # generate the analysis report
    print(f"\nSaving data analysis report...")
    report_path = os.path.join(heatmap_dir, f'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Data distribution analysis report for sample {wsi_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Recommended normalization method: {normalize_method}\n\n")
        f.write("Data statistics:\n")
        for key, value in analysis_info.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write(f"\nGenerated files:\n")
        f.write(f"  main heatmap: heatmap_{normalize_method}_auto.png\n")
        f.write(f"  thumbnail: thumbnail.png\n")
        if compare_methods is not None:
            for method in compare_methods:
                f.write(f"  comparison heatmap({method}): heatmap_{method}_compare.png\n")

        print(f"Analysis report saved at: {report_path}")

def analyze_data_distribution(scores):
    """
    Description: analyze distribution characteristics and automatically select the best normalization method
    Args:
        scores: patch score array (list, numpy array, or torch tensor)
    Returns:
        best_method: the recommended normalization method
        analysis_info: dictionary of analysis details
    """
    # handle torch tensor
    import torch
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # convert to a numpy array and flatten to 1D
    scores_array = np.array(scores).flatten()

    # basic statistics (ensure all are scalars)
    data_mean = float(scores_array.mean())
    data_std = float(scores_array.std())
    data_min = float(scores_array.min())
    data_max = float(scores_array.max())
    data_range = data_max - data_min

    # distribution characteristics
    skewness = float(stats.skew(scores_array))  # skewness, ensure scalar
    kurtosis = float(stats.kurtosis(scores_array))  # kurtosis, ensure scalar
    cv = float(data_std / data_mean) if data_mean != 0 else float('inf')  # coefficient of variation

    # percentile analysis (ensure all are scalars)
    p25 = float(np.percentile(scores_array, 25))
    p50 = float(np.percentile(scores_array, 50))  # median
    p75 = float(np.percentile(scores_array, 75))
    p90 = float(np.percentile(scores_array, 90))
    p95 = float(np.percentile(scores_array, 95))
    p99 = float(np.percentile(scores_array, 99))

    # concentration analysis
    iqr = p75 - p25  # interquartile range
    median_to_max_ratio = float(p50 / data_max) if data_max != 0 else 0.0
    low_value_ratio = float(np.sum(scores_array <= p75) / len(scores_array))  # low-value ratio

    # analysis details
    analysis_info = {
        'mean': data_mean,
        'std': data_std,
        'min': data_min,
        'max': data_max,
        'range': data_range,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'cv': cv,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'p95': p95,
        'p99': p99,
        'iqr': iqr,
        'median_to_max_ratio': median_to_max_ratio,
        'low_value_ratio': low_value_ratio
    }

    print("\n" + "="*60)
    print("                   Data distribution analysis")
    print("="*60)
    print(f"Basic statistics:")
    print(f"  mean: {data_mean:.6f}")
    print(f"  std: {data_std:.6f}")
    print(f"  range: [{data_min:.6f}, {data_max:.6f}]")
    print(f"  CV: {cv:.2f}")
    print(f"\nDistribution characteristics:")
    print(f"  Skewness: {skewness:.2f}")
    print(f"  Kurtosis: {kurtosis:.2f}")
    print(f"  median/max ratio: {median_to_max_ratio:.4f}")
    print(f"  low-value(<=P75) ratio: {low_value_ratio:.1%}")
    print(f"\nPercentiles:")
    print(f"  P25: {p25:.6f}, P50: {p50:.6f}, P75: {p75:.6f}")
    print(f"  P90: {p90:.6f}, P95: {p95:.6f}, P99: {p99:.6f}")

    # normalization method selection logic
    print(f"\nNormalization method selection logic:")

    # condition 1: data already normalized
    if data_min >= 0 and data_max <= 1 and data_range > 0.8:
        best_method = 'close'
        reason = "data already in [0,1] with a good spread"

    # condition 2: severe long-tail distribution (typical of medical images)
    elif (skewness > 3 and cv > 10 and median_to_max_ratio < 0.1 and low_value_ratio > 0.75):
        best_method = 'longtail'
        reason = f"severe long-tail distribution (skew={skewness:.1f}, CV={cv:.1f}, low-value ratio={low_value_ratio:.1%})"

    # condition 3: unknown distribution or outliers (safe choice)
    elif (cv > 5 or abs(kurtosis) > 2 or skewness > 2):
        best_method = 'rank'
        reason = f"irregular distribution or outliers (skew={skewness:.1f}, kurt={kurtosis:.1f}, CV={cv:.1f})"

    # condition 4: close to normal distribution
    elif (abs(skewness) < 0.5 and abs(kurtosis) < 1 and cv < 2):
        best_method = 'zscore'
        reason = f"close to normal distribution (skew={abs(skewness):.1f}, kurt={abs(kurtosis):.1f})"

    # condition 5: positively skewed, suitable for log transform
    elif (skewness > 1 and cv > 1 and data_min > 0):
        best_method = 'log'
        reason = f"positively skewed, suitable for log transform (skew={skewness:.1f})"

    # condition 6: known range without outliers
    elif (cv < 1 and abs(skewness) < 1):
        best_method = 'minmax'
        reason = f"relatively uniform distribution (CV={cv:.1f}, skew={abs(skewness):.1f})"

    # default choice
    else:
        best_method = 'rank'
        reason = "default choice, suitable for the general case"

    print(f"Recommended method: {best_method}")
    print(f"Reason: {reason}")
    print("="*60)

    return best_method, analysis_info

#=== Usage examples ===#
'''
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "model_lib"))
from HeatmapGenerator import heatmap_generate

tile_path = "path/to/HE.h5"
wsi_path = "path/to/HE.svs"
weight_path = "path/to/TCGA-A6-5660-01.npy"
patch_size = (256, 256)
thumbnail_dir = "path/to/thumbnail"
heatmap_dir = "path/to/heatmap"

# heatmap_params does not need to include wsi_name
heatmap_params = {
    'thumbnail_size_scale': (0.125, 0.125),
    'style': 'JET',
    'alpha': None,
    'normalize_method': 'rank',
    'smooth': True,
    'add_colorbar': True,
    'add_title': True
}

heatmap_generate(tile_path, wsi_path, weight_path, patch_size, thumbnail_dir, heatmap_dir, heatmap_params)
'''

'''
# usage example for heatmap_main
# obtain the data required for the heatmap
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "model_lib"))
import h5py
import os
import torch
from HeatmapGenerator import heatmap_main

#======= generate a single WSI heatmap =======#
heatmap_dir = "path/to/heatmap"
thumbnail_dir = "path/to/thumbnail"
h5_path = "path/to/TCGA-4N-A93T-01.h5"

wsi_path = "path/to/HE.svs"
with h5py.File(h5_path, 'r') as f:
    coordinates = f['TCGA-4N-A93T-01']['coordinates'][()]
    scores = f['TCGA-4N-A93T-01']['scores'][()]

tile_size = 256
patch_size = (tile_size, tile_size)
heatmap_main(wsi_path = wsi_path, coordinates = coordinates, scores = scores, patch_size = patch_size, thumbnail_dir = thumbnail_dir, heatmap_dir = heatmap_dir)
'''
