'''
===========================================================
WSI Heatmap Generator (Whole Slide Image Heatmap Generator)
===========================================================

Description:
    Generate heatmap visualizations for a WSI from existing patch coordinates and weights.
    Supports multiple color schemes, opacity adjustment, position/size tuning, etc.
    Applicable to result visualization for any patch-based deep-learning method.

Key features:
    - the thumbnail is scaled proportionally from the WSI size, preserving the aspect ratio
    - the heatmap is scaled from patch coordinates by the same ratio, precisely aligned with the thumbnail
    - the canvas size adapts automatically: max(thumbnail size, heatmap area size)
    - coordinate transform: position = coord x (thumbnail_size / wsi_size)
    - parameters are managed uniformly via heatmap_config.yaml

Pipeline:
    1. Parameter validation & config - validate inputs, load the config file
    2. Compute canvas bounds - get WSI size and patch coverage area size
    3. Generate thumbnail - obtain a thumbnail from the WSI, preserving aspect ratio
    4. Weight normalization - normalize weights with the chosen method (rank/minmax/zscore, etc.)
    5. Compute canvas size - max(thumbnail size, heatmap area size)
    6. Heatmap computation - build the weight heatmap, optionally with smoothing
    7. Color mapping - apply the chosen color scheme (JET/HOT/COOL, etc.)
    8. Heatmap overlay - align and overlay the heatmap on the thumbnail proportionally
    9. Enhancements - optionally add a colorbar and title

Inputs:
    1. WSI file: whole-slide image in .svs, .tif, etc.
    2. patch coordinates: N x 2 array, each row [x, y] (at the magnification given by patch_level)
    3. patch weights: N x 1 array, one weight per patch
    4. patch params: patch_size, patch_level (extraction magnification)

Outputs:
    1. thumbnail: a PIL Image, the WSI thumbnail (preserving aspect ratio)
    2. heatmap: a PIL Image, the weight heatmap overlaid on and aligned with the thumbnail

Data format example:
    coordinates = [[100, 200], [150, 250], [200, 300]]  # patch coords (numpy/torch supported)
    scores = [0.8, 0.6, 0.9]                           # patch weights (numpy/torch supported)
    patch_size = (256, 256)                            # patch size
    patch_level = 0                                    # patch extraction level (0=5x, 1=10x, 2=20x, 3=40x)

Quick usage example:
    # 1. Create the heatmap generator (auto-reads the WSI scan magnification)
    generator = PatchBasedHeatmapGenerator(
        slide_path="slide.svs",
        coordinates=coordinates,
        scores=scores,
        patch_size=(256, 256),
        patch_level=0  # patches extracted at 5x
    )

    # or specify the WSI scan magnification manually
    generator = PatchBasedHeatmapGenerator(
        slide_path="slide.svs",
        coordinates=coordinates,
        scores=scores,
        patch_size=(256, 256),
        patch_level=0,           # patches extracted at 5x
        wsi_magnification=40     # WSI scanned at 40x
    )

    # 2. Generate the heatmap (basic usage)
    thumbnail, heatmap = generator.generate_heatmap(
        thumbnail_size_scale=(0.1, 0.1),  # scale to 10% of the WSI size
        style='JET',
        alpha=0.3,
        normalize_method='rank'
    )

    # 3. Generate the heatmap (with colorbar and title)
    thumbnail, heatmap_pro = generator.generate_heatmap(
        thumbnail_size_scale=(0.1, 0.1),
        style='JET',
        alpha=0.3,
        normalize_method='rank',
        smooth=True,
        wsi_name='TCGA-Sample-01',
        add_colorbar=True,
        add_title=True
    )

Normalization method guide:
    - close    - no normalization, for data already in [0,1]
    - rank     - rank normalization, for data with outliers or unknown distribution (recommended)
    - minmax   - linear normalization, for data with known range and uniform distribution
    - zscore   - standardization, for normally distributed data
    - log      - log normalization, for positively skewed or wide-range data
    - longtail - long-tail-specific normalization, for severely skewed data such as medical images

Color scheme guide:
    - JET      - rainbow spectrum (blue->green->yellow->red), high contrast (recommended)
    - HOT      - hot spectrum (black->red->yellow->white), classic heatmap
    - COOL     - cool spectrum (cyan->magenta), soft contrast
    - TURBO    - improved rainbow spectrum, visually friendly
    - VIRIDIS  - recommended for scientific visualization, colorblind-friendly
    - default  - matplotlib default (coolwarm)

Config file: ../config/heatmap_config.yaml
'''

import cv2
import numpy as np
import openslide
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import rankdata
from scipy import ndimage
import os
from config_loader import ConfigLoader


# OpenCV colormap dictionary (class constant)
COLORMAP_DICT = {
    'JET': cv2.COLORMAP_JET, 'HOT': cv2.COLORMAP_HOT,
    'COOL': cv2.COLORMAP_COOL, 'SPRING': cv2.COLORMAP_SPRING,
    'PARULA': cv2.COLORMAP_PARULA, 'TURBO': cv2.COLORMAP_TURBO,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS, 'PLASMA': cv2.COLORMAP_PLASMA,
    'INFERNO': cv2.COLORMAP_INFERNO, 'MAGMA': cv2.COLORMAP_MAGMA
}

# Matplotlib color scheme list
MATPLOTLIB_STYLES = ['coolwarm', 'hot', 'viridis', 'plasma', 'inferno', 'magma', 'default']


class PatchBasedHeatmapGenerator:
    """
    WSI heatmap generator - WSI-ratio-based precisely aligned heatmap generator

    Key features:
        - magnification compatibility: supports different patch extraction magnifications (5x/10x/20x/40x), auto-computes the downsample factor
        - ratio alignment: thumbnail and heatmap are scaled from the WSI size by the same ratio, precisely aligned
        - independent sizing: thumbnail keeps the WSI aspect ratio; the heatmap covers only the patch area
        - canvas adaptation: canvas size is max(thumbnail, heatmap area)
        - format compatibility: supports numpy arrays and torch tensors as input
        - multiple normalizations: rank/minmax/zscore/log/longtail, etc.
        - multiple color schemes: JET/HOT/COOL/TURBO/VIRIDIS, etc.
        - smoothing: bidirectional-offset overlap sampling and Gaussian smoothing
        - flexible config: all parameters managed via a YAML config file

    Attributes:
        slide: OpenSlide object used to read the WSI
        coordinates: patch coordinate array (N x 2), at the magnification given by patch_level
        scores: patch weight array (N x 1)
        patch_size: patch size (width, height), at the magnification given by patch_level
        patch_level: patch extraction level (0=5x, 1=10x, 2=20x, 3=40x)
        patch_downsample: downsample factor from patch coordinates to level 0
        config: ConfigLoader object, the config file loader
    """

    # patch_level to magnification mapping
    LEVEL_TO_MAGNIFICATION = {0: 5, 1: 10, 2: 20, 3: 40}

    def __init__(self, slide_path, coordinates, scores, patch_size,
                 patch_level=3, wsi_magnification=None,
                 config_path='../config/heatmap_config.yaml'):
        """
        Initialize the heatmap generator

        Args:
            slide_path: path to the WSI image
            coordinates: patch coordinate array (N x 2), at the magnification given by patch_level
            scores: patch weight array (N x 1)
            patch_size: patch size (width, height), at the magnification given by patch_level
            patch_level: patch extraction level (0=5x, 1=10x, 2=20x, 3=40x), default 3 (40x)
            wsi_magnification: WSI scan magnification; if None, read automatically from the file, defaulting to 40x on failure
            config_path: path to the config file (relative or absolute)
        """
        # open the WSI file
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"WSI file not found: {slide_path}")
        self.slide = openslide.open_slide(slide_path)

        self.coordinates = self._convert_to_numpy(coordinates, "coordinates")
        self.scores = self._convert_to_numpy(scores, "scores")
        self.patch_size = patch_size
        self.patch_level = patch_level
        self._user_wsi_magnification = wsi_magnification  # user-specified WSI scan magnification

        # compute the downsample factor from patch coordinates to level 0
        self._compute_downsample_factor()

        # load the config file
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config = ConfigLoader(config_path)

        print(f"Initialization done: num patches={len(self.coordinates)}, patch size={patch_size}")
        print(f"  WSI magnification={self.wsi_magnification}x, patch_level={patch_level}({self.patch_magnification}x), "
              f"downsample factor={self.patch_downsample:.2f}")

    def _compute_downsample_factor(self):
        """
        Compute the downsample factor from patch coordinates to level 0

        downsample factor = WSI scan magnification / patch extraction magnification
        e.g. WSI=40x, patch_level=0(5x) -> downsample = 40/5 = 8
        """
        # get the WSI scan magnification: prefer auto-read; priority: user input > WSI property > default
        auto_read_success = False
        try:
            obj_power = self.slide.properties.get('openslide.objective-power')
            if obj_power is not None:
                auto_magnification = float(obj_power)
                auto_read_success = True
            else:
                # cannot auto-read, use user input or default
                if self._user_wsi_magnification is not None:
                    auto_magnification = float(self._user_wsi_magnification)
                    print(f"No magnification in WSI properties, using user value {auto_magnification}x")
                else:
                    auto_magnification = 40.0
                    print(f"No magnification in WSI properties, defaulting to 40x")
        except (ValueError, TypeError):
            if self._user_wsi_magnification is not None:
                auto_magnification = float(self._user_wsi_magnification)
                print(f"Failed to read magnification from WSI properties, using user value {auto_magnification}x")
            else:
                auto_magnification = 40.0
                print(f"Failed to read magnification from WSI properties, defaulting to 40x")

        if auto_read_success and self._user_wsi_magnification is not None and self._user_wsi_magnification != auto_magnification:
            print(f"User magnification({self._user_wsi_magnification}x) differs from WSI actual({auto_magnification}x), using the auto-read value")

        wsi_magnification = auto_magnification

        # get the patch extraction magnification
        if self.patch_level not in self.LEVEL_TO_MAGNIFICATION:
            raise ValueError(f"patch_level={self.patch_level} is not in LEVEL_TO_MAGNIFICATION; valid values: {list(self.LEVEL_TO_MAGNIFICATION.keys())}")
        patch_magnification = self.LEVEL_TO_MAGNIFICATION[self.patch_level]

        # compute the downsample factor
        self.patch_downsample = wsi_magnification / patch_magnification
        self.wsi_magnification = wsi_magnification
        self.patch_magnification = patch_magnification

    def generate_heatmap(self, 
                        thumbnail_size_scale=(0.1, 0.1),
                        style='JET', 
                        alpha=None,
                        normalize_method='rank',
                        smooth=None,
                        wsi_name='Sample',
                        add_colorbar=False,
                        add_title=False):
        """
        Generate the heatmap - core method

        Pipeline:
            1. Validate params -> 2. Get WSI size and patch area -> 3. Generate thumbnail (keep ratio)
            -> 4. Compute canvas size -> 5. Normalize weights -> 6. Compute heatmap data -> 7. Create heatmap image -> 8. Add annotations

        Args:
            thumbnail_size_scale (tuple): scale ratio (scale_x, scale_y)
                - e.g. (0.1, 0.1) means scaling to 10% of the WSI level 0 size
                - the range is controlled by config thumbnail.scale_lower_limit/upper_limit
            style (str): color scheme, options:
                - 'JET': rainbow spectrum (blue->green->yellow->red) [recommended]
                - 'HOT': hot spectrum (black->red->yellow->white)
                - 'COOL': cool spectrum (cyan->magenta)
                - 'TURBO': improved rainbow spectrum
                - 'VIRIDIS': recommended for scientific visualization
                - 'default': matplotlib default
            alpha (float): heatmap opacity, range [0.0, 1.0]
                - uses the config value when None
                - 0.0=fully transparent, 1.0=fully opaque
            normalize_method (str): normalization method, options:
                - 'close': no normalization
                - 'rank': rank normalization [recommended]
                - 'minmax': linear normalization
                - 'zscore': standardization
                - 'log': log normalization
                - 'longtail': long-tail-specific
            smooth (bool): whether to enable smoothing
                - uses the config value when None
                - True enables bidirectional-offset overlap sampling + Gaussian smoothing
            wsi_name (str): WSI sample name, used in the title
            add_colorbar (bool): whether to add a colorbar
            add_title (bool): whether to add a title

        Returns:
            tuple: (thumbnail, heatmap)
                - thumbnail (PIL.Image): WSI thumbnail, preserving the original aspect ratio
                - heatmap (PIL.Image): the heatmap overlaid on and aligned with the thumbnail

        Example:
            >>> thumbnail, heatmap = generator.generate_heatmap(
            ...     thumbnail_size_scale=(0.1, 0.1),
            ...     style='JET',
            ...     alpha=0.3,
            ...     normalize_method='rank'
            ... )
        """
        # ===== Step 1: parameter validation =====
        print("\n" + "="*60)
        print("        WSI Heatmap Generator")
        print("="*60)
        
        self._validate_parameters(thumbnail_size_scale, normalize_method)
        alpha_val = alpha if alpha is not None else self.config.get('alpha', 0.3)
        
        # ===== Step 2: compute canvas bounds =====
        print(f"\nStep 1: compute canvas bounds")
        print("-" * 30)

        coords = self.coordinates
        downsample = self.patch_downsample  # downsample factor from patch coordinates to level 0

        # patch coverage area size (converted to the level 0 coordinate system)
        # raw coordinates are at patch_level; multiply by downsample to convert to level 0
        canvas_patch_width = int((max(coord[0] for coord in coords) + self.patch_size[0]) * downsample)
        canvas_patch_height = int((max(coord[1] for coord in coords) + self.patch_size[1]) * downsample)

        # get WSI level 0 size
        wsi_width, wsi_height = self.slide.dimensions

        print(f"WSI size (level 0): {wsi_width} x {wsi_height}")
        print(f"WSI magnification: {self.wsi_magnification}x, patch magnification: {self.patch_magnification}x, downsample factor: {downsample}")
        print(f"Patch coverage area size (level 0): {canvas_patch_width} x {canvas_patch_height}")

        # ===== Step 2: generate thumbnail =====
        print(f"\nStep 2: generate thumbnail")
        print("-" * 30)

        scale_x, scale_y = thumbnail_size_scale
        # target thumbnail size based on the WSI size (preserving aspect ratio)
        target_width = int(wsi_width * scale_x) + 1
        target_height = int(wsi_height * scale_y) + 1

        # size limit checks
        min_size = self.config.get('thumbnail.min_size', 100)
        max_size = self.config.get('thumbnail.max_size', 50000)
        assert min_size <= target_width <= max_size, f"thumbnail width({target_width}) out of range"
        assert min_size <= target_height <= max_size, f"thumbnail height({target_height}) out of range"

        # get the thumbnail from the WSI (preserve aspect ratio, no forced resize)
        thumbnail = self.slide.get_thumbnail((target_width, target_height))
        actual_thumbnail_size = thumbnail.size

        # compute the actual scale ratio (based on WSI level 0 size)
        actual_scale_x = actual_thumbnail_size[0] / wsi_width
        actual_scale_y = actual_thumbnail_size[1] / wsi_height

        print(f"Target thumbnail size: {target_width} x {target_height}")
        print(f"Actual thumbnail size: {actual_thumbnail_size}")
        print(f"Actual scale ratio: X={actual_scale_x:.6f}, Y={actual_scale_y:.6f}")

        # ===== Step 3: load config parameters =====
        print(f"\nStep 3: load config parameters")
        print("-" * 30)

        x_offset = self.config.get('x_offset', 0)
        y_offset = self.config.get('y_offset', 0)

        # compute the display size of each patch in the heatmap
        # patch_size is at patch_level; first convert to level 0, then multiply by the scale ratio
        heatmap_patch_size = (
            int(self.patch_size[0] * downsample * actual_scale_x),
            int(self.patch_size[1] * downsample * actual_scale_y)
        )
        print(f"Heatmap patch display size: {heatmap_patch_size}")

        # ===== Step 5: weight normalization =====
        print(f"\nStep 4: weight normalization ({normalize_method})")
        print("-" * 30)

        scores = self._normalize_scores(self.scores.copy(), normalize_method)

        # ===== Step 6: compute canvas size =====
        # compute the heatmap area size (based on the scaled patch coordinates)
        heatmap_area_width = int(canvas_patch_width * actual_scale_x) + 1
        heatmap_area_height = int(canvas_patch_height * actual_scale_y) + 1

        # canvas size = max(thumbnail size, heatmap area size), to accommodate both
        canvas_width = max(actual_thumbnail_size[0], heatmap_area_width)
        canvas_height = max(actual_thumbnail_size[1], heatmap_area_height)
        canvas_offset_x = 0
        canvas_offset_y = 0

        print(f"Heatmap area size: {heatmap_area_width} x {heatmap_area_height}")
        print(f"Canvas size: {canvas_width} x {canvas_height}")

        # get smoothing params, used to compute the canvas expansion
        enable_smoothing = smooth if smooth is not None else self.config.get('smoothing.enable', True)
        overlap_factor = self.config.get('smoothing.overlap_factor', 1)
        step_size = self.config.get('smoothing.offset_step_size', 32)

        # compute the maximum expansion caused by smoothing offsets
        if enable_smoothing and overlap_factor >= 1:
            max_offset = overlap_factor * step_size  # maximum offset
        else:
            max_offset = 0

        # handle possible negative coordinates while accounting for smoothing-offset expansion
        # coordinate transform: patch coord x downsample factor x scale ratio = heatmap coord
        all_positions_x = [int(coord[0] * downsample * actual_scale_x) + x_offset for coord in coords]
        all_positions_y = [int(coord[1] * downsample * actual_scale_y) + y_offset for coord in coords]

        # min/max positions after accounting for smoothing offsets
        min_pos_x = min(all_positions_x) - max_offset
        max_pos_x = max(all_positions_x) + heatmap_patch_size[0] + max_offset
        min_pos_y = min(all_positions_y) - max_offset
        max_pos_y = max(all_positions_y) + heatmap_patch_size[1] + max_offset

        if min_pos_x <= 0 or min_pos_y <= 0 or max_pos_x >= canvas_width or max_pos_y >= canvas_height:
            canvas_offset_x = max(0, -min_pos_x)
            canvas_offset_y = max(0, -min_pos_y)
            canvas_width = max(canvas_width, max_pos_x) + canvas_offset_x
            canvas_height = max(canvas_height, max_pos_y) + canvas_offset_y

        if max_offset > 0:
            print(f"Canvas expansion: smoothing offset={max_offset}px, canvas size={canvas_width}x{canvas_height}")

        # ===== Step 7: compute overlay and counter =====
        print(f"\nStep 5: generate heatmap data")
        print("-" * 30)

        heatmap_params = {
            'canvas_width': canvas_width, 'canvas_height': canvas_height,
            'canvas_offset_x': canvas_offset_x, 'canvas_offset_y': canvas_offset_y,
            'actual_scale_x': actual_scale_x, 'actual_scale_y': actual_scale_y,
            'downsample': downsample,  # downsample factor from patch coordinates to level 0
            'x_offset': x_offset, 'y_offset': y_offset,
            'heatmap_patch_size': heatmap_patch_size, 'scores': scores
        }

        if enable_smoothing and overlap_factor >= 1:
            print(f"Smoothing enabled")
            overlay, counter = self._compute_smoothed_heatmap(**heatmap_params)
        else:
            print(f"Using the traditional method")
            overlay, counter = self._compute_traditional_heatmap(**heatmap_params)

        # ===== Step 8: generate the final heatmap =====
        print(f"\nStep 6: generate the final heatmap")
        print("-" * 30)

        heatmap_image,thumbnail_copy = self._create_heatmap_image(
            thumbnail, overlay, counter, actual_thumbnail_size,
            canvas_offset_x, canvas_offset_y, style, alpha_val
        )

        # ===== Step 9: add colorbar and title =====
        if add_colorbar or add_title:
            heatmap_image = self._add_annotations(
                heatmap_image, overlay, style, normalize_method, wsi_name,
                add_colorbar, add_title
            )
        
        print(f"Heatmap generation done [{style}+{normalize_method}]")
        print("="*60 + "\n")

        return thumbnail_copy, heatmap_image

    # =========================================================================
    #                           Private helper methods
    # =========================================================================

    def _convert_to_numpy(self, data, data_name):
        """
        Convert input data to numpy format

        Supports: numpy arrays, torch tensors, lists, and other convertible formats

        Args:
            data: input data (numpy/torch/list)
            data_name: data name, used in error messages

        Returns:
            numpy.ndarray: the converted numpy array

        Raises:
            ValueError: raised when conversion fails
        """
        if hasattr(data, 'cpu') and hasattr(data, 'numpy'):
            print(f'Detected {data_name} as a torch tensor, converting to numpy...')
            return data.cpu().numpy() if data.is_cuda else data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            try:
                return np.array(data)
            except Exception as e:
                raise ValueError(f"Failed to convert {data_name} to numpy: {e}")

    def _validate_parameters(self, thumbnail_size_scale, normalize_method):
        """
        Validate the input parameters

        Checks:
            1. scores and coordinates counts match
            2. patch_size is a 2-tuple
            3. thumbnail_size_scale is within the configured range
            4. normalize_method is a supported method

        Args:
            thumbnail_size_scale: scale ratio tuple
            normalize_method: normalization method name

        Raises:
            AssertionError: raised when a parameter is invalid
        """
        assert len(self.scores) == len(self.coordinates), \
            f"coordinate count({len(self.coordinates)}) does not match weight count({len(self.scores)})"
        assert len(self.patch_size) == 2, "patch_size must contain two elements"
        assert len(thumbnail_size_scale) == 2, "thumbnail_size_scale must contain two elements"

        scale_lower = self.config.get('thumbnail.scale_lower_limit', 0.001)
        scale_upper = self.config.get('thumbnail.scale_upper_limit', 1.0)
        assert scale_lower <= thumbnail_size_scale[0] <= scale_upper, f"scale_x out of range"
        assert scale_lower <= thumbnail_size_scale[1] <= scale_upper, f"scale_y out of range"

        available_methods = self.config.get('available_options.normalize_methods',
                                           ['close', 'rank', 'minmax', 'zscore', 'log', 'longtail'])
        assert normalize_method in available_methods, f"unsupported normalization method: {normalize_method}"

        print(f"Parameter validation passed: {len(self.coordinates)} patches")

    def _normalize_scores(self, scores, method):
        """
        Normalize the weight scores

        Normalization methods:
            - close: no processing, for already-normalized data
            - rank: normalize by rank to [0,1], robust to outliers [recommended]
            - minmax: linearly scale to [0,1]
            - zscore: Z-score standardization then map to [0,1]
            - log: log transform then normalize
            - longtail: piecewise nonlinear normalization, for long-tail distributions

        Args:
            scores: raw weight array
            method: normalization method name

        Returns:
            numpy.ndarray: the normalized weight array, range [0,1]
        """
        print(f"Original range: [{scores.min():.4f}, {scores.max():.4f}]")

        if method == 'close':
            pass
        elif method == 'rank':
            scores = rankdata(scores, 'average') / len(scores)
        elif method == 'minmax':
            smin, smax = scores.min(), scores.max()
            scores = (scores - smin) / (smax - smin) if smax > smin else np.zeros_like(scores)
        elif method == 'zscore':
            mean, std = scores.mean(), scores.std()
            if std > 0:
                scores = np.clip((scores - mean) / std + 3, 0, 6) / 6
            else:
                scores = np.full_like(scores, 0.5)
        elif method == 'log':
            offset = abs(scores.min()) + 1e-8 if scores.min() <= 0 else 0
            log_scores = np.log(scores + offset + 1e-8)
            lmin, lmax = log_scores.min(), log_scores.max()
            scores = (log_scores - lmin) / (lmax - lmin) if lmax > lmin else np.zeros_like(scores)
        elif method == 'longtail':
            scores = self._longtail_normalize(scores)

        print(f"Range after normalization: [{scores.min():.4f}, {scores.max():.4f}]")
        return scores

    def _longtail_normalize(self, scores):
        """Long-tail-specific normalization"""
        p75 = np.percentile(scores, 75)
        p90 = np.percentile(scores, 90)
        p95 = np.percentile(scores, 95)
        p99 = np.percentile(scores, 99)
        data_min, data_max = scores.min(), scores.max()

        scores_norm = np.zeros_like(scores)

        mask_low = scores <= p75
        if np.any(mask_low):
            scores_norm[mask_low] = 0.4 * np.sqrt((scores[mask_low] - data_min) / (p75 - data_min + 1e-8))

        mask_mid = (scores > p75) & (scores <= p90)
        if np.any(mask_mid):
            scores_norm[mask_mid] = 0.4 + 0.2 * (scores[mask_mid] - p75) / (p90 - p75 + 1e-8)

        mask_high = (scores > p90) & (scores <= p95)
        if np.any(mask_high):
            scores_norm[mask_high] = 0.6 + 0.15 * (scores[mask_high] - p90) / (p95 - p90 + 1e-8)

        mask_very_high = (scores > p95) & (scores <= p99)
        if np.any(mask_very_high):
            scores_norm[mask_very_high] = 0.75 + 0.15 * (scores[mask_very_high] - p95) / (p99 - p95 + 1e-8)

        mask_extreme = scores > p99
        if np.any(mask_extreme):
            scores_norm[mask_extreme] = 0.9 + 0.1 * (scores[mask_extreme] - p99) / (data_max - p99 + 1e-8)

        return scores_norm

    def _load_font(self, font_size):
        """Load a font, preferring a system font, falling back to the default font on failure"""
        try:
            if os.name == 'nt':
                return ImageFont.truetype("arial.ttf", font_size)
            else:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            return ImageFont.load_default()

    # =========================================================================
    #                           Heatmap computation methods
    # =========================================================================

    def _compute_smoothed_heatmap(self, canvas_width, canvas_height, canvas_offset_x, canvas_offset_y,
                                  actual_scale_x, actual_scale_y, downsample, x_offset, y_offset,
                                  heatmap_patch_size, scores):
        """
        Enhanced smoothed heatmap computation based on bidirectional-offset overlap sampling

        Algorithm:
            1. generate multiple offset versions of the heatmap around the original position
            2. each offset direction includes both positive and negative directions (bidirectional)
            3. merge all offset heatmaps using mean/max/median
            4. optionally apply a final Gaussian smoothing
            5. crop with the original boundary mask so edges match the original patch boundaries

        Offset pattern (with overlap_factor=1, 9 offsets total):
            (-s,-s) (0,-s) (s,-s)
            (-s, 0) (0, 0) (s, 0)   s = offset_step_size
            (-s, s) (0, s) (s, s)

        Args:
            canvas_width/height: canvas size
            canvas_offset_x/y: canvas offset
            actual_scale_x/y: actual scale ratio (based on WSI level 0)
            downsample: downsample factor from patch coordinates to level 0
            x_offset/y_offset: user-specified position offset
            heatmap_patch_size: patch display size in the heatmap
            scores: normalized weight array

        Returns:
            tuple: (overlay, counter)
                - overlay: heatmap data matrix
                - counter: count matrix (used to identify valid regions)
        """
        overlap_factor = self.config.get('smoothing.overlap_factor', 1)
        smoothing_method = self.config.get('smoothing.method', 'mean')
        step_size = self.config.get('smoothing.offset_step_size', 32)

        patch_num = len(self.coordinates)
        patch_w, patch_h = heatmap_patch_size

        # ===== compute the original boundary mask (no sampling offset), used to crop the offset-sampling expansion =====
        original_boundary_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
        for coord in self.coordinates:
            # coordinate transform: patch coord x downsample factor x scale ratio = heatmap coord
            canvas_x = int(coord[0] * downsample * actual_scale_x) + x_offset + canvas_offset_x
            canvas_y = int(coord[1] * downsample * actual_scale_y) + y_offset + canvas_offset_y
            if (0 <= canvas_x and 0 <= canvas_y and
                canvas_x + patch_w <= canvas_width and
                canvas_y + patch_h <= canvas_height):
                original_boundary_mask[canvas_y:canvas_y + patch_h, canvas_x:canvas_x + patch_w] = True
        print(f"  -> Computed the original boundary mask, used to crop the offset expansion")

        # build the bidirectional offset list: origin + 8 directions x overlap_factor layers
        offset_list = [(0, 0)]
        for i in range(1, overlap_factor + 1):
            s = i * step_size
            offset_list.extend([
                (s, 0), (-s, 0), (0, s), (0, -s),  # E/W/S/N
                (s, s), (-s, -s), (s, -s), (-s, s)  # four diagonals
            ])
        total_offsets = len(offset_list)
        print(f"  -> Bidirectional overlap sampling: overlap_factor={overlap_factor}, num offsets={total_offsets}")

        maps_per_offset = []

        for offset_x, offset_y in offset_list:
            current_overlay = np.zeros((canvas_height, canvas_width), dtype=np.float64)
            current_counter = np.zeros((canvas_height, canvas_width), dtype=np.uint16)

            for index in range(patch_num):
                score = scores[index]
                coord = self.coordinates[index]

                # coordinate transform: patch coord x downsample factor x scale ratio + offset = heatmap coord
                canvas_x = int(coord[0] * downsample * actual_scale_x) + x_offset + offset_x + canvas_offset_x
                canvas_y = int(coord[1] * downsample * actual_scale_y) + y_offset + offset_y + canvas_offset_y

                if (0 <= canvas_x and 0 <= canvas_y and
                    canvas_x + patch_w <= canvas_width and
                    canvas_y + patch_h <= canvas_height):
                    current_overlay[canvas_y:canvas_y + patch_h, canvas_x:canvas_x + patch_w] += score
                    current_counter[canvas_y:canvas_y + patch_h, canvas_x:canvas_x + patch_w] += 1

            # compute the average
            valid_mask = current_counter > 0
            current_overlay[valid_mask] /= current_counter[valid_mask]
            maps_per_offset.append(current_overlay)

        # merge all offset heatmaps
        print(f"  -> Merging {total_offsets} offset heatmaps, method: {smoothing_method}")
        stacked = np.stack(maps_per_offset)
        if smoothing_method == 'mean':
            merged_overlay = np.mean(stacked, axis=0)
        elif smoothing_method == 'max':
            merged_overlay = np.max(stacked, axis=0)
        elif smoothing_method == 'median':
            merged_overlay = np.median(stacked, axis=0)
        else:
            merged_overlay = np.mean(stacked, axis=0)

        # final smoothing (before cropping, using the expanded region to make edges smoother)
        if self.config.get('smoothing.enable_final', True):
            sigma = float(self.config.get('smoothing.final_sigma', 8.0))
            print(f"  -> Gaussian filter, sigma={sigma:.2f}")
            merged_overlay = ndimage.gaussian_filter(merged_overlay, sigma=sigma)

        # ===== crop with the original boundary mask, cutting off the offset-expanded parts =====
        merged_overlay = np.where(original_boundary_mask, merged_overlay, 0)
        print(f"  -> Cropped the heatmap with the original boundary mask to keep edges consistent")

        # update counter to the original boundary mask
        merged_counter = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
        merged_counter[original_boundary_mask] = 1

        return merged_overlay, merged_counter

    def _compute_traditional_heatmap(self, canvas_width, canvas_height, canvas_offset_x, canvas_offset_y,
                                     actual_scale_x, actual_scale_y, downsample, x_offset, y_offset,
                                     heatmap_patch_size, scores):
        """
        Traditional heatmap computation (no smoothing)

        Directly draws each patch's weight at its position; overlapping regions are averaged

        Args:
            canvas_width/height: canvas size
            canvas_offset_x/y: canvas offset
            actual_scale_x/y: actual scale ratio (based on WSI level 0)
            downsample: downsample factor from patch coordinates to level 0
            x_offset/y_offset: user-specified position offset
            heatmap_patch_size: patch display size in the heatmap
            scores: normalized weight array

        Returns:
            tuple: (overlay, counter)
        """
        overlay = np.zeros((canvas_height, canvas_width), dtype=np.float64)
        counter = np.zeros((canvas_height, canvas_width), dtype=np.uint16)

        patch_num = len(self.coordinates)
        patch_w, patch_h = heatmap_patch_size
        progress_step = max(1, patch_num // self.config.get('debug.progress_interval', 5))
        show_debug = self.config.get('debug.show_coordinate_debug', False)

        for index in range(patch_num):
            if index % progress_step == 0:
                print(f'Progress: {index}/{patch_num}')

            score = scores[index]
            coord = self.coordinates[index]

            # coordinate transform: patch coord x downsample factor x scale ratio = heatmap coord
            canvas_x = int(coord[0] * downsample * actual_scale_x) + x_offset + canvas_offset_x
            canvas_y = int(coord[1] * downsample * actual_scale_y) + y_offset + canvas_offset_y

            if index == 0 and show_debug:
                print(f'First patch: raw coord({coord[0]:.1f}, {coord[1]:.1f}) -> canvas position({canvas_x}, {canvas_y})')

            if (0 <= canvas_x and 0 <= canvas_y and
                canvas_x + patch_w <= canvas_width and
                canvas_y + patch_h <= canvas_height):
                overlay[canvas_y:canvas_y + patch_h, canvas_x:canvas_x + patch_w] += score
                counter[canvas_y:canvas_y + patch_h, canvas_x:canvas_x + patch_w] += 1

        print(f'Progress: {patch_num}/{patch_num}')

        # compute the average over overlapping regions
        valid_mask = counter > 0
        overlay[valid_mask] /= counter[valid_mask]

        return overlay, counter

    # =========================================================================
    #                           Image generation methods
    # =========================================================================

    def _create_heatmap_image(self, thumbnail, overlay, counter, actual_size,
                              canvas_offset_x, canvas_offset_y, style, alpha_val):
        """
        Create the overlaid heatmap - blend the heatmap data with the thumbnail

        Pipeline:
            1. expand the thumbnail size (if needed)
            2. apply the color scheme (JET/HOT/VIRIDIS, etc.)
            3. alpha-blend the heatmap with the thumbnail
            4. keep a white background outside the heatmap region

        Args:
            thumbnail: WSI thumbnail (PIL.Image)
            overlay: heatmap data matrix
            counter: count matrix
            actual_size: actual thumbnail size
            canvas_offset_x/y: canvas offset (offset caused by negative coordinates)
            style: color scheme name
            alpha_val: opacity value

        Returns:
            PIL.Image: the blended image of heatmap and thumbnail
        """
        thumbnail_copy = np.array(thumbnail.convert("RGB"))

        # expand the thumbnail size (handle the offset caused by negative coordinates)
        if (canvas_offset_x > 0 or canvas_offset_y > 0 or
            overlay.shape[1] > actual_size[0] or overlay.shape[0] > actual_size[1]):
            extended_thumbnail = np.full((overlay.shape[0], overlay.shape[1], 3), 255, dtype=np.uint8)
            extended_thumbnail[canvas_offset_y:canvas_offset_y + actual_size[1],
                             canvas_offset_x:canvas_offset_x + actual_size[0]] = thumbnail_copy
            thumbnail_copy = extended_thumbnail

        # apply the color scheme
        available_maps = self.config.get('available_options.color_maps', [])
        if style == 'default' or style not in available_maps:
            color_map = plt.get_cmap('coolwarm')
            color = (color_map(overlay) * 255)[:,:,:3].astype(np.uint8)
        else:
            heatmap_uint8 = (overlay * 255).astype(np.uint8)
            colormap = COLORMAP_DICT.get(style, cv2.COLORMAP_JET)
            color = cv2.applyColorMap(heatmap_uint8, colormap)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        # create the overlaid heatmap
        white_background = np.full_like(thumbnail_copy, 255, dtype=np.uint8)
        heat_mask = (counter > 0).astype(np.uint8)
        heat_mask_3d = np.stack([heat_mask, heat_mask, heat_mask], axis=2)

        heatmap_with_thumbnail = cv2.addWeighted(thumbnail_copy, 1 - alpha_val, color, alpha_val, 0)
        heatmap = np.where(heat_mask_3d, heatmap_with_thumbnail, white_background)

        print(f"Overlaid heatmap created")
        return Image.fromarray(heatmap), Image.fromarray(thumbnail_copy)

    # =========================================================================
    #                    Annotation methods (colorbar / title)
    # =========================================================================

    def _add_annotations(self, heatmap_image, overlay, style, normalize_method,
                         wsi_name, add_colorbar, add_title):
        """
        Add heatmap annotations (colorbar and title)

        Args:
            heatmap_image: the heatmap image
            overlay: heatmap data (used to compute the value range)
            style: color scheme
            normalize_method: normalization method name
            wsi_name: sample name
            add_colorbar: whether to add a colorbar
            add_title: whether to add a title

        Returns:
            PIL.Image: the annotated image
        """
        valid_data = overlay[overlay > 0]
        value_range = (float(valid_data.min()), float(valid_data.max())) if len(valid_data) > 0 else (0.0, 1.0)

        if add_colorbar:
            heatmap_image = self._add_colorbar(heatmap_image, value_range, style)
        if add_title:
            title_text = f"{wsi_name}_{normalize_method}"
            heatmap_image = self._add_title(heatmap_image, title_text)

        return heatmap_image

    def _add_colorbar(self, heatmap_image, value_range, style):
        """
        Add a colorbar to the heatmap

        Adds a vertical colorbar on the right side, showing the color mapping of heatmap values

        Args:
            heatmap_image: the heatmap image
            value_range: value range (min, max)
            style: color scheme name

        Returns:
            PIL.Image: the image with the colorbar added
        """
        original_width, original_height = heatmap_image.size
        right_padding = self.config.get('canvas.right_padding', 600)
        new_width = original_width + right_padding

        new_canvas = Image.new('RGB', (new_width, original_height), 'white')
        new_canvas.paste(heatmap_image, (0, 0))

        draw = ImageDraw.Draw(new_canvas)
        font_size = self.config.get('colorbar.font_size', 64)
        font = self._load_font(font_size)

        min_val, max_val = value_range
        colorbar_width = self.config.get('colorbar.width', 120)
        colorbar_height = int(original_height * self.config.get('colorbar.height_ratio', 0.6))
        colorbar_margin_right = self.config.get('colorbar.margin_right', 250)
        colorbar_x = new_width - colorbar_margin_right - colorbar_width
        colorbar_y = (original_height - colorbar_height) // 2

        is_matplotlib = style in MATPLOTLIB_STYLES
        if is_matplotlib:
            color_map = plt.get_cmap('coolwarm' if style == 'default' else style)

        # draw the colorbar
        for i in range(colorbar_height):
            ratio = (colorbar_height - 1 - i) / (colorbar_height - 1) if colorbar_height > 1 else 0
            if is_matplotlib:
                color_rgba = color_map(ratio)
                color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
            else:
                color_rgb = self._get_opencv_color(style, int(ratio * 255))
            y_pos = colorbar_y + i
            draw.line([(colorbar_x, y_pos), (colorbar_x + colorbar_width, y_pos)], fill=color_rgb, width=1)

        # draw the border
        draw.rectangle([(colorbar_x, colorbar_y),
                       (colorbar_x + colorbar_width, colorbar_y + colorbar_height)],
                      outline=(0, 0, 0), width=1)

        # draw the labels
        tick_vertical_ratio = self.config.get('colorbar.tick_vertical_ratio', 0.95)
        label_margin = self.config.get('colorbar.label_margin', 10)

        if self.config.get('colorbar.use_high_low_labels', True):
            tick_range_height = colorbar_height * tick_vertical_ratio
            tick_range_offset = (colorbar_height - tick_range_height) / 2

            high_y = colorbar_y + tick_range_offset
            high_label_x = colorbar_x + colorbar_width + label_margin
            draw.text((high_label_x, high_y - font_size // 2), "High", fill=(0, 0, 0), font=font)

            low_y = colorbar_y + tick_range_offset + tick_range_height
            draw.text((high_label_x, low_y - font_size // 2), "Low", fill=(0, 0, 0), font=font)
        else:
            tick_count = self.config.get('colorbar.tick_count', 10)
            tick_length = self.config.get('colorbar.tick_length', 20)
            tick_width = self.config.get('colorbar.tick_width', 5)

            for i in range(tick_count):
                tick_ratio = i / (tick_count - 1)
                tick_range_height = colorbar_height * tick_vertical_ratio
                tick_range_offset = (colorbar_height - tick_range_height) / 2
                tick_y = colorbar_y + tick_range_offset + int((tick_range_height - 1) * (1 - tick_ratio))
                tick_value = min_val + (max_val - min_val) * tick_ratio

                tick_x1 = colorbar_x + colorbar_width
                tick_x2 = tick_x1 + tick_length
                draw.line([(tick_x1, tick_y), (tick_x2, tick_y)], fill=(0, 0, 0), width=tick_width)

                label_x = tick_x2 + label_margin
                draw.text((label_x, tick_y - font_size // 2), f"{tick_value:.2f}", fill=(0, 0, 0), font=font)

        return new_canvas

    def _add_title(self, heatmap_image, title_text):
        """
        Add a title to the heatmap

        Adds a centered title at the bottom, format: "{sample_name}_{normalize_method}"

        Args:
            heatmap_image: the heatmap image
            title_text: the title text

        Returns:
            PIL.Image: the image with the title added
        """
        original_width, original_height = heatmap_image.size
        bottom_padding = self.config.get('canvas.bottom_padding', 200)
        new_height = original_height + bottom_padding

        new_canvas = Image.new('RGB', (original_width, new_height), 'white')
        new_canvas.paste(heatmap_image, (0, 0))

        draw = ImageDraw.Draw(new_canvas)
        font_size = self.config.get('title.font_size', 100)
        font = self._load_font(font_size)

        bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        title_margin_bottom = self.config.get('title.margin_bottom', 140)
        title_x = (original_width - text_width) // 2
        title_y = new_height - title_margin_bottom - text_height

        title_color = self.config.get('title.color', [0, 0, 0])
        draw.text((title_x, title_y), title_text, fill=tuple(title_color), font=font)

        return new_canvas

    def _get_opencv_color(self, color_map_name, value):
        """
        Get the color value from an OpenCV colormap

        Convert a single grayscale value to an RGB color of the given color scheme

        Args:
            color_map_name: color scheme name
            value: grayscale value (0-255)

        Returns:
            tuple: RGB color value (R, G, B)
        """
        single_pixel = np.array([[value]], dtype=np.uint8)
        colormap = COLORMAP_DICT.get(color_map_name.upper(), cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(single_pixel, colormap)
        color_bgr = colored[0, 0]
        return (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

    def __del__(self):
        """Release the OpenSlide resource"""
        if hasattr(self, 'slide'):
            del self.slide
