#!/usr/bin/env bash
set -euo pipefail

# Example commands used for TCGA-style pathomics files.
# Edit these paths before running.

DATA_PATH="/WorkSpace/liudongbo/Pathomics/multimagni/final_4/multi_tensor_1500_head_900_head.pth"
LABEL_PATH="/WorkSpace/liudongbo/Pathomics/multimagni/final_4/label_tensor_four_class_WGD_TP53.pt"
SAMPLE_ID_PATH="/WorkSpace/liudongbo/Pathomics/multimagni/final_4/sample_id.txt"
SAMPLE_ID="TCGA-SS-A7HO-01"
H5_PATH="/IData/DataCenter/TCGA/CRC_TCGA/OMICSData/9.Pathomics/2.Tile/SVS/${SAMPLE_ID}/HE.h5"
WSI_PATH="/IData/DataCenter/TCGA/CRC_TCGA/OMICSData/9.Pathomics/1.RawImage/SVS/${SAMPLE_ID}/HE.svs"
PATCH_LEVEL=2
PATCH_SIZE=512
CLIN_DIM=57
CLASS_DIM=4

python analysis/interpretability/run_random_heatmap_smoke.py \
  --wsi "${WSI_PATH}" \
  --h5 "${H5_PATH}" \
  --out-dir /tmp/xbm_test_01_random_heatmap \
  --patch-level "${PATCH_LEVEL}" \
  --patch-size "${PATCH_SIZE}" \
  --normalize-method rank \
  --save-thumbnail

python analysis/interpretability/run_random_model_forward_smoke.py \
  --data-path "${DATA_PATH}" \
  --label-path "${LABEL_PATH}" \
  --sample-id-path "${SAMPLE_ID_PATH}" \
  --sample-id "${SAMPLE_ID}" \
  --out-dir /tmp/xbm_test_02_scale_fusion \
  --clin-dim "${CLIN_DIM}" \
  --class-dim "${CLASS_DIM}"

python analysis/interpretability/run_integrated_gradients.py \
  --data-path "${DATA_PATH}" \
  --label-path "${LABEL_PATH}" \
  --sample-id-path "${SAMPLE_ID_PATH}" \
  --sample-id "${SAMPLE_ID}" \
  --out-dir /tmp/xbm_test_03_ig \
  --clin-dim "${CLIN_DIM}" \
  --class-dim "${CLASS_DIM}" \
  --n-steps 4

python analysis/interpretability/run_ig_heatmap.py \
  --ig-path "/tmp/xbm_test_03_ig/${SAMPLE_ID}_ig.pth" \
  --wsi "${WSI_PATH}" \
  --h5 "${H5_PATH}" \
  --out-dir /tmp/xbm_test_04_ig_heatmap \
  --patch-level "${PATCH_LEVEL}" \
  --patch-size "${PATCH_SIZE}" \
  --normalize-method rank \
  --save-thumbnail

python analysis/interpretability/run_cross_axg.py \
  --data-path "${DATA_PATH}" \
  --label-path "${LABEL_PATH}" \
  --sample-id-path "${SAMPLE_ID_PATH}" \
  --sample-id "${SAMPLE_ID}" \
  --out-dir /tmp/xbm_test_05_axg \
  --clin-dim "${CLIN_DIM}" \
  --class-dim "${CLASS_DIM}"

python analysis/interpretability/run_axg_heatmap.py \
  --axg-path "/tmp/xbm_test_05_axg/${SAMPLE_ID}_cross_axg.pth" \
  --wsi "${WSI_PATH}" \
  --h5 "${H5_PATH}" \
  --out-dir /tmp/xbm_test_06_axg_heatmap \
  --patch-level "${PATCH_LEVEL}" \
  --patch-size "${PATCH_SIZE}" \
  --normalize-method rank \
  --save-thumbnail

python analysis/interpretability/run_scale_fusion_heatmap.py \
  --scale-path "/tmp/xbm_test_02_scale_fusion/${SAMPLE_ID}_scale_fusion.pth" \
  --wsi "${WSI_PATH}" \
  --h5 "${H5_PATH}" \
  --out-dir /tmp/xbm_test_07_scale_fusion_heatmap \
  --patch-level "${PATCH_LEVEL}" \
  --patch-size "${PATCH_SIZE}" \
  --normalize-method rank \
  --save-thumbnail
