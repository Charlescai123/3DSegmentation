#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_VISIBLE_DEVICES=1

# SOCKET
HOST="localhost"
PORT="12321"

# INFERENCE PARAM
PROJECT_NAME="abbpcd_fashion_inference"
VOXEL_SIZE=0.008
BATCH_SIZE=4
NUM_LABELS=6
NUM_TARGETS=`expr ${NUM_LABELS} + 1`
DATASET_YAML="dataset/configuration/dataset_yaml/abbpcd_fashion.yaml"
DATASETS="abbpcd_fashion"
SAVE_VISUALIZATIONS="true"
EXPORT_THRESHOLD="0.4"
DATA_EXPORT="true"
TEST_MODE="test"
USE_DBSCAN="false"
CURR_DBSCAN=0.3
CURR_TOPK=10
CURR_QUERY=160
CKPT_PATH="saved/abbpcd_fashion_18B_overlap_benchmark/best.ckpt"

# INFERENCE
python main.py \
  general.project_name="${PROJECT_NAME}" \
  general.experiment_name="${PROJECT_NAME}" \
  general.checkpoint=${CKPT_PATH} \
  general.inference=true \
  general.host=${HOST} \
  general.port=${PORT} \
  general.dataset_yaml=${DATASET_YAML} \
  general.eval_on_segments=false \
  general.train_on_segments=false \
  general.train_mode=false \
  general.save_visualizations=${SAVE_VISUALIZATIONS} \
  general.export_threshold=${EXPORT_THRESHOLD} \
  general.export=${DATA_EXPORT} \
  general.num_targets=${NUM_TARGETS} \
  data.batch_size=${BATCH_SIZE} \
  data.num_labels=${NUM_LABELS} \
  data.voxel_size=${VOXEL_SIZE} \
  data/datasets=${DATASETS} \
  data.test_mode=${TEST_MODE} \
  model.num_queries=${CURR_QUERY} \
  general.topk_per_image=${CURR_TOPK} \
  general.use_dbscan=${USE_DBSCAN} \
  general.dbscan_eps=${CURR_DBSCAN}





