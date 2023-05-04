#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_VISIBLE_DEVICES=3

# TRAINING PARAM
PROJECT_NAME="abbpcd_fashion_18B"
VOXEL_SIZE=0.008
BATCH_SIZE=4
NUM_LABELS=6
NUM_TARGETS=`expr ${NUM_LABELS} + 1`
BACKBONE="models.Res16UNet18B"
DATASETS="abbpcd_fashion"
TRAIN_MODE="train"
MAX_EPOCHS=601
CHECK_VAL_EVERY_N_EPOCH=10
RESUME_FROM_CKPT="./saved/${PROJECT_NAME}_benchmark/last-epoch.ckpt"

# TESTING PARAM
SAVE_VISUALIZATIONS="true"
EXPORT_THRESHOLD="0.4"
DATA_EXPORT="true"
TEST_MODE="test"
USE_DBSCAN="false"
CURR_DBSCAN=0.3
CURR_TOPK=20
CURR_QUERY=150
CKPT_PATH="saved/${PROJECT_NAME}_benchmark/best.ckpt"

# TRAIN
python main.py \
  general.project_name=${PROJECT_NAME} \
  general.experiment_name="${PROJECT_NAME}_benchmark" \
  general.eval_on_segments=false \
  general.train_on_segments=false \
  general.num_targets=${NUM_TARGETS} \
  data.batch_size=${BATCH_SIZE} \
  data.num_labels=${NUM_LABELS} \
  data.voxel_size=${VOXEL_SIZE} \
  data/datasets=${DATASETS} \
  data.train_mode=${TRAIN_MODE} \
  trainer.max_epochs=${MAX_EPOCHS} \
  model.config.backbone._target_=${BACKBONE} \
  trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}
#  trainer.resume_from_checkpoint=${RESUME_FROM_CKPT}


# TEST
# python main.py \
#   general.project_name="${PROJECT_NAME}_eval" \
#   general.experiment_name="${PROJECT_NAME}_benchmark_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
#   general.checkpoint=${CKPT_PATH} \
#   general.eval_on_segments=false \
#   general.train_on_segments=false \
#   general.train_mode=false \
#   general.save_visualizations=${SAVE_VISUALIZATIONS} \
#   general.export=${DATA_EXPORT} \
#   general.export_threshold=${EXPORT_THRESHOLD} \
#   general.num_targets=${NUM_TARGETS} \
#   data.batch_size=${BATCH_SIZE} \
#   data.num_labels=${NUM_LABELS} \
#   data.voxel_size=${VOXEL_SIZE} \
#   data/datasets=${DATASETS} \
#   data.test_mode=${TEST_MODE} \
#   model.num_queries=${CURR_QUERY} \
#   model.config.backbone._target_=${BACKBONE} \
#   general.topk_per_image=${CURR_TOPK} \
#   general.use_dbscan=${USE_DBSCAN} \
#   general.dbscan_eps=${CURR_DBSCAN}





