#!/bin/bash

# Generation of raw dataset
python inference/create_custom_dataset.py

# Preprocessing on raw dataset
python dataset/preprocessing/abbpcd_preprocessing_test.py preprocess

# Run test script for inference
bash scripts/abbpcd/abbpcd_inference_custom_dataset.sh