"""
Introduction: Global variable table

Content: To generate point cloud data, usually a list of variables should be specified:

    Variables:
        - `dataset_name`:      the dataset name specified in the json file
        - `json_path`:         the json file which contains all dataset infomation
        - `labels`:            identify all the predefined labels the dataset belongs to
        - `label_types`:       indicate the numbers for splitting the dataset into different types
        - `label_color_map`:   color value to produce labels for PCD in clutter with filtered masks 
        - `split_ratio`:       ratio set for splitting raw data into train/test/validation
        - `intrinsics`:        camera intrinsics parameters necessary for pcd generation

        - `noise_filt_dict`:   defined parameters for filter to remove the noise in pcd

            Parameters
            ----------------
                filt_mode:     can either be 'depth_value' or 'frequency'
                threshold:     min/max to depth_value (one val to frequency)
                plot:          whether plot the processed result of filter
                plot_bins:     the interval for plotting the figure in x axis

Maintainer: Yihao Cai <yihao.cai@us.abb.com>
"""

import open3d as o3d
import json

# Indicate dataset name in json
dataset_name = "dhl"

# Json file for getting the dataset info
json_path = "./datasets.json"
with open(json_path, 'r') as f:
    dataset = json.load(f)
assert dataset_name in dataset

# Load dataset info by specifying its name
dataset_info = dataset[dataset_name]

# Json file path for getting the color map and labels
labels, label_color_map = [], {}
color_map_path = dataset_info['color_map_path']
with open(color_map_path, 'r') as f:
    map_data = json.load(f)

for _label, color_maps in map_data.items():
    labels.append(_label)
    label_color_map[_label] = []
    for color, value in color_maps.items():
        label_color_map[_label].append(tuple(value))

labels = tuple(labels)

# Dataset split ratio
split_ratio = dataset_info['split_ratio']

# Dataset generation property
data_ag = dataset_info['generation']['data_augmentation']
bin_gen = dataset_info['generation']['bin_gen']
write_ply = dataset_info['generation']['write_ply']
write_png = dataset_info['generation']['write_png']
write_txt = dataset_info['generation']['write_txt']
verbose = dataset_info['generation']['verbose']

# For noise filter
apply_filter = dataset_info['generation']['noise_filter']['apply']
noise_filter_dict = dataset_info['generation']['noise_filter']['property']

# Camera intrinsics parameter
cameras = dataset_info['cameras']


# Hierarchy for the data folders
# suffix = f"0/{dataset_info['camera']['serial_number']}"
color_name = "test-image.roi.color.png"
depth_name = "test-image.roi.depth.png"
