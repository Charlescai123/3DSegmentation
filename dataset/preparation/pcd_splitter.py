"""
Introduction: This module contains tools which split raw dataset to prepare them for training 

Content: The module includes the following functions and class:
    
    Classes: 
        - `PCDSplitter`:        split raw point cloud dataset into train/test/validation part
        
    Class Functions:
        - `inst_db_spawn`:      spawn a database in memory for all instances
        - `random_pick`:        pick out elements with certain number in a list randomly
        - `dict_ratio_split`:   split the dict with ratio into (train/test/validation)
        - `pcd_random_split`:   pcd splitting following format as (train/test/validation)
        - `pcd_split_s3dis`:    pcd splitting following format as s3dis
    
    Public Functions:
        - `check_path`:         check whether a path exists and create one if not 
        - `one_scene_info`:     collect name and instance number of a single scene
        - `all_scene_info`:     collect name and instance number of all scenes in a dataset

Maintainer: Yihao Cai <yihao.cai@us.abb.com>
"""

import numpy as np
import open3d as o3d
import cv2
import os
import distutils.dir_util
from variables import *
from pathlib import Path
import random
import shutil


# Create folder if not exist
def check_path(folder, verbose=False):
    if not os.path.exists(folder):
        os.mkdir(folder)
        if verbose:
            print(f"{folder} not exist, creating one.")

# Collect single scene information
def one_scene_info(single_scene_path, mode="output"):
    cnt = 0
    if mode == "input":
        txt_path = f"{single_scene_path}"
    elif mode == "output":
        txt_path = f"{single_scene_path}/txt"
    else:
        print(f"Mode {mode} not exist!")
        exit()
    txt_num = len([f.path for f in os.scandir(
        txt_path) if f.is_dir()])

    scene_name = single_scene_path.split('/')[-1]
    cnt += txt_num
    return scene_name, cnt

# Collect whole scene of dataset information
def all_scene_info(dataset_path, mode="output"):
    scenes, cnt = {}, 0
    scene_folders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    for scene_folder in scene_folders:
        scene_name, num = one_scene_info(scene_folder, mode=mode)
        scenes[scene_name] = num
        cnt += num
    return scenes, cnt


class PCDSplitter():
    def __init__(self, labels=labels, split_ratio=split_ratio) -> None:
        self.split_ratio = split_ratio
        self.labels = labels
        pass

    # Spawn a database for instance
    def inst_db_spawn(self, input_labels=None):
        if input_labels == None:
            input_labels = self.labels
        inst_db = {}
        for label in input_labels:
            inst_db[label] = []
        return inst_db

    # Randomly pick elements from a list
    def random_pick(self, origin_list, number):
        gen_list = []
        assert type(origin_list) == list and number <= len(origin_list)
        for _ in range(number):
            random_elem = random.choice(origin_list)
            gen_list.append(random_elem)
            origin_list.remove(random_elem)
        return gen_list, origin_list

    # Split data of dict in ratio (train/test/validation)
    def dict_ratio_split(self, data_dict, ratio=None):
        dataset_split = {
            'train': {},
            'test': {},
            'validation': {}}
        ratio_set =  {
            'train': [],
            'test': [],
            'validation': []}
        if ratio != None:
            assert type(ratio) == dict and len(ratio) == 3
            train_ratio = ratio['train'] / sum(ratio.values())
            validation_ratio = ratio['validation'] / sum(ratio.values())
        else:
            train_ratio = self.label_types['train'] / sum(self.label_types.values())
            validation_ratio = self.label_types['validation'] / sum(self.label_types.values())

        for scene_name, scene_paths in data_dict.items():
            total_num = len(scene_paths)

            ratio_set['train'] = round(train_ratio * total_num)
            ratio_set['validation'] = round(validation_ratio * total_num)
            ratio_set['test'] = total_num - ratio_set['train'] - ratio_set['validation']
            assert ratio_set['train'] > 0 and ratio_set['test'] > 0 and ratio_set['validation'] > 0
            
            for mode, number in ratio_set.items():
                dataset_split[mode][scene_name], _ = self.random_pick(data_dict[scene_name], number)
            assert data_dict[scene_name] == []

        return dataset_split


    # Split into train/test/validation
    def pcd_random_split(self, raw_data_input, split_data_output, ratio=None, split_mode="move"):
        assert os.path.exists(raw_data_input)

        scene_list = {}
        scene_folders = [f.path for f in os.scandir(
            Path(raw_data_input)) if f.is_dir()]
        scenes, scene_num = all_scene_info(raw_data_input)
        for k, v in scenes.items():
            scene_list[k] = []
        scene_keys = list(scenes.keys())

        if ratio == None:
            assert scene_num == sum(self.label_types.values())  

        # store and sort txt files in instances
        for scene_path in scene_folders:
            print(scene_path)
            scene_name = scene_path.split('/')[-1]
            print(scene_name)
            assert scene_name in scene_keys

            txt_folders = [f.path for f in os.scandir(
                scene_path / Path("txt")) if f.is_dir()]
            scene_list[scene_name] += txt_folders

        check_path(split_data_output)
        dataset = self.dict_ratio_split(scene_list, ratio=ratio)

        for mode, mode_dict in dataset.items():
            folder_name = f"{split_data_output}/{mode}"
            check_path(folder_name)
            for scene, scene_insts in mode_dict.items():
                for scene_inst in scene_insts:
                    this_scene_name = scene_inst.split('/')[-1]
                    scene_output = f"{folder_name}/{this_scene_name}"
                    if split_mode == "copy":
                        shutil.copytree(scene_inst, scene_output)
                        print(f"{scene_inst} copied!")
                    elif split_mode == "move":
                        shutil.move(scene_inst, folder_name)
                        print(f"{scene_inst} moved!")
                    else:
                        print(f"mode {split_mode} not exists!")
                        raise RuntimeError
                    
            print(f"{mode} folder generated.")
        
    # Split ABBPCD Cluster Dataset following s3dis format
    def pcd_split_s3dis(self, input_path, output_path):
        assert os.path.exists(input_path)

        scenes = {}
        scene_folders = [f.path for f in os.scandir(
            Path(input_path)) if f.is_dir()]

        check_path(output_path)

        for scene_path in scene_folders:

            scene_name = scene_path.split('/')[-1]
            scene_files = [f.path for f in os.scandir(
                scene_path / Path("txt")) if f.is_dir()]
            scenes[scene_name] = scene_files

        scene_keys = list(scenes.keys())

        idx = 0
        for i in range(1, 7):
            check_path(f"{output_path}/Area_{i}")

            for _ in range(80):
                random_scene_file = random.choice(scenes[scene_keys[idx]])
                random_scene_name = random_scene_file.split('/')[-1]
                print(random_scene_file)

                # copy files
                distutils.dir_util.copy_tree(random_scene_file, f"{output_path}/Area_{i}/{random_scene_name}")
                print(f"{random_scene_file} copied done!")

                # remove instance
                scenes[scene_keys[idx]].remove(random_scene_file)

                # idx changing
                idx = (idx + 1) % len(scene_keys)

def test():
    print("Testing")

    # files_list = os.listdir(src_folder)
    # print(files_list)
    # check_path(f"{dst_folder}/test_folder")
    # for files in files_list:
    #     print(files)
    #     shutil.move(f"{src_folder}/{files}", f"{dst_folder}/test_folder")

    # scenes, num = all_scene_info("../Dataset/Raw_Gen_Dataset/ABBPCD_Envelope_Bin")
    # input_scenes, num1 = all_scene_info("../Dataset/Collected_Dataset/Complex_Scenario/amazon_envelope/amazon_envelope_data", mode="input")
    # output_scenes, num2 = all_scene_info("../Dataset/Raw_Gen_Dataset/ABBPCD_Envelope_Bin", mode="output")
    # print(input_scenes, num1)
    # print(output_scenes, num2)

    # new_dict = {k: v * 4 for k, v in input_scenes.items()}
    # print(new_dict)
    # assert new_dict == output_scenes

    pass

if __name__ == "__main__":
    test()
    pass
