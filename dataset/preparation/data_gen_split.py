"""
Introduction: This module leverages PCDGenerator to generate pcd with labels and 
                PCDSplitter for pcd preparation 

Content: The module includes the following functions:

    Function:
        - `filt_black_masks`:        filter out black masks after mask generation process
        - `single_scene_process`:    produce raw data with annotations for a single scene
        - `abbpcd_gen_with_label`:   raw data gen entrance (support sole-thread/multi-thread)
        - `main`:                    main entrance for generating raw data and split it

Maintainer: Yihao Cai <yihao.cai@us.abb.com>
"""

import numpy as np
import time
import cv2
import os
from matplotlib import pyplot as plt
from pcd_splitter import *
from pcd_generator import *
from variables import *
from joblib import Parallel, delayed
import glob

# Filter all the black masks in the list
def filt_black_masks(inst_masks, black_mask):
    assert type(inst_masks) == list
    new_masks = []
    for inst_mask in inst_masks:
        if not (inst_mask == black_mask).all():
            new_masks.append(inst_mask)
    return new_masks

# Process single scene
def single_scene_process(ori_path, label_path, out_path, bg_gen=False, data_ag=True, noise_filt=True,
                         write_ply=True, write_png=True, write_txt=True, verbose=False):
    scene_insts = [f.path for f in os.scandir(label_path) if f.is_dir()]

    scene_name = ori_path.split('/')[-1]
    scene_path = f"{out_path}"
    check_path(scene_path)

    if write_ply:
        ply_path = f"{out_path}/ply"
        check_path(ply_path)

    if write_png:
        png_path = f"{out_path}/png"
        check_path(png_path)

    if write_txt:
        txt_path = f"{out_path}/txt"
        check_path(txt_path)

    idx = 1
    for scene_inst in scene_insts:

        # Check camera serial number
        files = glob.glob(f'{scene_inst}/*/*')
        for f in files:
            if os.path.isdir(f):
                l = f.split('/')
                suffix = l[-2] + '/' + l[-1]
                break
        for _, cam_info in cameras.items():
            if suffix.split('/')[-1] == cam_info['serial_number']:
                _intrinsics = cam_info['intrinsics']
                _width = cam_info['width']
                _height = cam_info['height']
                break

        # Init PCDGenerator()
        pcd_generator = PCDGenerator(intrinsics=_intrinsics, width=_width, height=_height)

        # Begin process
        cnt = scene_inst.rsplit('-', maxsplit=1)[-1]
        inst_name = scene_inst.split('/')[-1]

        img_bgr_ori = cv2.imread(
            f"{ori_path}/{inst_name}/{suffix}/{color_name}")
        img_bgr = cv2.imread(f"{scene_inst}/{suffix}/{color_name}")
        img_dpt = cv2.imread(f"{scene_inst}/{suffix}/{depth_name}")
        img_gray = cv2.imread(f"{scene_inst}/{suffix}/{color_name}", 0)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_ori = cv2.cvtColor(img_bgr_ori, cv2.COLOR_BGR2RGB)

        if noise_filt:
            # From 3 channels to 1 channel
            img_dpt = cv2.cvtColor(img_dpt, cv2.COLOR_BGR2GRAY)
            # Apply noise filter with threshold
            img_dpt = pcd_generator.noise_filter(img_dpt)

        black_mask = np.zeros(
            img_gray.shape, dtype=np.uint8)       # Black Mask
        combined_mask = black_mask                                  # Combined Mask

        # Generate Masks
        masks = {}
        mappings = label_color_map
        for label_name, color_th_list in mappings.items():
            masks[label_name] = []
            inst_masks = pcd_generator.mask_gen_instance(img_rgb, color_th_list)
            for inst_mask in inst_masks:
                masks[label_name].append(inst_mask)
                combined_mask = cv2.bitwise_or(combined_mask, inst_mask)
        bin_mask = ~combined_mask

        # Augmented data
        if data_ag:
            bin_mask_ag = pcd_generator.data_augment(bin_mask)
            combined_mask_ag = pcd_generator.data_augment(combined_mask)
            img_rgb_ori_ag = pcd_generator.data_augment(img_rgb_ori)
            img_dpt_ag = pcd_generator.data_augment(img_dpt)
            assert len(img_rgb_ori_ag) == len(img_dpt_ag) == len(
                combined_mask_ag) == len(bin_mask_ag)

        # Write PNG
        if write_png:
            png_inst_path = f"{scene_name}_{cnt}"
            check_path(f"{png_path}/{png_inst_path}")
            img_bgr_ori_filt = cv2.bitwise_and(
                img_bgr_ori, img_bgr_ori, mask=combined_mask)
            img_bgr_filt = cv2.bitwise_and(
                img_bgr, img_bgr, mask=combined_mask)
            cv2.imwrite(
                f"{png_path}/{png_inst_path}/color_origin.png", img_bgr_ori)
            cv2.imwrite(f"{png_path}/{png_inst_path}/color_label.png", img_bgr)
            cv2.imwrite(
                f"{png_path}/{png_inst_path}/color_filt.png", img_bgr_ori_filt)
            cv2.imwrite(
                f"{png_path}/{png_inst_path}/label_filt.png", img_bgr_filt)
            cv2.imwrite(f"{png_path}/{png_inst_path}/mask.png", combined_mask)

        # Write PLY
        if write_ply and not (combined_mask == black_mask).all():
            ply_inst_path = f"{ply_path}/{scene_name}_{cnt}"
            check_path(ply_inst_path)

            for label_name, inst_masks in masks.items():
                filtered_inst_masks = filt_black_masks(inst_masks, black_mask)
                if len(filtered_inst_masks) > 0:
                    for id, mask in enumerate(filtered_inst_masks, 1):
                        pcd_generator.pcd_write(img_rgb_ori, img_dpt, ply_plot=False, format='ply',
                                                 output_path=f"{ply_inst_path}/{label_name}_{id}.ply", mask=mask)
            if bg_gen:
                pcd_generator.pcd_write(img_rgb=img_rgb_ori, img_depth=img_dpt, ply_plot=False, format='ply',
                                         output_path=f"{ply_inst_path}/bin_1.ply", mask=bin_mask)

        # Write TXT (should be 4 for 1 scene)
        if write_txt:
            for label_name, inst_masks in masks.items():
                filtered_inst_masks = filt_black_masks(inst_masks, black_mask)

                # Write txt of annotations
                if len(filtered_inst_masks) > 0:
                    for id, mask in enumerate(filtered_inst_masks, 1):
                        # If use data augmentation
                        if data_ag:
                            mask_ag = pcd_generator.data_augment(mask)
                            assert len(img_rgb_ori_ag) == len(img_dpt_ag) == len(
                                mask_ag) == len(combined_mask_ag) == len(bin_mask_ag)

                            # Generated augmented dataset
                            for i in range(idx, idx + len(mask_ag)):
                                check_path(f"{txt_path}/{scene_name}_{i}")
                                check_path(
                                    f"{txt_path}/{scene_name}_{i}/Annotations")
                                pcd_generator.pcd_write(img_rgb_ori_ag[i - idx], img_dpt_ag[i - idx], ply_plot=False, format='txt',
                                                         output_path=f"{txt_path}/{scene_name}_{i}/Annotations/{label_name}_{id}.txt", mask=mask_ag[i - idx])

                            if bg_gen:
                                for i in range(idx, idx + len(mask_ag)):
                                    pcd_generator.pcd_write(img_rgb_ori_ag[i - idx], img_dpt_ag[i - idx], ply_plot=False, format='txt',
                                                             output_path=f"{txt_path}/{scene_name}_{i}/Annotations/bin_1.txt", mask=bin_mask_ag[i - idx])

                        else:
                            check_path(f"{txt_path}/{scene_name}_{idx}")
                            check_path(
                                f"{txt_path}/{scene_name}_{idx}/Annotations")
                            pcd_generator.pcd_write(img_rgb_ori, img_dpt, ply_plot=False, format='txt',
                                                     output_path=f"{txt_path}/{scene_name}_{idx}/Annotations/{label_name}_{id}.txt", mask=mask)

                            if bg_gen:
                                pcd_generator.pcd_write(img_rgb_ori, img_dpt, ply_plot=False, format='txt',
                                                         output_path=f"{txt_path}/{scene_name}_{idx}/Annotations/bin_1.txt", mask=bin_mask)

                # Write txt of whole scene
                if len(filtered_inst_masks) > 0:
                    if data_ag:
                        for i in range(idx, idx + len(img_dpt_ag)):
                            txt_inst_path = f"{txt_path}/{scene_name}_{i}"
                            if bg_gen:
                                pcd_generator.pcd_write(img_rgb_ori_ag[i - idx], img_dpt_ag[i - idx], ply_plot=False, format='txt',
                                                         output_path=f"{txt_inst_path}/{scene_name}_{i}.txt")
                            else:
                                pcd_generator.pcd_write(img_rgb_ori_ag[i - idx], img_dpt_ag[i - idx], ply_plot=False, format='txt',
                                                         output_path=f"{txt_inst_path}/{scene_name}_{i}.txt", mask=combined_mask_ag[i - idx])
                    else:
                        if bg_gen:
                            pcd_generator.pcd_write(img_rgb_ori, img_dpt, ply_plot=False, format='txt',
                                                     output_path=f"{txt_path}/{scene_name}_{idx}/{scene_name}_{idx}.txt")
                        else:
                            pcd_generator.pcd_write(img_rgb_ori, img_dpt, ply_plot=False, format='txt',
                                                     output_path=f"{txt_path}/{scene_name}_{idx}/{scene_name}_{idx}.txt", mask=combined_mask)
        if data_ag:
            idx += len(img_dpt_ag)
        else:
            idx += 1
        if verbose:
            print(f"Processing {scene_inst} done!")

    print(f"Processing {scene_name} done!")

    # Validate scene has been generated in a correct way
    input_scene_name, num1 = one_scene_info(ori_path, mode="input")
    output_scene_name, num2 = one_scene_info(out_path, mode="output")
    if verbose:
        print(f"number of original scene instance is: {num1}")
        print(f"number of generated scene instance is: {num2}")

    if data_ag:
        num1 = num1 * len(img_dpt_ag)

    # Must be true if all txt files generated successfully
    assert input_scene_name == output_scene_name and num1 == num2

# Generate ABBPCD (use multi_gpu or not)


def abbpcd_gen_with_label(origin_path, label_path, output_path, bg_gen=True, data_ag=True, filter=True,
                          write_ply=True, write_png=True, write_txt=True, multi_gpu=None, verbose=True):
    assert os.path.exists(origin_path)
    assert os.path.exists(label_path)

    scene_names = [f.path.split('/')[-1]
                   for f in os.scandir(label_path) if f.is_dir()]
    check_path(output_path)

    if multi_gpu == None:
        for scene_name in scene_names:
            single_scene_process(f"{origin_path}/{scene_name}", f"{label_path}/{scene_name}", f"{output_path}/{scene_name}", noise_filt=filter,
                                 bg_gen=bg_gen, data_ag=data_ag, write_ply=write_ply, write_png=write_png, write_txt=write_txt, verbose=verbose)
    else:
        _ = Parallel(n_jobs=multi_gpu, verbose=10)(
            delayed(single_scene_process)(f"{origin_path}/{scene_name}", f"{label_path}/{scene_name}", f"{output_path}/{scene_name}", noise_filt=filter,
                                          bg_gen=bg_gen, data_ag=data_ag, write_ply=write_ply, write_png=write_png, write_txt=write_txt, verbose=verbose)
            for scene_name in scene_names
        )
    _, num1 = all_scene_info(origin_path, mode="input")
    _, num2 = all_scene_info(output_path, mode="output")
    print(f"number of original scene instance is: {num1}")
    print(f"number of generated scene instance is: {num2}")

# For testing


def test():
    # input_scene_info, num1 = all_scene_info(origin_path, mode="input")
    # output_scene_info, num2 = all_scene_info(output_folder, mode="output")
    # print(f"number of original scene instance is: {num1}")
    # print(f"number of generated scene instance is: {num2}")
    # print(f"{input_scene_info}")
    # print(f"{output_scene_info}")
    pass


def main(raw_gen=True, data_split=True):
    # Input Path
    input_origin = dataset_info['input_data_path']
    input_label = dataset_info['input_label_path']

    # Output Raw Path
    output_raw_folder = dataset_info['output_raw_path']
    output_split_folder = dataset_info['output_split_path']

    if raw_gen:
        # Generate Raw Data
        start_time = time.time()
        abbpcd_gen_with_label(origin_path=input_origin, label_path=input_label, filter=apply_filter, data_ag=data_ag, bg_gen=bin_gen,
                              output_path=output_raw_folder, write_ply=write_ply, write_png=write_png, write_txt=write_txt, multi_gpu=-1, verbose=verbose)
        end_time = time.time()
        process_time = end_time - start_time
        process_time = "{:.2f}".format(process_time)
        print(f"The time cost for generating data is: {process_time}s")

    if data_split:
        # Data Split
        start_time = time.time()
        PCDSplitter().pcd_random_split(output_raw_folder, output_split_folder,
                                       ratio=split_ratio, split_mode="move")
        end_time = time.time()
        process_time = end_time - start_time
        process_time = "{:.2f}".format(process_time)
        print(f"The time cost for splitting data is: {process_time}s")


if __name__ == "__main__":
    # test()
    main(raw_gen=True, data_split=True)
