"""
Introduction: This module contains functions for generating point cloud data (PCD).

Content: The module includes the class PCDGenerator with following functions:
    
    Classes: 
        - `PCDGenerator`:       used for generating raw point cloud data with collected dataset
    
    Class Functions:
        - `pcd_write`:          write pcd to either ply or txt format after getting processed by masks
        - `rgb_filter`:         use background subtraction and thresh value to filter out objects 
        - `data_augment`:       data augmentation for dataset scaling (flip horizontally/vertically/both)
        - `noise_filter`:       apply a passthrough filter on depth image to decrease noises
        - `mask_gen_semantic`:  generate mask for labeling pcd (semantic)
        - `mask_gen_instance`:  generate mask for labeling pcd (instance)

    Public Functions:
        - `one_ply_visualize`:  visualize a single ply file
        - `all_ply_visualize`:  visualize all ply files in a scene/dataset
        
Maintainer: Yihao Cai <yihao.cai@us.abb.com>
"""

import numpy as np
import open3d as o3d
import cv2
import os
from matplotlib import pyplot as plt
from typing import List
from pcd_splitter import *
from variables import *

# One ply visualization
def one_ply_visualize(input_ply):
    pcd = o3d.io.read_point_cloud(input_ply)
    o3d.visualization.draw_geometries([pcd])
    
# For all ply visualization
def all_ply_visualize():
    input_path = "./Result_Collapse"
    scene_folders = [f.path for f in os.scandir(input_path) if f.is_dir()]
    print(scene_folders)
    for scene_folder in scene_folders:
        ply_files = [f.path for f in os.scandir(f"{scene_folder}/ply") if f.is_file()]
        print(ply_files)
        for ply_file in ply_files:
            one_ply_visualize(ply_file)

class PCDGenerator():
    #intrinsics = intrinsics

    def __init__(self, intrinsics, width, height, label_color_map=label_color_map) -> None:
        self.label_color_map = label_color_map

        fx, _, cx, _, fy, cy, _, _, _ = intrinsics
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
        )
        pass

    def rgb_filter(self, rgb_input, rgbBg_input, threshVal, res_plot=False):
        # Load original image
        img = cv2.imread(rgb_input)
        imgBg = cv2.imread(rgbBg_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # From BGR to RGB
        imgBg = cv2.cvtColor(imgBg, cv2.COLOR_BGR2RGB)  # From BGR to RGB

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayBg = cv2.cvtColor(imgBg, cv2.COLOR_BGR2GRAY)

        # Convert to binary inverted image
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Background Subtractor
        sharped_img = cv2.subtract(img, imgBg)

        # Generate Mask
        mask = np.zeros(sharped_img.shape, np.uint8)
        sharped_gray = cv2.cvtColor(sharped_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(sharped_gray, threshVal, 255, cv2.THRESH_BINARY)

        # Result
        result = cv2.bitwise_and(img, img, mask=mask)

        if res_plot:
            # Final output
            plt.figure(figsize=(10, 10))

            plt.subplot(2, 3, 1)
            plt.axis('off')
            plt.title("Original Image")
            plt.imshow(img, cmap="gray")

            plt.subplot(2, 3, 2)
            plt.imshow(gray, cmap="gray")
            plt.axis('off')
            plt.title("GrayScale Image")

            plt.subplot(2, 3, 3)
            plt.imshow(thresh, cmap="gray")
            plt.axis('off')
            plt.title("Threshold Image")

            plt.subplot(2, 3, 4)
            plt.imshow(sharped_img, cmap="gray")
            plt.axis('off')
            plt.title("Sharped Image")

            plt.subplot(2, 3, 5)
            plt.imshow(mask, cmap="gray")
            plt.axis('off')
            plt.title("Generated Mask")

            plt.subplot(2, 3, 6)
            plt.imshow(result, cmap="gray")
            plt.axis('off')
            plt.title("Result Image")

            plt.show()

        return mask, result

    # Generate Point Cloud Data
    def pcd_write(self, img_rgb, img_depth, output_path, mask=None, format="ply", 
                  ply_plot=False, depth_scale=0.1, verbose=False):

        masked_depth = cv2.bitwise_and(img_depth, img_depth, mask=mask)
        
        o3d_rgb = o3d.geometry.Image(img_rgb)
        o3d_depth = o3d.geometry.Image(masked_depth)

        # Get rgbd data with depth scale to be 0.1
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_depth, convert_rgb_to_intensity=False, depth_scale=depth_scale)

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.intrinsics)

        assert format in ["ply", "txt"]

        if format == "ply":    # Write ply in the ascii format
            o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=True)
        else:                  # Write txt in the ascii format
            xyz = np.asarray(point_cloud.points)
            rgb = np.asarray(point_cloud.colors) * 255
            pcd = np.hstack([xyz, rgb])
            np.savetxt(output_path, pcd, fmt='%.6f')
        
        if verbose:
            print(f"Saving {output_path} successfully!")
            
        # Visualization
        if ply_plot:
            o3d.visualization.draw_geometries([point_cloud])


    # Data Augmentation
    @staticmethod
    def data_augment(img_data) -> List: 
        img_flip01 = cv2.flip(img_data, -1)
        img_flip0 = cv2.flip(img_data, 0)
        img_flip1 = cv2.flip(img_data, 1)
        return [img_data, img_flip0, img_flip1, img_flip01]
    
    
    # Generate Mask by Color (Semantic)
    def mask_gen_semantic(self, img_rgb, rgb_threshold, mode=1):
        lower, upper = None, None
        rgb_th_np = np.array((rgb_threshold), dtype=np.uint8)

        if mode == 1: # single threshold mode
            assert len(rgb_th_np.shape) == 1
            lower = rgb_th_np
            upper = lower
        elif mode == 2: # lower and upper boundary mode
            assert len(rgb_th_np.shape) == 2
            lower, upper = rgb_th_np
        else:
            print(f"mode error, exit!")
            return None

        mask = cv2.inRange(img_rgb, lower, upper)
        ratio = cv2.countNonZero(mask) / (img_rgb.size / 3)
        print('pixel percentage:', np.round(ratio * 100, 2))
        return mask
    
    # Generate Mask by Color (Semantic Instance)
    def mask_gen_instance(self, img_rgb, rgb_threshold, verbose=False):
        assert type(rgb_threshold) == list and type(rgb_threshold[0]) == tuple
        masks = []
        lower, upper = None, None

        for color_th in rgb_threshold:
            lower = np.array((color_th), dtype=np.uint8)
            upper = lower        
            mask = cv2.inRange(img_rgb, lower, upper)
            ratio = cv2.countNonZero(mask) / (img_rgb.size / 3)
            if verbose:
                print(f'pixel percentage for {color_th}:', np.round(ratio * 100, 2))
            masks.append(mask)

        return masks

    # Filter the noise inside the depth image
    def noise_filter(self, depth_img: np.ndarray, filt_dict=noise_filter_dict) -> np.ndarray:
        assert len(depth_img.shape) == 2
        filt_mode = filt_dict['filt_mode']
        threshold = filt_dict['threshold']
        plot = filt_dict['plot']
        plot_bins = filt_dict['plot_bins']

        # serve as bilateral filter
        if filt_mode == 'depth_value':
            assert len(threshold) == 2
            min_th, max_th = threshold
            assert min_th <= max_th

            new_depth = np.copy(depth_img)
            new_depth[depth_img < min_th] = 0
            new_depth[depth_img > max_th] = 0
        
        # remove all noise with frequency lower than a value
        elif filt_mode == 'frequency':
            assert type(threshold) == int
            pass

        else:
            print("No noise filter mode indicated!")
            raise RuntimeError

        if plot:
            assert len(plot_bins) == 2
            low, high = plot_bins

            origin = [i for i in depth_img.flatten() if i > 0]
            result = [i for i in new_depth.flatten() if i > 0]
            origin = np.asarray(origin)
            result = np.asarray(result)
            
            hist_ori = cv2.calcHist([origin],[0],None,[high],[low, high], False)
            hist = cv2.calcHist([result],[0],None,[high],[low, high], False)
           
            plt.rcParams.update({'font.size': 8})
            fig = plt.figure(figsize=(10, 8))
            #fig.suptitle('Multiple Graphs', fontsize=16)
            

            # Plot origin statistics
            plt.subplot(2, 2, 1)
            plt.ylabel('Frequency')
            plt.title('Origin Depth Curve Chart')
            plt.plot(hist_ori)

            plt.subplot(2, 2, 2)
            plt.hist(origin, bins=range(low, high, 1))
            plt.title('Origin Depth Histogram')

            # Plot filtered statistics
            plt.subplot(2, 2, 3)
            plt.xlabel('Depth Image Value')
            plt.ylabel('Frequency')
            plt.title('Filtered Depth Curve Chart')
            plt.plot(hist)

            plt.subplot(2, 2, 4)
            plt.hist(result, bins=range(low, high, 1))
            plt.xlabel('Depth Image Value')
            plt.title('Filtered Depth Histogram')
            
            # Plot show
            plt.tight_layout()
            plt.show()

        return new_depth

# For testing
def test():
    print("testing")
    # rgb_img = cv2.imread("/mnt/c/Users/Charlescai/Desktop/test-image.roi.color.png")
    # depth_img = cv2.imread("/mnt/c/Users/Charlescai/Desktop/test-image.roi.depth.png")
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)      # From BGR to RGB
    # depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    # depth_img = PCDGenerator().noise_filter(depth_img, [45, 48], plot=True)
    fashion_data_path = "/mnt/c/Users/Charlescai/Desktop/fashion_test/fashion_raw"
    folders = [f.path for f in os.scandir(fashion_data_path) if f.is_dir()]
    output_path = "/mnt/c/Users/Charlescai/Desktop/fashion_test/fashion_ply"
    check_path(output_path)

    for i in range(len(folders)):
        rgb = f"{folders[i]}/0/6CD146030E49/test-image.roi.color.png"
        depth = f"{folders[i]}/0/6CD146030E49/test-image.roi.depth.png"
        out_path = f"{output_path}/item_filtered{i}.ply"
        rgb_data = cv2.imread(rgb)
        depth_data = cv2.imread(depth)
        depth_data = cv2.cvtColor(depth_data, cv2.COLOR_BGR2GRAY)     # From 3 channels to 1 channel
        depth_data = PCDGenerator().noise_filter(depth_data)
        PCDGenerator().pcd_write(rgb_data, depth_data, out_path)
        print(f"saving {out_path} successfully!")
    
if __name__ == "__main__":
    test()
