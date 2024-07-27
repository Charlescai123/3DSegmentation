"""

    Introduction: "create_custom_dataset.py" is use for inference on custom dataset
    Input: Convert rgb and depth image 
    Output: The .txt file get saved in "/data/raw/dataset_name" folder

    Note:   1. Change "/configuration/dataset_yaml/abbpcd_inference_custom.yaml" for different dataset label.
            2. "noise_filter": Change thresholding for different bin and cell (min_th, max_th)
            3. Change prefix and intrinisc parameters for different camera
            4. In "scripts/abbpcd/abbpcd_inference_custom_dataset.sh" change the checkpoint.

"""

import open3d as o3d
import numpy as np
import glob
import os
import cv2

# This three variables needs to be change for different dataset
dataset_name = "abbpcd_inference_custom"  # Dataset name
data_folder = "./inference/custom_data/"  # Custome dataset folder
label_name = "custom_test"

input_dirs = sorted(glob.iglob(data_folder + "/*"))  # Input Dirs
prefix = "/0/6CD146030E37/"  # Path Prefix (changes for different camera)
rgb_name = "test-image.roi.color.png"  # RGB Name
depth_name = "test-image.roi.depth.png"  # Depth Name

result_dir = f"./data/raw/{dataset_name}"  # Result Dir
test_dir = result_dir + "/test"  # test directory
train_dir = result_dir + "/train"  # train directory

ply_plot = "false"  # Visualization

data_dir = f"{data_folder}"
dir_list = []

subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
subdirs.sort()
dir_list.extend(subdirs)


def txt_create(rgb_image, depth_filtered, label, test_dir, cnt):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image(rgb_image),
                                                              o3d.geometry.Image(depth_filtered), depth_scale=0.1,
                                                              convert_rgb_to_intensity=False)

    # Change the intrinsic parameters for different camera here
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=612.815308, fy=612.397461, cx=312.01944,
                                                   cy=227.341553)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    # In case of visualize ply file
    # if ply_plot:
    #     o3d.io.write_point_cloud(f"{label}{cnt}.ply", point_cloud, write_ascii=True)

    coordinate = np.asarray(point_cloud.points)
    color = np.asarray(point_cloud.colors)
    color = color * 255.0
    coordinate_color = np.hstack((coordinate, color))

    if not os.path.exists(f"{test_dir}/{label}{cnt}"):
        print(f"{test_dir}/{label}{cnt} not exists, creating...")
        try:
            os.makedirs(f"{test_dir}/{label}{cnt}")
            os.makedirs(f"{test_dir}/{label}{cnt}/Annotations")
        except FileExistsError:
            pass

    np.savetxt(f"{test_dir}/{label}{cnt}/{label}{cnt}.txt", coordinate_color, delimiter=' ', fmt='%.6f')
    np.savetxt(f"{test_dir}/{label}{cnt}/Annotations/{label}{cnt}.txt", coordinate_color, delimiter=' ', fmt='%.6f')


def noise_filter(depth_image):
    image = cv2.imread(depth_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_depth = np.copy(gray)

    # Thresholding to eliminate above the bin height
    min_th, max_th = 32, 60
    new_depth[gray < min_th] = 0
    new_depth[gray > max_th] = 0

    return new_depth


def main():
    cnt = 1

    if not os.path.exists(result_dir):
        print(f"{result_dir} not exists, creating...")
        os.makedirs(result_dir)

    if not os.path.exists(test_dir):
        print(f"{test_dir} not exists, creating...")
        os.makedirs(test_dir)

    for x in dir_list:
        rgb_image = data_folder + x + prefix + rgb_name
        depth_image = data_folder + x + prefix + depth_name

        depth_filtered = noise_filter(depth_image)
        # .txt file creation in /data/raw/
        txt_create(rgb_image, depth_filtered, label_name, test_dir, cnt)
        cnt += 1


if __name__ == "__main__":
    main()
