"""

    Introduction: "create.py" is use for inference of single input depth and rgb image
    Input: RGB and Depth image (data_folder)
    Output: ".npy" file of scene in "./data/processed/abbpcd_inference" folder

    Note: Change "/configuration/dataset_yaml/abbpcd_inference.yaml" for different dataset label.

"""

import open3d as o3d
import numpy as np
import glob
import os
import yaml
from pathlib import Path
import cv2
import shutil
import time

# This four variables needs to be change for different dataset
dataset_yaml = "./dataset/configuration/dataset_yaml/abbpcd_inference.yaml"
save_dir = "./data/processed/abbpcd_inference"

# Path Prefix (changes for different camera)
prefix = "/0/6CD146030E3C/"
rgb_name = "test-image.roi.color.png"                # RGB Name
depth_name = "test-image.roi.depth.png"              # Depth Name

ply_plot = False                                  # Visualization
# intrinsic_mat = [609.213196, 0, 313.040894, 0, 608.660034, 241.966293, 0, 0, 1]
intrinsic_mat = [612.815308, 0, 312.01944, 0, 612.397461, 227.341553, 0, 0, 1]
fx, _, cx, _, fy, cy, _, _, _ = intrinsic_mat
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy)


def create_label(dataset_yaml, save_dir='/tmp'):

    if not os.path.exists(save_dir):
        print(f"{save_dir} not exists, creating...")
        os.makedirs(save_dir)

    with open(dataset_yaml, 'r') as file:
        dataset_info = yaml.safe_load(file)

    class_map = dataset_info['class_map']
    color_map = dataset_info['color_map']

    label_database = dict()
    for class_name, class_id in class_map.items():
        label_database[class_id] = {
            'color': color_map[class_id],
            'name': class_name,
            'validation': False
        }

    _save_yaml(f"{save_dir}/label_database.yaml", label_database)
    print(f"label yaml created as {save_dir}/label_database.yaml")


def noise_filter(depth_image, threshold=[32, 50]):

    image = cv2.imread(depth_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to eliminate above the bin height
    min_th, max_th = threshold
    gray[gray < min_th] = 0
    gray[gray > max_th] = 0

    return gray


def find_files(start_dir, file_pattern):
    res = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(file_pattern):
                res.append(os.path.join(root, file))
    return res


def create_npy(path, save_dir="/tmp"):

    rgb_path = find_files(path, "test-image.roi.color.png")[0]
    depth_path = find_files(path, "test-image.roi.depth.png")[0]

    if not os.path.exists(save_dir):
        print(f"{save_dir} not exists, creating...")
        os.makedirs(save_dir)

    filt_depth = noise_filter(depth_path)

    o3d_rgb = o3d.io.read_image(rgb_path)
    o3d_depth = o3d.geometry.Image(filt_depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=0.1, convert_rgb_to_intensity=False)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics)

    # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.008)

    # Visualize ply file
    ply_plot = False
    if ply_plot:
        # pcd = o3d.io.write_point_cloud(f"{label}{cnt}.ply", point_cloud, write_ascii=True)
        o3d.visualization.draw_geometries([point_cloud])

    xyz = np.asarray(point_cloud.points)
    rgb = np.asarray(point_cloud.colors) * 255
    xyz_rgb = np.hstack((xyz, rgb))

    ones = np.ones((xyz_rgb.shape[0], 6))  # Add six columns of all 1s to the array 
    data_with_ones = np.hstack((xyz_rgb, ones))
    points = data_with_ones

    filepath = f"{save_dir}/test/inference.npy"
    filebase = {
        "scene": filepath.split("/")[-1],
        "raw_filepath": str(filepath),
        "file_len": len(points),
    }

    filebase["raw_segmentation_filepath"] = ""

    # gt_data = (points[:, -2] + 1) * 1000 + \
    #     points[:, -1] + 1  # set ground truth data

    processed_filepath = Path(filepath)
    if not processed_filepath.parent.exists():
        processed_filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(processed_filepath, points.astype(np.float32))
    print(f"{processed_filepath} successfully saved")

    filebase["filepath"] = str(processed_filepath)
    filebase["instance_gt_filepath"] = ""
    filebase["color_mean"] = [
        float((points[:, 3] / 255).mean()),
        float((points[:, 4] / 255).mean()),
        float((points[:, 5] / 255).mean()),
    ]
    filebase["color_std"] = [
        float(((points[:, 3] / 255) ** 2).mean()),
        float(((points[:, 4] / 255) ** 2).mean()),
        float(((points[:, 5] / 255) ** 2).mean()),
    ]

    color_mean_std = {
        'mean': filebase['color_mean'],
        'std': filebase['color_std']
    }

    _save_yaml(f"{save_dir}/test_database.yaml", [filebase])
    _save_yaml(f"{save_dir}/color_mean_std.yaml", color_mean_std)


def _save_yaml(path, file):
    with open(path, "w") as f:
        yaml.safe_dump(file, f, default_style=None, default_flow_style=False)


def main():

    create_label(dataset_yaml, save_dir)
    cnt = 0
    # create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # connect the socket to the server
    sock.connect((HOST, PORT))

    max_try = 3

    try:
        while True:
            retry = 0
            start_time = time.time()
            file_exist = os.listdir(data_dir)

            if file_exist:
                files = os.listdir(data_dir)
                files.sort(key=lambda x: os.path.getmtime(
                    os.path.join(data_dir, x)))
                latest_file = os.path.join(data_dir, files[-1])
                x = os.path.split(latest_file)[-1]
                print("latest_file", x)
                # subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
                # subdirs.sort()
                # dir_list.extend(subdirs)
                # print(dir_list)
                # x = dir_list[0]

                rgb_image = data_folder + x + prefix + rgb_name
                print("rgb_image", rgb_image)
                depth_image = data_folder + x + prefix + depth_name
                print("depth_image", depth_image)
                depth_filtered = noise_filter(depth_image)
                # .txt file creation in /data/raw/
                txt_create(rgb_image, depth_filtered, x, save_dir, cnt)
                cnt += 1
                print("Process_file", x)
                shutil.rmtree(data_folder+x)
                end_time = time.time()
                # send a message to the server if the directory exists
                msg = f"Directory exists! - {file_exist}"
                try:
                    # time.sleep(4)
                    sock.sendall(msg.encode())
                    print(f"{file_exist}")
                except BrokenPipeError as e:
                    print("Broken pipe error", e)
                    retry += 1
                    if retry == max_try:
                        print("maximum try reached")
                        raise
                    else:
                        # time.sleep(0.1)
                        sock = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect((HOST, PORT))

            else:
                print("Directory does not exist.")

            # wait for some time before checking again
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting program")
        sock.close()


if __name__ == "__main__":
    main()
