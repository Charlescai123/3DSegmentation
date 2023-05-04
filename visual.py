#visualization.py

import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import argparse
from datasets.scannet200.scannet200_constants import VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_20
from datasets.scannet200.scannet200_constants import VALID_CLASS_IDS_200, CLASS_LABELS_200, SCANNET_COLOR_MAP_200
import random


def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--ply_path', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--ext', action="store_true")

    # Parse the arguments
    args = parser.parse_args()
    ply_path = args.ply_path
    mask_dir = args.mask_dir
    scene_name = args.scene_name

    # Load ply
    scene = o3d.io.read_point_cloud(ply_path)
    scene_mask = o3d.io.read_point_cloud(ply_path)

    # Read txt for the scene
    with open(mask_dir + "/" + scene_name + ".txt") as f:
        lines = f.readlines()

    # Split the lines into file, instance and score and get the label
    inst=[]

    for l in lines:
        file, inst_i, score = l.split()

        if float(score) < 0.8:
            #print("Score too low, skipping iteration\n")
            continue

        # Create array of instances and get label
        inst.append(inst_i)
        try:
            label = CLASS_LABELS_20[VALID_CLASS_IDS_20.index(int(inst_i))+2]
            print(label)
        except:
            print("Skipped " + inst_i)
            continue

        # Read the mask from the file
        with open(mask_dir + "/" + file) as f:
            mask = list(map(bool, (map(int, f.readlines()))))

        # if mask.count(1) < 100:
        #     continue

        # Apply the mask with a color
        colors = []
        inst_color = list(SCANNET_COLOR_MAP_20[VALID_CLASS_IDS_20[CLASS_LABELS_20.index(label)]])
        #inst_color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        for i in range(len(scene_mask.points)):
            if mask[i]:
                colors.append([inst_color[0]/255., inst_color[1]/255., inst_color[2]/255.])
                #colors.append(inst_color)

            else:
                colors.append(scene_mask.colors[i])

        scene_mask.colors = o3d.utility.Vector3dVector(colors)

    # Visualize scene
    if args.ext:
        ev = o3d.visualization.ExternalVisualizer()
        ev.set(scene_mask)
    else:
        ev = o3d.visualization.Visualizer()
        ev.create_window()
        ev.add_geometry(scene_mask)
        ev.run()
        ev.destroy_window()


if __name__ == "__main__":
    main()