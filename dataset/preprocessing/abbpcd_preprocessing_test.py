import re
import os
import numpy as np
import yaml
from fire import Fire
from natsort import natsorted
from loguru import logger
from pathlib import Path
from base_preprocessing_test import BasePreprocessingTest


class ABBPCDPreprocessing(BasePreprocessingTest):
    def __init__(
            self,
            data_dir: str = "./data/raw/abbpcd_inference_custom",
            save_dir: str = "./data/processed/abbpcd_inference_custom",
            modes: tuple = ("test",),
            dataset_yaml: str = "./dataset/configuration/dataset_yaml/abbpcd_inference_custom.yaml",
            n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        with open(dataset_yaml, 'r') as file:
            dataset_info = yaml.safe_load(file)

        self.class_map = dataset_info['class_map']
        self.color_map = dataset_info['color_map']

        self.create_label_database()
        # add file path for different modes (train/validation/test)
        for mode in self.modes:
            filepaths = []
            assert os.path.exists(f"{self.data_dir}/{mode}")
            folder_paths = [f.path for f in os.scandir(
                self.data_dir / mode) if f.is_dir()]
            self.files[mode] = natsorted(folder_paths)

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                'color': self.color_map[class_id],
                'name': class_name,
                'validation': False
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def _buf_count_newlines_gen(self, fname):
        def _make_gen(reader):
            while True:
                b = reader(2 ** 16)
                if not b:
                    break
                yield b

        with open(fname, "rb") as f:
            count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
        return count

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """

        filebase = {
            "filepath": filepath,
            "scene": filepath.split("/")[-1],
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        # print(f"filepath: {filepath}")
        scene_name = filepath.split("/")[-1]
        instance_counter = 0
        cluster_points = []

        # Created 6 dummy columns with value 1 to meet the size requirement of preprocessing script
        for instance in [f for f in os.scandir(self.data_dir / mode / scene_name / "Annotations")
                         if f.name.endswith(".txt")]:
                            data = np.loadtxt(instance)
                            ones = np.ones((data.shape[0], 6))                       # Add six columns of all 1s to the array 
                            data_with_ones = np.hstack((data, ones))
                            points = data_with_ones

        # To confirm point cloud data size match
        pcd_size = self._buf_count_newlines_gen(f"{filepath}/{scene_name}.txt")
        if points.shape[0] != pcd_size:
            print(f"FILE SIZE DOES NOT MATCH FOR {filepath}/{scene_name}.txt")
            print(f"({points.shape[0]} vs. {pcd_size})")

        filebase["raw_segmentation_filepath"] = ""

        gt_data = (points[:, -2] + 1) * 1000 + \
            points[:, -1] + 1  # set ground truth data

        file_len = len(points)
        filebase["file_len"] = file_len

        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = self.save_dir / \
            "instance_gt" / mode / f"{scene_name}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        # Calc MEAN and STD of point cloud data (RGB)
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
        return filebase

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/abbpcd/train_database.yaml"
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    @logger.catch
    def fix_bugs_in_labels(self):
        pass

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    Fire(ABBPCDPreprocessing)
