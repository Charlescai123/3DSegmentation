import os
import yaml
import numpy as np
import argparse
from natsort import natsorted

# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--dataset_yaml', type=str, required=True)

# Parse the arguments
args = parser.parse_args()
dataset_yaml_path = args.dataset_yaml
template_yaml_path = "templates/dataset_template.yaml"
conf_output_folder = "../../conf/data/datasets/"

def script_gen():
    pass

def config_gen(dataset_yaml=dataset_yaml_path, template_yaml=template_yaml_path,
               output=conf_output_folder):
    assert os.path.exists(dataset_yaml)
    assert os.path.exists(template_yaml)
    assert os.path.exists(output)

    with open(dataset_yaml, 'r') as f:
        dataset = yaml.safe_load(f)
        dataset_name = dataset['dataset_name']
    with open(template_yaml, 'r') as f:
        template = yaml.safe_load(f)

    for data_type, _ in template.items():
        template[data_type]['dataset_name'] = dataset_name
        template[data_type]['data_dir'] += dataset_name
        template[data_type]['dataset_yaml'] += f"{dataset_name}.yaml"
        template[data_type]['label_db_filepath'] += f"{dataset_name}/label_database.yaml"
        template[data_type]['color_mean_std'] += f"{dataset_name}/color_mean_std.yaml"

    output_file = output + f"{dataset_name}.yaml"

    with open(output_file, 'w') as f:
        f.write("# @package data\n")    # Important Annotations to add!
        yaml.dump(template, f)

    print(f"{output_file} generated successfully")


def test():
    pass


if __name__ == '__main__':
    config_gen()
