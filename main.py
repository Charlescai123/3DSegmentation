import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything
import warnings
import socket
import json
from inference.create import *


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        print("param gpu_devices is None, getting variable from environment")
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        # cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="base_config.yaml")
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(config_path="conf", config_name="base_config.yaml")
def test(cfg: DictConfig):

    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )
    runner.test(model)

@hydra.main(config_path="conf", config_name="base_config.yaml")
def inference(cfg: DictConfig):
    HOST = cfg['general']['host']
    PORT = cfg['general']['port']

    print("Start inference process...")
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )
    print("Model loading completed, start socket program")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print(f"Socket listening on address: {HOST}:{PORT}")
        s.listen()
        try:
            while True:
                conn, addr = s.accept()
                with conn:
                    print(f"Socket connected to {addr}")
                    payload = conn.recv(4096)
                    if payload:
                        payload = payload.decode("utf-8").split('\r\n\r\n')[-1]
                        print("received from client: \n{}".format(payload))
                        try:
                            payload = json.loads(payload)
                            assert "path" in payload
                        except:
                            print("cannot parse the json content...")
                            continue
                        
                        # Do inference
                        path = payload['path']
                        create_label(cfg['general']['dataset_yaml'], save_dir="./data/processed/abbpcd_fashion")
                        create_npy(path, save_dir="./data/processed/abbpcd_fashion")
                        runner.test(model)
                        #shutil.rmtree()

                        # Send reply
                        resp = {"status": "success"}
                        resp_content = json.dumps(resp).encode("utf-8")
                        resp_header = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(resp_content)}"
                        response = (resp_header + "\r\n\r\n").encode("utf-8") + resp_content
                        conn.sendall(response)
                        print("model inference executed successfully")
                        print(f"Socket continue listening on address: {HOST}:{PORT}")

        except KeyboardInterrupt:
            print("Exiting program")
            conn.close()

@hydra.main(config_path="conf", config_name="base_config.yaml")
def main(cfg: DictConfig):
    if cfg['general']['inference']:
        inference(cfg)
    elif cfg['general']['train_mode']:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":

    # Filter all deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
