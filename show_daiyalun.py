from hydra import core, initialize, compose
from omegaconf import OmegaConf

import torch
import numpy as np

from cross_view_transformer.common import setup_data_module

from pathlib import Path

from cross_view_transformer.common import load_backbone

import torch
import time
import imageio
import ipywidgets as widgets

from cross_view_transformer.visualizations.nuscenes_stitch_viz import NuScenesStitchViz

# *********************************************** #

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ hydra-core

# CHANGE ME
DATASET_DIR = '/home/daiyalun/slam_test/cross_view_transformers-master/datasets/nuscenes'
LABELS_DIR = '/home/daiyalun/slam_test/cross_view_transformers-master/datasets/cvt_labels_nuscenes'


# core.global_hydra.GlobalHydra.instance().clear()        # required for Hydra in notebooks

initialize(config_path='./config')

# Add additional command line overrides
cfg = compose(
    config_name='config',
    overrides=[
        'experiment.save_dir=../logs/',                 # required for Hydra in notebooks
        'data=nuscenes',
        f'data.dataset_dir={DATASET_DIR}',
        f'data.labels_dir={LABELS_DIR}',
        'data.version=v1.0-mini',
        'loader.batch_size=1',
    ]
)

# resolve config references
OmegaConf.resolve(cfg)

print(list(cfg.keys()))

# *********************************************** #
# SPLIT = 'val_qualitative_000'
SPLIT = 'val'
SUBSAMPLE = 10


data = setup_data_module(cfg)

dataset = data.get_split(SPLIT, loader=False)
dataset = torch.utils.data.ConcatDataset(dataset)
dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

print(len(dataset))

# *********************************************** #
# Download a pretrained model (13 Mb)
VEHICLE_MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_vehicles_50k.ckpt'
VEHICLE_CHECKPOINT_PATH = './logs/cvt_nuscenes_vehicles_50k.ckpt'

ROAD_MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_road_75k.ckpt'
ROAD_CHECKPOINT_PATH = './logs/cvt_nuscenes_road_75k.ckpt'

# !mkdir -p $(dirname ${VEHICLE_CHECKPOINT_PATH})
# !wget $VEHICLE_MODEL_URL -O $VEHICLE_CHECKPOINT_PATH
# !wget $ROAD_MODEL_URL -O $ROAD_CHECKPOINT_PATH


vehicle_network = load_backbone(VEHICLE_CHECKPOINT_PATH)
road_network = load_backbone(ROAD_CHECKPOINT_PATH)

# *********************************************** #
GIF_PATH = './predictions.gif'

# Show more confident predictions, note that if show_images is True, GIF quality with be degraded.
viz = NuScenesStitchViz(vehicle_threshold=0.6, road_threshold=0.6, show_images=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vehicle_network.to(device)
vehicle_network.eval()

road_network.to(device)
road_network.eval()

images = list()

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        vehicle_pred = vehicle_network(batch)['bev']
        road_pred = road_network(batch)['bev']

        visualization = np.vstack(viz(batch, road_pred, vehicle_pred))

        images.append(visualization)


# Save a gif
duration = [0.5 for _ in images[:-1]] + [2 for _ in images[-1:]]
imageio.mimsave(GIF_PATH, images, duration=duration)