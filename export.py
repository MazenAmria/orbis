import os

import torch
from omegaconf import OmegaConf

from .models.second_stage.fm_model import ModelInference
from .util import instantiate_from_config


def create_model_for_inference(
    model_folder: str,
    ckpt_path: str = "checkpoints/last.ckpt",
) -> ModelInference:
    model_config = OmegaConf.load(os.path.join(model_folder, "config.yaml"))
    model = instantiate_from_config(model_config)
    checkpoint = torch.load(os.path.join(model_folder, ckpt_path), map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
