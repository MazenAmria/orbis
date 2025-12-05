"""A script to export the inference model."""

import os

import torch
from omegaconf import OmegaConf

from models.first_stage.vqgan import VQModelIF, VQModelInference
from models.second_stage.fm_model import ModelIF, ModelInference
from util import instantiate_from_config


def load_orbis() -> ModelIF:
    world_model_folder = os.path.expandvars("$WM_WORK_DIR/orbis_288x512")
    ckpt_path = "checkpoints/last.ckpt"

    world_model_config = OmegaConf.load(os.path.join(world_model_folder, "config.yaml"))
    model = instantiate_from_config(world_model_config.model)

    checkpoint = torch.load(os.path.join(world_model_folder, ckpt_path), map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    model.load_state_dict(checkpoint, strict=False)
    return model


def create_inference_model() -> ModelInference:
    inference_model_folder = os.path.expandvars("$WM_WORK_DIR/orbis_288x512_encoding")
    inference_model_config = OmegaConf.load(os.path.join(inference_model_folder, "config.yaml"))
    model = instantiate_from_config(inference_model_config)
    return model


inference = create_inference_model()
orbis = load_orbis()

orbis_ae: VQModelIF = orbis.ae
inference_ae: VQModelInference = inference.ae

inference_ae.encoder = orbis_ae.encoder
inference_ae.encoder2 = orbis_ae.encoder2
inference_ae.conv = orbis_ae.quant_conv
inference_ae.conv2 = orbis_ae.quant_conv2

inference.vit = orbis.ema_vit

# use torch jit to compile the model
x = torch.randn(1, 2, 3, 288, 512)
f = torch.ones (1) * 20.0

traced_model = torch.jit.trace(inference, (x, f))

# save checkpoint
inference_model_folder = os.path.expandvars("$WM_WORK_DIR/orbis_288x512_encoding")
save_path = os.path.join(inference_model_folder, "compiled/inference.pt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

traced_model.save(save_path)
print(f"Model saved to {save_path}. You can now move this file.")
