"""A script to export the inference model."""

import os

import torch
from omegaconf import OmegaConf

from .models.first_stage.vqgan import VQModelIF, VQModelInference
from .models.second_stage.fm_model import ModelIF, ModelInference
from .util import instantiate_from_config


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

# use torch summary to verify the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference = inference.to(device)

x = torch.randn(1, 2, 3, 288, 512).to(device)
f = torch.ones (1).to(device) * 20.0

with torch.no_grad():
    output, features = inference(x, f)

print("Inference model output shape:", output.shape)
print("Inference model features count:", len(features))
print("Inference model features shape:", features[0].shape)

# save checkpoint
inference_model_folder = os.path.expandvars("$WM_WORK_DIR/orbis_288x512_encoding")
save_path = os.path.join(inference_model_folder, "checkpoints/last.ckpt")

torch.save({"state_dict": inference.state_dict()}, save_path)
print(f"Saved inference model checkpoint to {save_path}")
