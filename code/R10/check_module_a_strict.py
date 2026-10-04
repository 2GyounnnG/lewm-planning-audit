"""Strict-load preflight for the native pinned Module A LeWM; no trajectories.

Run remotely with R3_ROOT=/workspace/r10/shared_data/r3 and PYTHONPATH
containing /workspace/r10/shared_data/r3 and /workspace/r10/swm_a.
This script has been syntax-checked locally, but has NOT run remotely.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from torch import nn

ROOT = Path(os.environ["R3_ROOT"])
SWM = Path("/workspace/r10/swm_a")
REVISION = "63988116d34cde56aea1240d5e58eb158ac67dc0"
WEIGHTS_SHA256 = "eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def build_model():
    revision = subprocess.check_output(
        ["git", "-C", str(SWM), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError("Module A source revision mismatch")

    import stable_worldmodel
    from stable_worldmodel.wm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
    # Only reuse the source-hash-checked ViT constructor, not the old JEPA.
    from r3.model import _vit_factory

    if not Path(stable_worldmodel.__file__).resolve().is_relative_to(SWM):
        raise RuntimeError("SWM import escaped the pinned Module A checkout")
    constructors = {
        "stable_pretraining.backbone.utils.vit_hf": _vit_factory(),
        "stable_worldmodel.wm.lewm.module.Predictor": Predictor,
        "stable_worldmodel.wm.lewm.module.Embedder": Embedder,
        "stable_worldmodel.wm.lewm.module.MLP": MLP,
        "torch.nn.BatchNorm1d": nn.BatchNorm1d,
    }

    def construct(spec):
        if spec["_target_"] not in constructors:
            raise ValueError("Unexpected constructor: " + spec["_target_"])
        args = {
            key: construct(value) if isinstance(value, dict) and "_target_" in value else value
            for key, value in spec.items() if not key.startswith("_")
        }
        factory = constructors[spec["_target_"]]
        return functools.partial(factory, **args) if spec.get("_partial_") else factory(**args)

    config_path = ROOT / "official/reacher/config.json"
    weights_path = ROOT / "official/reacher/weights.pt"
    if sha256(weights_path) != WEIGHTS_SHA256:
        raise RuntimeError("Official Reacher weight hash mismatch")
    config = json.loads(config_path.read_text())
    component_names = ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
    if config["_target_"] != "stable_worldmodel.wm.lewm.LeWM":
        raise ValueError("Unexpected model constructor")
    if set(config) != {"_target_", *component_names}:
        raise ValueError("Unexpected config fields")
    if config["encoder"].get("pretrained") is not False:
        raise ValueError("No remote backbone downloads are allowed")
    model = LeWM(**{key: construct(config[key]) for key in component_names}).float().eval()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Strict load did not match")
    if any(torch.is_floating_point(value) and not torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("Nonfinite checkpoint tensors")
    model.requires_grad_(False)
    receipt = {
        "status": "NATIVE_MODULE_A_STRICT_LOAD_PASS",
        "swm_revision": revision,
        "swm_file": stable_worldmodel.__file__,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "weights_sha256": sha256(weights_path),
        "config_sha256": sha256(config_path),
        "state_dict_keys": len(state),
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "trajectories_started": 0,
        "optimizer_updates": 0,
    }
    return model, receipt


if __name__ == "__main__":
    _, receipt = build_model()
    print(json.dumps(receipt, indent=2))
