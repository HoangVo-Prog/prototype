"""Frozen evaluated retriever adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional

import torch


@dataclass
class RetrieverAdapter:
    name: str
    model: torch.nn.Module
    device: torch.device
    has_grab: bool

    def eval(self) -> None:
        self.model.eval()

    def encode_text(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(batch.to(self.device))

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch.to(self.device))

    def encode_text_grab(self, batch: torch.Tensor) -> torch.Tensor:
        if not self.has_grab:
            raise RuntimeError(f"Retriever {self.name} has no GRAB branch")
        return self.model.encode_text_grab(batch.to(self.device))

    def encode_image_grab(self, batch: torch.Tensor) -> torch.Tensor:
        if not self.has_grab:
            raise RuntimeError(f"Retriever {self.name} has no GRAB branch")
        return self.model.encode_image_grab(batch.to(self.device))


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, logger: logging.Logger) -> None:
    from utils.checkpoint import load_state_dict

    logger.info("Loading retriever checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    state_dict = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"Checkpoint does not contain a model state dict: {checkpoint_path}")
    load_state_dict(model, state_dict)


def model_has_grab(model: torch.nn.Module, repo_args: SimpleNamespace) -> bool:
    return (
        not bool(getattr(repo_args, "only_global", False))
        and hasattr(model, "visul_emb_layer")
        and hasattr(model, "texual_emb_layer")
    )


def load_retriever(
    retriever_name: str,
    repo_args: SimpleNamespace,
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> RetrieverAdapter:
    from model import build_model

    if retriever_name == "trained_clip":
        repo_args.only_global = True
        repo_args.return_all = False
        repo_args.modify_k = False
    model = build_model(repo_args, num_classes)
    load_checkpoint(model, checkpoint_path, logger)
    if device.type == "cpu":
        model = model.float()
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    has_grab = model_has_grab(model, repo_args)
    logger.info("Loaded retriever=%s has_grab=%s only_global=%s", retriever_name, has_grab, repo_args.only_global)
    return RetrieverAdapter(name=retriever_name, model=model, device=device, has_grab=has_grab)
