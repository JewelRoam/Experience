import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Any

from experience.symbolic_tensor.module.symbolic_transform import SymbolicTransformModule
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


EXPERIENCE_DIR = os.path.join(os.path.dirname(__file__), "experience")


class NaiveModel(nn.Module):
    """
    Naive symbolic transform model.

    Wraps a single SymbolicTransformModule layer.
    Experience is loaded from the experience/ directory if available,
    otherwise initialized with seed examples.

    Args:
        forward_prompt: The task prompt for the symbolic transform.
        topk: Number of experience entries to select.
    """

    def __init__(self, forward_prompt: str = "", topk: int = 16):
        super().__init__()
        self.transform = SymbolicTransformModule(
            experience_shape=[1, 3],
            forward_prompt=forward_prompt,
            topk=topk,
        )

    def load_experience(self, experience: torch.Tensor):
        """Load experience tensor into the transform layer."""
        self.transform.experience = experience
        self.transform.experience.requires_grad_(True)

    def parameters(self, recurse: bool = True):
        yield from self.transform.parameters(recurse)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return self.transform(input)
