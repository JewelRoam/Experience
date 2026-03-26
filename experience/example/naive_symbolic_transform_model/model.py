import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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
        output_prompt: Callable that builds the forward prompt. None uses default.
        query_prompt: Callable that builds the query keyword prompt. None uses default.
        grad_input_prompt: Callable that builds the grad_input prompt. None uses default.
        grad_exp_key_prompt: Callable that builds the experience key gradient prompt. None uses default.
        grad_exp_value_prompt: Callable that builds the experience value gradient prompt. None uses default.
        task_prompt: High-level task description (e.g. "Translate Python To Viba").
        topk: Number of experience entries to select.
        llm_env: Environment variable dict for LLM client. None uses os.environ defaults.
    """

    def __init__(
        self,
        output_prompt: Optional[Callable[..., str]] = None,
        query_prompt: Optional[Callable[..., str]] = None,
        grad_input_prompt: Optional[Callable[..., str]] = None,
        grad_exp_key_prompt: Optional[Callable[..., str]] = None,
        grad_exp_value_prompt: Optional[Callable[..., str]] = None,
        task_prompt: str = "",
        topk: int = 16,
        llm_env: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.transform = SymbolicTransformModule(
            experience_shape=[1, 3],
            output_prompt=output_prompt,
            query_prompt=query_prompt,
            grad_input_prompt=grad_input_prompt,
            grad_exp_key_prompt=grad_exp_key_prompt,
            grad_exp_value_prompt=grad_exp_value_prompt,
            task_prompt=task_prompt,
            topk=topk,
            llm_env=llm_env,
        )

    def load_experience(self, experience: torch.Tensor):
        """Load experience tensor into the transform layer."""
        self.transform.experience = experience
        self.transform.experience.requires_grad_(True)

    def parameters(self, recurse: bool = True):
        yield from self.transform.parameters(recurse)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return self.transform(input)
