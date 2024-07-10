"""Class for holding model conditioning.

Incorporates different conditioning available to the model, such
as timesteps, class labels, image embeddings, text embeddings, etc.

Possible conditioning signals include:

classes
timestep
timestep_embedding
text_tokens
text_embeddings
image_imbeddings
"""

from abc import abstractmethod
import torch
from transformers import T5Tokenizer
from typing import Dict, List


class ContextAdapter(torch.nn.Module):
    """Basic block which accepts a context conditioning."""

    @abstractmethod
    def forward(self, context: Dict):
        """Apply the module to `x` given `context` conditioning."""


class NullContextAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, context: Dict):
        return None


class IgnoreContextAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, context: Dict, *args, **kwargs):
        return context


class IgnoreInputPreprocessor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class UnconditionalTextPromptsAdapter(torch.nn.Module):
    def forward(self, context: Dict):
        new_context = context.copy()
        text_prompts = context["text_prompts"]
        new_context["text_prompts"] = [""] * len(text_prompts)
        return new_context


class TextEmbeddingsAdapter(torch.nn.Module):
    def __init__(self, swap_context_channels: bool = False, **kwargs):
        super().__init__()
        self._swap_context_channels = swap_context_channels

    def forward(self, context: Dict):
        return (
            context["text_embeddings"].permute(0, 2, 1)
            if self._swap_context_channels
            else context["text_embeddings"]
        )


class T5TextPromptsPreprocessor(torch.nn.Module):
    def __init__(self, max_length: int, model_name: str, **kwargs):
        super().__init__()
        self._max_length = max_length

        self._tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    def forward(self, context: Dict, device, **kwargs):
        if "text_prompts" in context:
            prompts = context["text_prompts"]
            with torch.no_grad():
                tokens_dict = self._tokenizer(
                    prompts,
                    max_length=self._max_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=False,
                    return_tensors="pt",
                )

            # Add the text tokens to the context
            assert "text_tokens" not in context
            text_inputs_on_device = {}
            for k, v in tokens_dict.items():
                text_inputs_on_device[k] = v.detach().to(device)
            context["text_tokens"] = text_inputs_on_device
        return context
