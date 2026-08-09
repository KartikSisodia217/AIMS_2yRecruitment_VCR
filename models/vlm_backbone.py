"""
VLM Backbone abstractions and implementations.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List, Tuple

class VLMBackbone(ABC):
    """Abstract interface for Vision-Language Model backends.
    
    The rest of the codebase interacts with the VLM only through this interface.
    This allows swapping Qwen2.5-VL for another VLM without changing downstream code.
    """
    
    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Return the hidden dimension of the model's representations."""
        ...
    
    @abstractmethod
    def encode(self, image: Any, text: str) -> torch.Tensor:
        """Encode an image+text pair and return a hidden representation.
        
        Args:
            image: PIL Image or tensor
            text: Input text (question + candidate, etc.)
        
        Returns:
            Tensor of shape [hidden_dim] - a single vector representation
        
        NOTE: The exact representation strategy (last token, mean pool, etc.)
        is a RESEARCH DECISION that should be configurable.
        # RESEARCH_DECISION: representation extraction strategy
        """
        ...
    
    @abstractmethod  
    def encode_text_only(self, text: str) -> torch.Tensor:
        """Encode text without image (for blind branch).
        
        Returns:
            Tensor of shape [hidden_dim]
        """
        ...
    
    @abstractmethod
    def compute_log_likelihood(self, image: Any, prompt: str, completion: str) -> float:
        """Compute log-likelihood of completion given image+prompt.
        
        Used for answer scoring via generative likelihood.
        
        Args:
            image: PIL Image or None (for blind scoring)
            prompt: The prompt text (e.g., question)
            completion: The candidate text to score
        
        Returns:
            Log-likelihood score (higher = more likely)
        """
        ...
    
    @abstractmethod
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of trainable parameters."""
        ...
    
    @abstractmethod
    def freeze(self) -> None:
        """Freeze all parameters."""
        ...
    
    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        ...
    
    @abstractmethod
    def get_total_params(self) -> int: ...
    
    @abstractmethod
    def get_trainable_params(self) -> int: ...
    
    @property
    @abstractmethod
    def device(self) -> torch.device: ...
    
    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...


class MockVLMBackbone(VLMBackbone):
    """Mock VLM that returns deterministic fake representations.
    
    Used for testing the full pipeline without loading a real VLM.
    All outputs are deterministic given the same inputs.
    """
    
    def __init__(self, hidden_dim: int = 2048, seed: int = 42):
        self.hidden_dim_value = hidden_dim
        self._seed = seed
        self._device = torch.device('cpu')
        self._dtype = torch.float32
        # Create a small trainable linear layer so we can test parameter counting
        self._dummy_param = nn.Linear(hidden_dim, hidden_dim)
    
    def get_hidden_dim(self) -> int:
        return self.hidden_dim_value
    
    def encode(self, image, text) -> torch.Tensor:
        # Use hash of text to produce deterministic but varied representations
        # Return tensor of shape [hidden_dim]
        seed = hash(str(text)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(self.hidden_dim_value, generator=gen, dtype=self._dtype)
    
    def encode_text_only(self, text) -> torch.Tensor:
        # Similar but different from encode (no image contribution)
        seed = hash('blind_' + str(text)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(self.hidden_dim_value, generator=gen, dtype=self._dtype)
    
    def compute_log_likelihood(self, image, prompt, completion) -> float:
        # Return deterministic score based on text content
        seed = hash(str(prompt) + str(completion)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(1, generator=gen).item()
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return list(self._dummy_param.parameters())
    
    def freeze(self) -> None:
        for p in self.get_trainable_parameters():
            p.requires_grad = False
            
    def unfreeze(self) -> None:
        for p in self.get_trainable_parameters():
            p.requires_grad = True
            
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self._dummy_param.parameters())
        
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


class Qwen25VLBackbone(VLMBackbone):
    """Qwen2.5-VL backbone implementation.
    
    NOT YET FUNCTIONAL — requires:
    - Model weights downloaded
    - qwen-vl-utils installed
    - Sufficient RAM/VRAM
    
    This is a placeholder showing the intended API usage.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", 
                 dtype: str = "float32", device: str = "cpu",
                 max_pixels: int = 602112, min_pixels: int = 200704,
                 attn_implementation: str = "sdpa",
                 representation_strategy: str = "last_token"):
        # RESEARCH_DECISION: representation_strategy options:
        # - "last_token": use the last token's hidden state
        # - "mean_pool": mean pool over all tokens
        # - "cls_token": use a specific CLS-like token if available
        # - "eos_token": use the EOS token's hidden state
        self._model = None
        self._processor = None
        self._model_name = model_name
        self._dtype_str = dtype
        self._device_str = device
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.attn_implementation = attn_implementation
        self.representation_strategy = representation_strategy
    
    def load(self):
        """Actually load the model and processor. Called explicitly."""
        raise NotImplementedError(
            "Qwen2.5-VL loading not yet implemented. "
            "Use MockVLMBackbone for testing. "
            "See ARCHITECTURE.md for implementation plan."
        )

    def get_hidden_dim(self) -> int:
        raise NotImplementedError()
        
    def encode(self, image: Any, text: str) -> torch.Tensor:
        raise NotImplementedError()
        
    def encode_text_only(self, text: str) -> torch.Tensor:
        raise NotImplementedError()
        
    def compute_log_likelihood(self, image: Any, prompt: str, completion: str) -> float:
        raise NotImplementedError()
        
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        raise NotImplementedError()
        
    def freeze(self) -> None:
        raise NotImplementedError()
        
    def unfreeze(self) -> None:
        raise NotImplementedError()
        
    def get_total_params(self) -> int:
        raise NotImplementedError()
        
    def get_trainable_params(self) -> int:
        raise NotImplementedError()
        
    @property
    def device(self) -> torch.device:
        raise NotImplementedError()
        
    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError()
