"""
Base LLM Backend Interface
Defines common interface for all LLM backends (MLX, vLLM, Ollama)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class BackendConfig:
    """Configuration for LLM backends"""
    backend_type: str
    model_name: str
    host: str = "localhost"
    port: int = 11434
    timeout_seconds: int = 30
    max_retries: int = 3
    
    # Backend-specific parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    
    # MoE-specific parameters
    num_experts: int = 5
    expert_selection_method: str = "top_k"  # top_k, sampling, etc.
    
    # Advanced configuration
    batch_size: int = 1
    use_gpu: bool = True
    precision: str = "float16"  # float16, float32, int8
    
    # Custom parameters for each backend
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


class BaseLLMBackend(ABC):
    """
    Abstract base class for all LLM backends
    
    Provides common interface for:
    - MLX (Apple Silicon optimization)
    - vLLM (high-throughput inference)  
    - Ollama (local model serving)
    """
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self.is_initialized = False
        self.model = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the backend and load model
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_logits(
        self, 
        prompt: str, 
        num_experts: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate expert routing logits for a given prompt
        
        Args:
            prompt: Input text prompt
            num_experts: Number of experts (defaults to config.num_experts)
            
        Returns:
            Numpy array of logits for expert routing
        """
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        routing_weights: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate text using MoE routing
        
        Args:
            prompt: Input text prompt
            routing_weights: Optional pre-computed routing weights
            
        Returns:
            Generated text
        """
        pass
    
    async def estimate_ambiguity(self, prompt: str) -> float:
        """
        Estimate prompt ambiguity for routing control
        
        Default implementation based on prompt analysis.
        Backends can override with model-specific methods.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Ambiguity score between 0.0 and 1.0
        """
        if not prompt or len(prompt.strip()) == 0:
            return 0.5  # Neutral ambiguity for empty prompts
        
        # Simple heuristics for ambiguity estimation
        factors = []
        
        # Length factor: very short or very long prompts can be ambiguous
        length_factor = min(1.0, abs(len(prompt) - 50) / 100.0)
        factors.append(length_factor)
        
        # Question words indicate uncertainty
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        question_count = sum(1 for word in question_words if word in prompt.lower())
        question_factor = min(1.0, question_count / 3.0)
        factors.append(question_factor)
        
        # Uncertainty words
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain']
        uncertainty_count = sum(1 for word in uncertainty_words if word in prompt.lower())
        uncertainty_factor = min(1.0, uncertainty_count / 2.0)
        factors.append(uncertainty_factor)
        
        # Average the factors
        base_ambiguity = np.mean(factors)
        
        # Add some randomness to prevent deterministic behavior
        noise = np.random.normal(0, 0.05)
        ambiguity = np.clip(base_ambiguity + noise, 0.0, 1.0)
        
        return float(ambiguity)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health and performance
        
        Returns:
            Dictionary with health status and metrics
        """
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "Backend not initialized",
                    "backend_type": self.config.backend_type
                }
            
            # Basic connectivity test
            test_logits = await self.generate_logits("health check", 3)
            
            return {
                "status": "healthy",
                "backend_type": self.config.backend_type,
                "model_name": self.config.model_name,
                "num_experts": len(test_logits),
                "test_logits_shape": test_logits.shape,
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend_type": self.config.backend_type
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "backend_type": self.config.backend_type,
            "model_name": self.config.model_name,
            "num_experts": self.config.num_experts,
            "initialized": self.is_initialized,
            "config": self.config.__dict__
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.is_initialized = False
        self.model = None
    
    def __enter__(self):
        return self
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class BackendError(Exception):
    """Base exception for backend errors"""
    pass


class BackendInitializationError(BackendError):
    """Raised when backend fails to initialize"""
    pass


class BackendInferenceError(BackendError):
    """Raised when inference fails"""
    pass


class BackendTimeoutError(BackendError):
    """Raised when backend operation times out"""
    pass