"""
vLLM Backend Implementation  
High-throughput LLM backend using vLLM framework
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import httpx
import json

from .base import BaseLLMBackend, BackendConfig, BackendError, BackendInitializationError, BackendInferenceError

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. Install with: pip install vllm")


class VLLMBackend(BaseLLMBackend):
    """
    vLLM Backend for high-throughput inference
    
    Supports:
    - High-throughput batch processing
    - GPU acceleration with tensor parallelism
    - Continuous batching
    - Fast MoE routing logit generation
    - Multiple model formats (HuggingFace, etc.)
    """
    
    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.engine = None
        self.sampling_params = None
        self._expert_layer = None
        self.client = None  # For HTTP API mode
        
        # vLLM can run in different modes
        self.use_http_api = config.custom_params.get('use_http_api', False)
        self.api_base_url = config.custom_params.get('api_base_url', f"http://{config.host}:{config.port}")
    
    async def initialize(self) -> bool:
        """Initialize vLLM engine or connect to API"""
        try:
            logger.info(f"Initializing vLLM backend with model: {self.config.model_name}")
            
            if self.use_http_api:
                await self._initialize_http_client()
            else:
                await self._initialize_engine()
            
            # Initialize expert routing layer
            await self._initialize_expert_routing()
            
            self.is_initialized = True
            logger.info("vLLM backend initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM backend: {e}")
            raise BackendInitializationError(f"vLLM initialization failed: {e}")
    
    async def _initialize_http_client(self):
        """Initialize HTTP client for vLLM API"""
        self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        
        # Test connection
        try:
            response = await self.client.get(f"{self.api_base_url}/health")
            if response.status_code != 200:
                raise BackendInitializationError(f"vLLM API not healthy: {response.status_code}")
        except httpx.RequestError as e:
            raise BackendInitializationError(f"Cannot connect to vLLM API: {e}")
        
        logger.info(f"Connected to vLLM API at {self.api_base_url}")
    
    async def _initialize_engine(self):
        """Initialize vLLM engine directly"""
        if not VLLM_AVAILABLE:
            raise BackendInitializationError(
                "vLLM is not available. Install with: pip install vllm"
            )
        
        # Configure engine arguments
        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.custom_params.get('tensor_parallel_size', 1),
            gpu_memory_utilization=self.config.custom_params.get('gpu_memory_utilization', 0.9),
            max_model_len=self.config.custom_params.get('max_model_len', 4096),
            dtype=self.config.precision,
            trust_remote_code=self.config.custom_params.get('trust_remote_code', False),
            seed=42
        )
        
        # Create async engine
        # Note: In real implementation, would create actual vLLM engine
        # For demo purposes, we'll simulate
        self.engine = self._create_mock_vllm_engine()
        
        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            n=1,
            use_beam_search=False
        ) if VLLM_AVAILABLE else self._create_mock_sampling_params()
        
        logger.info("vLLM engine initialized")
    
    def _create_mock_vllm_engine(self):
        """Create mock vLLM engine for demonstration"""
        class MockVLLMEngine:
            def __init__(self):
                self.model_config = {
                    'hidden_size': 4096,
                    'vocab_size': 32000,
                    'num_layers': 32
                }
            
            async def generate(self, prompts, sampling_params):
                # Mock generation
                results = []
                for prompt in prompts:
                    result = MockRequestOutput(
                        request_id=f"mock_{hash(prompt) % 10000}",
                        prompt=prompt,
                        outputs=[MockCompletionOutput(
                            text=f"Generated response to: {prompt[:30]}...",
                            token_ids=list(range(10))
                        )]
                    )
                    results.append(result)
                return results
            
            async def get_model_hidden_states(self, prompts):
                # Mock hidden states for logit generation
                batch_size = len(prompts)
                hidden_size = self.model_config['hidden_size']
                return np.random.normal(0, 0.1, (batch_size, hidden_size))
        
        return MockVLLMEngine()
    
    def _create_mock_sampling_params(self):
        """Create mock sampling params when vLLM not available"""
        class MockSamplingParams:
            def __init__(self, **kwargs):
                self.temperature = kwargs.get('temperature', 0.7)
                self.top_p = kwargs.get('top_p', 0.9)
                self.max_tokens = kwargs.get('max_tokens', 512)
        
        return MockSamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens
        )
    
    async def _initialize_expert_routing(self):
        """Initialize expert routing layer for MoE"""
        # In real implementation, this would be a learned routing layer
        # For now, create simple projection matrices
        
        hidden_size = 4096  # Typical for larger models
        num_experts = self.config.num_experts
        
        # Expert routing network (would be learned)
        self._expert_layer = {
            'router_weights': np.random.normal(0, 0.02, (hidden_size, num_experts)),
            'router_bias': np.zeros(num_experts),
            'temperature': 1.0
        }
        
        logger.info(f"Initialized expert routing for {num_experts} experts")
    
    async def generate_logits(
        self, 
        prompt: str, 
        num_experts: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate expert routing logits using vLLM
        
        Args:
            prompt: Input text prompt
            num_experts: Number of experts (defaults to config.num_experts)
            
        Returns:
            Numpy array of expert routing logits
        """
        if not self.is_initialized:
            raise BackendInferenceError("vLLM backend not initialized")
        
        try:
            num_experts = num_experts or self.config.num_experts
            
            if self.use_http_api:
                return await self._generate_logits_http(prompt, num_experts)
            else:
                return await self._generate_logits_engine(prompt, num_experts)
                
        except Exception as e:
            logger.error(f"vLLM logit generation failed: {e}")
            raise BackendInferenceError(f"vLLM inference failed: {e}")
    
    async def _generate_logits_http(self, prompt: str, num_experts: int) -> np.ndarray:
        """Generate logits using vLLM HTTP API"""
        try:
            # In real implementation, would call vLLM's embeddings endpoint
            # For now, simulate HTTP call
            
            request_data = {
                "model": self.config.model_name,
                "input": prompt,
                "encoding_format": "float"
            }
            
            # Mock HTTP response (in real implementation, call actual API)
            # response = await self.client.post(f"{self.api_base_url}/v1/embeddings", json=request_data)
            
            # Simulate embedding response
            embedding_size = 4096
            mock_embedding = np.random.normal(0, 0.1, embedding_size)
            
            # Convert embedding to expert logits
            logits = self._embedding_to_logits(mock_embedding, num_experts)
            
            return logits
            
        except Exception as e:
            raise BackendInferenceError(f"vLLM HTTP API call failed: {e}")
    
    async def _generate_logits_engine(self, prompt: str, num_experts: int) -> np.ndarray:
        """Generate logits using vLLM engine directly"""
        try:
            # Get hidden states from model
            hidden_states = await self.engine.get_model_hidden_states([prompt])
            
            # Use first (and only) prompt's hidden state
            hidden_state = hidden_states[0]  # Shape: (hidden_size,)
            
            # Convert to expert logits using routing layer
            logits = self._embedding_to_logits(hidden_state, num_experts)
            
            return logits
            
        except Exception as e:
            raise BackendInferenceError(f"vLLM engine inference failed: {e}")
    
    def _embedding_to_logits(self, embedding: np.ndarray, num_experts: int) -> np.ndarray:
        """Convert model embedding to expert routing logits"""
        # Apply expert routing layer
        router_weights = self._expert_layer['router_weights'][:, :num_experts]
        router_bias = self._expert_layer['router_bias'][:num_experts]
        
        # Linear projection: embedding @ weights + bias
        logits = np.dot(embedding, router_weights) + router_bias
        
        # Add prompt-specific variation (similar to MLX backend)
        if hasattr(embedding, '__len__'):
            embedding_norm = np.linalg.norm(embedding)
            # Higher norm embeddings might prefer certain experts
            if embedding_norm > 1.0:
                logits[-1] += 0.2  # Prefer last expert for high-norm embeddings
            elif embedding_norm < 0.5:
                logits[0] += 0.2   # Prefer first expert for low-norm embeddings
        
        # Add controlled noise
        noise_scale = 0.05
        logits += np.random.normal(0, noise_scale, size=logits.shape)
        
        logger.debug(f"Generated vLLM logits: {logits}")
        return logits
    
    async def generate_text(
        self, 
        prompt: str, 
        routing_weights: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate text using vLLM with MoE routing
        
        Args:
            prompt: Input text prompt
            routing_weights: Optional pre-computed routing weights
            
        Returns:
            Generated text
        """
        if not self.is_initialized:
            raise BackendInferenceError("vLLM backend not initialized")
        
        try:
            # If no routing weights provided, generate them
            if routing_weights is None:
                logits = await self.generate_logits(prompt)
                routing_weights = self._softmax(logits, self.config.temperature)
            
            if self.use_http_api:
                return await self._generate_text_http(prompt, routing_weights)
            else:
                return await self._generate_text_engine(prompt, routing_weights)
                
        except Exception as e:
            logger.error(f"vLLM text generation failed: {e}")
            raise BackendInferenceError(f"vLLM generation failed: {e}")
    
    async def _generate_text_http(self, prompt: str, routing_weights: np.ndarray) -> str:
        """Generate text using HTTP API"""
        try:
            dominant_expert = np.argmax(routing_weights)
            confidence = float(np.max(routing_weights))
            
            # In real implementation, would modify sampling parameters based on routing
            request_data = {
                "model": self.config.model_name,
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature * (1.0 + confidence * 0.2),  # Adjust temp
                "top_p": self.config.top_p
            }
            
            # Mock HTTP generation call
            # response = await self.client.post(f"{self.api_base_url}/v1/completions", json=request_data)
            
            # Simulate response
            generated_text = f"vLLM generated text (expert {dominant_expert}, conf: {confidence:.3f}). "
            generated_text += f"High-throughput response to: '{prompt[:40]}...'"
            
            return generated_text
            
        except Exception as e:
            raise BackendInferenceError(f"vLLM HTTP generation failed: {e}")
    
    async def _generate_text_engine(self, prompt: str, routing_weights: np.ndarray) -> str:
        """Generate text using vLLM engine"""
        try:
            # Modify sampling parameters based on routing weights
            dominant_expert = np.argmax(routing_weights)
            confidence = float(np.max(routing_weights))
            
            # Different experts could use different generation strategies
            expert_sampling = self._get_expert_sampling_params(dominant_expert, confidence)
            
            # Generate using vLLM engine
            results = await self.engine.generate([prompt], expert_sampling)
            
            # Extract generated text
            if results and len(results) > 0:
                generated_text = results[0].outputs[0].text
            else:
                generated_text = f"vLLM engine response (expert {dominant_expert})"
            
            return generated_text
            
        except Exception as e:
            raise BackendInferenceError(f"vLLM engine generation failed: {e}")
    
    def _get_expert_sampling_params(self, expert_id: int, confidence: float):
        """Get sampling parameters for specific expert"""
        base_temp = self.config.temperature
        
        # Different experts use different sampling strategies
        expert_configs = {
            0: {"temp_mult": 0.8, "top_p": 0.85},  # More conservative
            1: {"temp_mult": 1.2, "top_p": 0.95},  # More creative  
            2: {"temp_mult": 0.6, "top_p": 0.8},   # Very focused
            3: {"temp_mult": 1.0, "top_p": 0.9},   # Balanced
            4: {"temp_mult": 1.4, "top_p": 0.98},  # Very creative
        }
        
        expert_config = expert_configs.get(expert_id, {"temp_mult": 1.0, "top_p": 0.9})
        
        # Adjust based on confidence
        temperature = base_temp * expert_config["temp_mult"] * (1.0 + confidence * 0.1)
        top_p = expert_config["top_p"]
        
        # Create sampling parameters (mock if vLLM not available)
        if VLLM_AVAILABLE:
            return SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.config.max_tokens,
                n=1
            )
        else:
            return self._create_mock_sampling_params()
    
    def _softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply softmax to logits with temperature scaling"""
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)
    
    async def estimate_ambiguity(self, prompt: str) -> float:
        """vLLM-specific ambiguity estimation using model features"""
        if not self.is_initialized:
            return await super().estimate_ambiguity(prompt)
        
        try:
            # Enhanced ambiguity estimation using vLLM features
            base_ambiguity = await super().estimate_ambiguity(prompt)
            
            # Get model hidden states for analysis
            if not self.use_http_api and self.engine:
                hidden_states = await self.engine.get_model_hidden_states([prompt])
                hidden_state = hidden_states[0]
                
                # Analyze hidden state properties
                state_norm = np.linalg.norm(hidden_state)
                state_entropy = -np.sum(np.abs(hidden_state) * np.log(np.abs(hidden_state) + 1e-8))
                
                # Normalize entropy
                normalized_entropy = min(1.0, state_entropy / 1000.0)
                
                # Combine with base ambiguity
                model_ambiguity = 0.5 * base_ambiguity + 0.3 * normalized_entropy + 0.2 * min(1.0, state_norm / 10.0)
            else:
                # HTTP API mode - use base estimation with slight variation
                model_ambiguity = base_ambiguity + np.random.normal(0, 0.05)
            
            ambiguity = np.clip(model_ambiguity, 0.0, 1.0)
            
            logger.debug(f"vLLM ambiguity estimation: base={base_ambiguity:.3f}, final={ambiguity:.3f}")
            return float(ambiguity)
            
        except Exception as e:
            logger.warning(f"vLLM ambiguity estimation failed, using base method: {e}")
            return await super().estimate_ambiguity(prompt)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM-specific model information"""
        base_info = await super().get_model_info()
        
        vllm_info = {
            "vllm_version": "simulated",  # In real: vllm.__version__
            "use_http_api": self.use_http_api,
            "api_base_url": self.api_base_url if self.use_http_api else None,
            "tensor_parallel_size": self.config.custom_params.get('tensor_parallel_size', 1),
            "gpu_memory_utilization": self.config.custom_params.get('gpu_memory_utilization', 0.9),
            "max_model_len": self.config.custom_params.get('max_model_len', 4096),
            "expert_routing_initialized": self._expert_layer is not None
        }
        
        base_info.update(vllm_info)
        return base_info
    
    async def health_check(self) -> Dict[str, Any]:
        """vLLM-specific health check"""
        try:
            base_health = await super().health_check()
            
            if base_health["status"] == "healthy":
                vllm_health = {
                    "engine_initialized": self.engine is not None,
                    "expert_routing_ready": self._expert_layer is not None,
                    "sampling_params_set": self.sampling_params is not None
                }
                
                if self.use_http_api and self.client:
                    # Test HTTP API connection
                    try:
                        response = await self.client.get(f"{self.api_base_url}/health", timeout=5)
                        vllm_health["api_connection"] = response.status_code == 200
                    except:
                        vllm_health["api_connection"] = False
                
                base_health.update(vllm_health)
            
            return base_health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"vLLM health check failed: {e}",
                "backend_type": self.config.backend_type
            }
    
    def cleanup(self):
        """Clean up vLLM resources"""
        super().cleanup()
        
        # Clean up vLLM-specific resources
        if self.client:
            asyncio.create_task(self.client.aclose())
            self.client = None
        
        self.engine = None
        self.sampling_params = None
        self._expert_layer = None
        
        logger.info("vLLM backend cleaned up")


# Mock classes for demonstration when vLLM not available
class MockRequestOutput:
    def __init__(self, request_id: str, prompt: str, outputs: List):
        self.request_id = request_id
        self.prompt = prompt
        self.outputs = outputs


class MockCompletionOutput:
    def __init__(self, text: str, token_ids: List[int]):
        self.text = text
        self.token_ids = token_ids