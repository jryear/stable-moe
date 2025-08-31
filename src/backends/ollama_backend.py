"""
Ollama Backend Implementation
Local model serving backend using Ollama
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import httpx
import json

from .base import BaseLLMBackend, BackendConfig, BackendError, BackendInitializationError, BackendInferenceError

logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLMBackend):
    """
    Ollama Backend for local model serving
    
    Supports:
    - Local model hosting
    - Multiple model formats (GGML, GGUF, etc.)  
    - Streaming generation
    - Custom model loading
    - MoE routing via embeddings
    """
    
    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.client = None
        self.base_url = f"http://{config.host}:{config.port}"
        self._expert_embeddings = None
        
    async def initialize(self) -> bool:
        """Initialize Ollama client and verify model availability"""
        try:
            logger.info(f"Initializing Ollama backend with model: {self.config.model_name}")
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
            
            # Check Ollama server health
            await self._check_ollama_health()
            
            # Verify model is available
            await self._ensure_model_available()
            
            # Initialize expert routing using embeddings
            await self._initialize_expert_routing()
            
            self.is_initialized = True
            logger.info("Ollama backend initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama backend: {e}")
            raise BackendInitializationError(f"Ollama initialization failed: {e}")
    
    async def _check_ollama_health(self):
        """Check if Ollama server is running and healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/api/version")
            if response.status_code != 200:
                raise BackendInitializationError(f"Ollama server unhealthy: {response.status_code}")
            
            version_info = response.json()
            logger.info(f"Connected to Ollama server version: {version_info.get('version', 'unknown')}")
            
        except httpx.RequestError as e:
            raise BackendInitializationError(f"Cannot connect to Ollama server at {self.base_url}: {e}")
    
    async def _ensure_model_available(self):
        """Ensure the specified model is available, pull if necessary"""
        try:
            # Check if model exists locally
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            model_available = any(
                self.config.model_name in model_name or model_name.startswith(self.config.model_name)
                for model_name in available_models
            )
            
            if not model_available:
                logger.info(f"Model {self.config.model_name} not found locally, attempting to pull...")
                await self._pull_model()
            else:
                logger.info(f"Model {self.config.model_name} is available locally")
                
        except Exception as e:
            raise BackendInitializationError(f"Model availability check failed: {e}")
    
    async def _pull_model(self):
        """Pull model from Ollama registry"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.config.model_name},
                timeout=300  # Model pulling can take a while
            )
            
            # Stream the pull response
            async for line in response.aiter_lines():
                if line:
                    try:
                        pull_data = json.loads(line)
                        if 'status' in pull_data:
                            logger.info(f"Pull status: {pull_data['status']}")
                        if pull_data.get('status') == 'success':
                            break
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Successfully pulled model: {self.config.model_name}")
            
        except Exception as e:
            raise BackendInitializationError(f"Failed to pull model {self.config.model_name}: {e}")
    
    async def _initialize_expert_routing(self):
        """Initialize expert routing using Ollama embeddings"""
        try:
            # Create expert prototypes for routing
            expert_prompts = [
                "technical and precise analysis",
                "creative and innovative thinking", 
                "practical and direct solutions",
                "comprehensive and detailed explanation",
                "conversational and accessible approach"
            ]
            
            # Get embeddings for expert prototypes
            self._expert_embeddings = []
            
            for i, prompt in enumerate(expert_prompts[:self.config.num_experts]):
                try:
                    embedding = await self._get_embedding(prompt)
                    self._expert_embeddings.append(embedding)
                    logger.debug(f"Generated embedding for expert {i}: {len(embedding)} dimensions")
                except Exception as e:
                    logger.warning(f"Failed to get embedding for expert {i}, using random: {e}")
                    # Fallback to random embedding
                    self._expert_embeddings.append(np.random.normal(0, 0.1, 384).tolist())  # Common embedding size
            
            logger.info(f"Initialized expert routing with {len(self._expert_embeddings)} experts")
            
        except Exception as e:
            logger.warning(f"Expert routing initialization failed, using fallback: {e}")
            # Fallback to random embeddings
            embedding_dim = 384
            self._expert_embeddings = [
                np.random.normal(0, 0.1, embedding_dim).tolist()
                for _ in range(self.config.num_experts)
            ]
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.config.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            
            embedding_data = response.json()
            return embedding_data.get('embedding', [])
            
        except Exception as e:
            logger.warning(f"Failed to get Ollama embedding: {e}")
            # Return empty list to trigger fallback
            return []
    
    async def generate_logits(
        self, 
        prompt: str, 
        num_experts: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate expert routing logits using Ollama embeddings
        
        Args:
            prompt: Input text prompt
            num_experts: Number of experts (defaults to config.num_experts)
            
        Returns:
            Numpy array of expert routing logits
        """
        if not self.is_initialized:
            raise BackendInferenceError("Ollama backend not initialized")
        
        try:
            num_experts = num_experts or self.config.num_experts
            
            # Get embedding for input prompt
            prompt_embedding = await self._get_embedding(prompt)
            
            if not prompt_embedding:
                # Fallback to simple heuristics if embedding fails
                return await self._fallback_logit_generation(prompt, num_experts)
            
            # Calculate similarity with expert embeddings
            logits = []
            prompt_vec = np.array(prompt_embedding)
            
            for i in range(num_experts):
                if i < len(self._expert_embeddings):
                    expert_vec = np.array(self._expert_embeddings[i])
                    
                    # Cosine similarity
                    similarity = np.dot(prompt_vec, expert_vec) / (
                        np.linalg.norm(prompt_vec) * np.linalg.norm(expert_vec) + 1e-8
                    )
                    
                    # Convert similarity to logit (similarity is in [-1, 1])
                    logit = similarity * 2.0  # Scale to [-2, 2] range
                    logits.append(logit)
                else:
                    # Fallback for missing experts
                    logits.append(np.random.normal(0, 0.5))
            
            logits = np.array(logits)
            
            # Add prompt-specific adjustments
            logits = self._adjust_logits_for_prompt(logits, prompt)
            
            logger.debug(f"Generated Ollama logits: {logits}")
            return logits
            
        except Exception as e:
            logger.error(f"Ollama logit generation failed: {e}")
            raise BackendInferenceError(f"Ollama inference failed: {e}")
    
    async def _fallback_logit_generation(self, prompt: str, num_experts: int) -> np.ndarray:
        """Fallback logit generation when embeddings fail"""
        logger.warning("Using fallback logit generation")
        
        # Simple heuristic-based logit generation
        logits = np.random.normal(0, 0.3, num_experts)
        
        # Add some structure based on prompt characteristics
        prompt_lower = prompt.lower()
        
        # Technical prompts favor first expert
        if any(word in prompt_lower for word in ['technical', 'algorithm', 'code', 'data']):
            logits[0] += 0.5
        
        # Creative prompts favor second expert
        if any(word in prompt_lower for word in ['creative', 'story', 'imagine', 'art']):
            if num_experts > 1:
                logits[1] += 0.5
        
        # Question prompts favor explanation expert
        if any(word in prompt_lower for word in ['what', 'how', 'why', 'explain']):
            if num_experts > 3:
                logits[3] += 0.3
        
        return logits
    
    def _adjust_logits_for_prompt(self, logits: np.ndarray, prompt: str) -> np.ndarray:
        """Adjust logits based on prompt characteristics"""
        # Length-based adjustment
        if len(prompt) < 20:
            # Short prompts might need more direct expert
            logits[2] += 0.2 if len(logits) > 2 else 0
        elif len(prompt) > 200:
            # Long prompts might need comprehensive expert
            logits[-1] += 0.2
        
        # Add controlled noise for variation
        noise = np.random.normal(0, 0.1, size=logits.shape)
        logits += noise
        
        return logits
    
    async def generate_text(
        self, 
        prompt: str, 
        routing_weights: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate text using Ollama with MoE routing influence
        
        Args:
            prompt: Input text prompt
            routing_weights: Optional pre-computed routing weights
            
        Returns:
            Generated text
        """
        if not self.is_initialized:
            raise BackendInferenceError("Ollama backend not initialized")
        
        try:
            # If no routing weights provided, generate them
            if routing_weights is None:
                logits = await self.generate_logits(prompt)
                routing_weights = self._softmax(logits, self.config.temperature)
            
            # Determine dominant expert and modify generation parameters
            dominant_expert = np.argmax(routing_weights)
            confidence = float(np.max(routing_weights))
            
            # Modify prompt based on expert selection
            enhanced_prompt = self._enhance_prompt_for_expert(prompt, dominant_expert)
            
            # Generate text using Ollama
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self._get_expert_temperature(dominant_expert, confidence),
                        "top_p": self.config.top_p,
                        "num_predict": self.config.max_tokens
                    }
                }
            )
            response.raise_for_status()
            
            generation_data = response.json()
            generated_text = generation_data.get('response', '')
            
            # Add routing metadata for debugging
            if self.config.custom_params.get('include_routing_info', False):
                generated_text += f"\n\n[Routed to expert {dominant_expert}, confidence: {confidence:.3f}]"
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Ollama text generation failed: {e}")
            raise BackendInferenceError(f"Ollama generation failed: {e}")
    
    def _enhance_prompt_for_expert(self, prompt: str, expert_id: int) -> str:
        """Enhance prompt based on selected expert"""
        expert_prefixes = {
            0: "Provide a technical and precise response: ",
            1: "Think creatively and innovatively about: ",
            2: "Give a practical and direct answer to: ",
            3: "Provide a comprehensive and detailed explanation of: ",
            4: "Respond in a conversational and accessible way to: "
        }
        
        prefix = expert_prefixes.get(expert_id, "")
        return f"{prefix}{prompt}" if prefix else prompt
    
    def _get_expert_temperature(self, expert_id: int, confidence: float) -> float:
        """Get temperature setting for specific expert"""
        base_temp = self.config.temperature
        
        expert_temp_multipliers = {
            0: 0.7,  # Technical - more focused
            1: 1.3,  # Creative - more diverse
            2: 0.8,  # Practical - moderately focused
            3: 1.0,  # Comprehensive - balanced
            4: 1.1   # Conversational - slightly creative
        }
        
        multiplier = expert_temp_multipliers.get(expert_id, 1.0)
        
        # Adjust based on confidence (higher confidence = lower temperature)
        confidence_adjustment = 1.0 - (confidence * 0.2)
        
        return base_temp * multiplier * confidence_adjustment
    
    def _softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply softmax to logits"""
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)
    
    async def estimate_ambiguity(self, prompt: str) -> float:
        """Ollama-specific ambiguity estimation using embeddings"""
        if not self.is_initialized:
            return await super().estimate_ambiguity(prompt)
        
        try:
            base_ambiguity = await super().estimate_ambiguity(prompt)
            
            # Get prompt embedding for analysis
            prompt_embedding = await self._get_embedding(prompt)
            
            if prompt_embedding:
                # Analyze embedding properties for ambiguity
                embedding_vec = np.array(prompt_embedding)
                
                # Calculate similarity variance with expert embeddings
                similarities = []
                for expert_embedding in self._expert_embeddings:
                    expert_vec = np.array(expert_embedding)
                    similarity = np.dot(embedding_vec, expert_vec) / (
                        np.linalg.norm(embedding_vec) * np.linalg.norm(expert_vec) + 1e-8
                    )
                    similarities.append(similarity)
                
                # High variance in similarities indicates ambiguity
                similarity_variance = np.var(similarities)
                variance_ambiguity = min(1.0, similarity_variance * 5.0)  # Scale to [0, 1]
                
                # Combine with base ambiguity
                combined_ambiguity = 0.6 * base_ambiguity + 0.4 * variance_ambiguity
            else:
                # Fallback to base ambiguity with slight variation
                combined_ambiguity = base_ambiguity + np.random.normal(0, 0.03)
            
            ambiguity = np.clip(combined_ambiguity, 0.0, 1.0)
            
            logger.debug(f"Ollama ambiguity estimation: base={base_ambiguity:.3f}, final={ambiguity:.3f}")
            return float(ambiguity)
            
        except Exception as e:
            logger.warning(f"Ollama ambiguity estimation failed, using base method: {e}")
            return await super().estimate_ambiguity(prompt)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama-specific model information"""
        base_info = await super().get_model_info()
        
        ollama_info = {
            "ollama_server_url": self.base_url,
            "expert_embeddings_initialized": len(self._expert_embeddings) if self._expert_embeddings else 0,
            "embedding_dimension": len(self._expert_embeddings[0]) if self._expert_embeddings and self._expert_embeddings[0] else 0
        }
        
        # Try to get additional model info from Ollama
        try:
            response = await self.client.post(
                f"{self.base_url}/api/show",
                json={"name": self.config.model_name}
            )
            if response.status_code == 200:
                model_data = response.json()
                ollama_info.update({
                    "model_size": model_data.get('size', 'unknown'),
                    "model_format": model_data.get('format', 'unknown'),
                    "model_family": model_data.get('details', {}).get('family', 'unknown')
                })
        except:
            pass  # Skip if model info unavailable
        
        base_info.update(ollama_info)
        return base_info
    
    async def health_check(self) -> Dict[str, Any]:
        """Ollama-specific health check"""
        try:
            base_health = await super().health_check()
            
            if base_health["status"] == "healthy":
                # Additional Ollama-specific checks
                ollama_health = {
                    "ollama_server_reachable": False,
                    "model_available": False,
                    "expert_embeddings_ready": self._expert_embeddings is not None
                }
                
                # Test server connection
                try:
                    response = await self.client.get(f"{self.base_url}/api/version", timeout=5)
                    ollama_health["ollama_server_reachable"] = response.status_code == 200
                except:
                    pass
                
                # Test model availability
                try:
                    response = await self.client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.config.model_name,
                            "prompt": "test",
                            "stream": False,
                            "options": {"num_predict": 1}
                        },
                        timeout=10
                    )
                    ollama_health["model_available"] = response.status_code == 200
                except:
                    pass
                
                base_health.update(ollama_health)
                
                # Update overall status based on Ollama-specific checks
                if not ollama_health["ollama_server_reachable"] or not ollama_health["model_available"]:
                    base_health["status"] = "degraded"
            
            return base_health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Ollama health check failed: {e}",
                "backend_type": self.config.backend_type
            }
    
    def cleanup(self):
        """Clean up Ollama resources"""
        super().cleanup()
        
        if self.client:
            asyncio.create_task(self.client.aclose())
            self.client = None
        
        self._expert_embeddings = None
        
        logger.info("Ollama backend cleaned up")