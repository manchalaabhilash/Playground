import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
import logging

logger = logging.getLogger(__name__)

# Add imports for model cache
from src.model_cache import model_cache
from src.model_monitoring import monitor_model_inference

class OptimizedLLM:
    """
    Optimized LLM for efficient inference
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the optimized LLM
        
        Args:
            model_path: Path to the model
            device: Device to use for inference
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_id = os.path.basename(model_path)
    
    def load_model(self, use_8bit: bool = False, use_4bit: bool = False, 
                  use_cache: bool = True) -> bool:
        """
        Load the model
        
        Args:
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization
            use_cache: Whether to use model cache
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model is in cache
            if use_cache:
                cached_model = model_cache.get(
                    model_path=self.model_path,
                    model_type="optimized",
                    use_8bit=use_8bit,
                    use_4bit=use_4bit
                )
                
                if cached_model:
                    # Model found in cache
                    self.model, self.tokenizer = cached_model
                    logger.info(f"Loaded model from cache: {self.model_path}")
                    return True
            
            # Load model from disk
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model: {self.model_path}")
            
            # Determine quantization parameters
            quantization_config = None
            if use_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
            elif use_8bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
            
            # Add to cache if requested
            if use_cache:
                model_cache.put(
                    model_path=self.model_path,
                    model_type="optimized",
                    model=(self.model, self.tokenizer),
                    use_8bit=use_8bit,
                    use_4bit=use_4bit
                )
            
            logger.info(f"Model loaded successfully: {self.model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def export_to_onnx(self, output_dir="./onnx_model"):
        """Export model to ONNX format for faster inference"""
        if self.model is None or self.tokenizer is None:
            logger.error("Model must be loaded before exporting to ONNX")
            return False
        
        try:
            from optimum.exporters import OnnxConfig, TasksManager
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get appropriate config for the model
            onnx_config = TasksManager.get_exporter_config_constructor(
                model=self.model,
                task="text-generation"
            )()
            
            # Export the model
            model_path = TasksManager.export(
                model=self.model,
                tokenizer=self.tokenizer,
                config=onnx_config,
                output=output_dir
            )
            
            logger.info(f"Model exported to ONNX format at {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {str(e)}")
            return False
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7,
                top_p: float = 0.9, top_k: int = 50) -> str:
        """
        Generate text
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        
        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        # Define the generation function
        def _generate():
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Count tokens
            input_token_count = len(inputs.input_ids[0])
            output_token_count = len(outputs[0]) - input_token_count
            
            # Add token counts to result
            result = generated_text
            result.input_token_count = input_token_count
            result.output_token_count = output_token_count
            
            return result
        
        # Monitor the generation
        return monitor_model_inference(self.model_id, _generate)
