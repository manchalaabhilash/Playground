import os
import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class DynamicQuantizer:
    """Module for dynamic quantization of PyTorch models"""
    
    def __init__(self, model=None, model_path=None, output_dir="./quantized_models"):
        self.model = model
        self.model_path = model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.quantized_model = None
    
    def load_model(self):
        """Load model if not provided during initialization"""
        if self.model is not None:
            logger.info("Using provided model")
            return True
        
        try:
            if self.model_path is None:
                logger.error("Either model or model_path must be provided")
                return False
            
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32  # Ensure model is in FP32 for quantization
            )
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def quantize(self, quantization_type="dynamic", bits=8):
        """Quantize model using dynamic or static quantization"""
        try:
            if self.model is None:
                success = self.load_model()
                if not success:
                    return False
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set model to eval mode
            self.model.eval()
            
            if quantization_type == "dynamic":
                logger.info(f"Applying dynamic quantization with {bits} bits")
                
                if bits == 8:
                    # Apply dynamic quantization to INT8
                    self.quantized_model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},  # Quantize only linear layers
                        dtype=torch.qint8
                    )
                elif bits == 4:
                    # For 4-bit, we need to use a different approach
                    # This is a simplified version - in practice, you might need a more sophisticated approach
                    logger.warning("4-bit dynamic quantization is experimental")
                    from transformers import BitsAndBytesConfig
                    
                    # Reload model with 4-bit quantization
                    self.quantized_model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                    )
                else:
                    logger.error(f"Unsupported bit width: {bits}")
                    return False
            
            elif quantization_type == "static":
                logger.info(f"Applying static quantization with {bits} bits")
                
                # For static quantization, we need calibration data
                # This is a simplified example
                
                # Define a calibration function
                def calibrate(model):
                    # Generate some random input for calibration
                    sample_input = torch.randint(
                        0, 
                        self.tokenizer.vocab_size if self.tokenizer else 10000, 
                        (4, 128)
                    )
                    
                    # Run model with calibration inputs
                    with torch.no_grad():
                        model(sample_input)
                
                # Prepare for static quantization
                model_prepared = torch.quantization.prepare(self.model)
                
                # Calibrate
                calibrate(model_prepared)
                
                # Convert to quantized model
                self.quantized_model = torch.quantization.convert(model_prepared)
            
            else:
                logger.error(f"Unknown quantization type: {quantization_type}")
                return False
            
            # Save quantized model
            quantized_model_path = os.path.join(self.output_dir, f"model_quantized_{bits}bit.pt")
            torch.save(self.quantized_model.state_dict(), quantized_model_path)
            
            # Save model config and tokenizer
            if hasattr(self.model, 'config'):
                self.model.config.save_pretrained(self.output_dir)
            
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.output_dir)
            
            logger.info(f"Model quantized and saved to {quantized_model_path}")
            return True
        except Exception as e:
            logger.error(f"Error during quantization: {str(e)}")
            return False
    
    def benchmark(self, input_text="Hello, how are you?", num_runs=10, max_length=50):
        """Benchmark the quantized model against the original"""
        try:
            if self.model is None or self.quantized_model is None:
                logger.error("Both original and quantized models must be available for benchmarking")
                return False
            
            if self.tokenizer is None:
                logger.error("Tokenizer is required for benchmarking")
                return False
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            # Benchmark original model
            logger.info("Benchmarking original model...")
            original_times = []
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                with torch.no_grad():
                    self.model.generate(**inputs, max_length=max_length)
                end.record()
                
                torch.cuda.synchronize()
                original_times.append(start.elapsed_time(end))
            
            avg_original_time = sum(original_times) / len(original_times)
            
            # Benchmark quantized model
            logger.info("Benchmarking quantized model...")
            quantized_times = []
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                with torch.no_grad():
                    self.quantized_model.generate(**inputs, max_length=max_length)
                end.record()
                
                torch.cuda.synchronize()
                quantized_times.append(start.elapsed_time(end))
            
            avg_quantized_time = sum(quantized_times) / len(quantized_times)
            
            # Calculate speedup and memory savings
            speedup = avg_original_time / avg_quantized_time
            
            # Estimate memory usage
            original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)  # MB
            quantized_size = sum(p.numel() * p.element_size() for p in self.quantized_model.parameters()) / (1024 * 1024)  # MB
            memory_reduction = (original_size - quantized_size) / original_size * 100
            
            # Log results
            logger.info(f"Original model average time: {avg_original_time:.2f} ms")
            logger.info(f"Quantized model average time: {avg_quantized_time:.2f} ms")
            logger.info(f"Speedup: {speedup:.2f}x")
            logger.info(f"Original model size: {original_size:.2f} MB")
            logger.info(f"Quantized model size: {quantized_size:.2f} MB")
            logger.info(f"Memory reduction: {memory_reduction:.2f}%")
            
            # Save benchmark results
            with open(os.path.join(self.output_dir, "benchmark_results.txt"), "w") as f:
                f.write(f"Original model average time: {avg_original_time:.2f} ms\n")
                f.write(f"Quantized model average time: {avg_quantized_time:.2f} ms\n")
                f.write(f"Speedup: {speedup:.2f}x\n")
                f.write(f"Original model size: {original_size:.2f} MB\n")
                f.write(f"Quantized model size: {quantized_size:.2f} MB\n")
                f.write(f"Memory reduction: {memory_reduction:.2f}%\n")
            
            return True
        except Exception as e:
            logger.error(f"Error during benchmarking: {str(e)}")
            return False