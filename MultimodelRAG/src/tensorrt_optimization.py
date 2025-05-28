import os
import torch
import logging
import torch_tensorrt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class TensorRTOptimizer:
    """Module for optimizing models with TensorRT for faster inference"""
    
    def __init__(self, model=None, model_path=None, output_dir="./tensorrt_models"):
        self.model = model
        self.model_path = model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.optimized_model = None
    
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
                device_map="cuda" if torch.cuda.is_available() else None
            )
            
            # Ensure model is on CUDA
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                logger.warning("CUDA not available, TensorRT optimization requires GPU")
                return False
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def optimize_with_tensorrt(self, batch_size=1, sequence_length=128, precision="fp16"):
        """Optimize model with TensorRT"""
        try:
            if self.model is None:
                success = self.load_model()
                if not success:
                    return False
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set model to eval mode
            self.model.eval()
            
            # Create example inputs
            example_inputs = torch.randint(
                0, 
                self.tokenizer.vocab_size if self.tokenizer else 10000, 
                (batch_size, sequence_length),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Set precision
            if precision == "fp16":
                enabled_precisions = {torch.float16}
            elif precision == "fp32":
                enabled_precisions = {torch.float32}
            else:
                logger.warning(f"Unknown precision: {precision}, using fp16")
                enabled_precisions = {torch.float16}
            
            # Compile with TensorRT
            logger.info("Starting TensorRT compilation...")
            self.optimized_model = torch_tensorrt.compile(
                self.model,
                inputs=[example_inputs],
                enabled_precisions=enabled_precisions,
                workspace_size=1 << 30,  # 1GB workspace
                min_block_size=1,
                max_aux_streams=4
            )
            
            # Save optimized model
            optimized_model_path = os.path.join(self.output_dir, "model_tensorrt.pt")
            torch.save(self.optimized_model.state_dict(), optimized_model_path)
            
            # Save tokenizer if available
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.output_dir)
            
            logger.info(f"Model optimized with TensorRT and saved to {optimized_model_path}")
            return True
        except Exception as e:
            logger.error(f"Error during TensorRT optimization: {str(e)}")
            return False
    
    def benchmark(self, input_text="Hello, how are you?", num_runs=10, max_length=50):
        """Benchmark the optimized model against the original"""
        try:
            if self.model is None or self.optimized_model is None:
                logger.error("Both original and optimized models must be available for benchmarking")
                return False
            
            if self.tokenizer is None:
                logger.error("Tokenizer is required for benchmarking")
                return False
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Benchmark original model
            logger.info("Benchmarking original model...")
            original_times = []
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    self.model.generate(**inputs, max_length=max_length)
                end_time.record()
                
                torch.cuda.synchronize()
                original_times.append(start_time.elapsed_time(end_time))
            
            avg_original_time = sum(original_times) / len(original_times)
            
            # Benchmark optimized model
            logger.info("Benchmarking TensorRT optimized model...")
            optimized_times = []
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    self.optimized_model.generate(**inputs, max_length=max_length)
                end_time.record()
                
                torch.cuda.synchronize()
                optimized_times.append(start_time.elapsed_time(end_time))
            
            avg_optimized_time = sum(optimized_times) / len(optimized_times)
            
            # Calculate speedup
            speedup = avg_original_time / avg_optimized_time
            
            # Log results
            logger.info(f"Original model average time: {avg_original_time:.2f} ms")
            logger.info(f"TensorRT model average time: {avg_optimized_time:.2f} ms")
            logger.info(f"Speedup: {speedup:.2f}x")
            
            # Save benchmark results
            with open(os.path.join(self.output_dir, "benchmark_results.txt"), "w") as f:
                f.write(f"Original model average time: {avg_original_time:.2f} ms\n")
                f.write(f"TensorRT model average time: {avg_optimized_time:.2f} ms\n")
                f.write(f"Speedup: {speedup:.2f}x\n")
            
            return True
        except Exception as e:
            logger.error(f"Error during benchmarking: {str(e)}")
            return False