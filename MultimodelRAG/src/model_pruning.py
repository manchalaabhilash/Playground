import os
import torch
import logging
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelPruner:
    """Module for pruning PyTorch models to reduce size and improve inference speed"""
    
    def __init__(self, model=None, model_path=None, output_dir="./pruned_models"):
        self.model = model
        self.model_path = model_path
        self.output_dir = output_dir
        self.tokenizer = None
        self.pruned_model = None
        self.original_size = 0
        self.pruned_size = 0
    
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
                device_map="auto"
            )
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def prune_model(self, method="magnitude", sparsity=0.3, target_modules=None):
        """Prune model using specified method and sparsity level"""
        try:
            if self.model is None:
                success = self.load_model()
                if not success:
                    return False
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set model to eval mode
            self.model.eval()
            
            # Calculate original size
            self.original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)  # MB
            
            # Clone model for pruning
            self.pruned_model = type(self.model)(self.model.config)
            self.pruned_model.load_state_dict(self.model.state_dict())
            
            # Determine target modules if not specified
            if target_modules is None:
                target_modules = ["query", "key", "value", "dense", "fc1", "fc2"]
            
            # Count parameters before pruning
            total_params = 0
            for name, module in self.pruned_model.named_modules():
                if any(target in name for target in target_modules):
                    if isinstance(module, torch.nn.Linear):
                        total_params += module.weight.numel()
            
            logger.info(f"Total parameters in target modules: {total_params:,}")
            
            # Apply pruning based on method
            pruned_params = 0
            
            if method == "magnitude":
                logger.info(f"Applying magnitude pruning with sparsity {sparsity}")
                
                for name, module in self.pruned_model.named_modules():
                    if any(target in name for target in target_modules):
                        if isinstance(module, torch.nn.Linear):
                            # Get weight tensor
                            weight = module.weight.data
                            
                            # Calculate threshold for pruning
                            threshold = torch.quantile(weight.abs().flatten(), sparsity)
                            
                            # Create mask for pruning
                            mask = (weight.abs() > threshold).float()
                            
                            # Apply mask to weights
                            module.weight.data = weight * mask
                            
                            # Count pruned parameters
                            pruned_params += (mask == 0).sum().item()
            
            elif method == "random":
                logger.info(f"Applying random pruning with sparsity {sparsity}")
                
                for name, module in self.pruned_model.named_modules():
                    if any(target in name for target in target_modules):
                        if isinstance(module, torch.nn.Linear):
                            # Get weight tensor
                            weight = module.weight.data
                            
                            # Create random mask
                            mask = torch.rand_like(weight) > sparsity
                            
                            # Apply mask to weights
                            module.weight.data = weight * mask
                            
                            # Count pruned parameters
                            pruned_params += (mask == 0).sum().item()
            
            elif method == "structured":
                logger.info(f"Applying structured pruning with sparsity {sparsity}")
                
                for name, module in self.pruned_model.named_modules():
                    if any(target in name for target in target_modules):
                        if isinstance(module, torch.nn.Linear):
                            # Get weight tensor
                            weight = module.weight.data
                            
                            # Calculate importance of each output neuron (row)
                            importance = torch.norm(weight, dim=1)
                            
                            # Determine number of neurons to keep
                            num_neurons = weight.size(0)
                            keep_neurons = int(num_neurons * (1 - sparsity))
                            
                            # Get indices of neurons to keep
                            _, indices = torch.topk(importance, keep_neurons)
                            
                            # Create mask
                            mask = torch.zeros_like(weight)
                            mask[indices, :] = 1
                            
                            # Apply mask to weights
                            module.weight.data = weight * mask
                            
                            # Count pruned parameters
                            pruned_params += (mask == 0).sum().item()
            
            else:
                logger.error(f"Unknown pruning method: {method}")
                return False
            
            # Calculate pruning statistics
            pruning_ratio = pruned_params / total_params
            logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of target parameters)")
            
            # Calculate pruned size
            self.pruned_size = sum(p.numel() * p.element_size() for p in self.pruned_model.parameters()) / (1024 * 1024)  # MB
            
            # Save pruned model
            pruned_model_path = os.path.join(self.output_dir, f"model_pruned_{method}_{int(sparsity*100)}pct")
            self.pruned_model.save_pretrained(pruned_model_path)
            
            # Save tokenizer
            if self.tokenizer:
                self.tokenizer.save_pretrained(pruned_model_path)
            
            logger.info(f"Model pruned and saved to {pruned_model_path}")
            
            # Save pruning info
            with open(os.path.join(self.output_dir, "pruning_info.txt"), "w") as f:
                f.write(f"Pruning method: {method}\n")
                f.write(f"Sparsity: {sparsity}\n")
                f.write(f"Target modules: {target_modules}\n")
                f.write(f"Total parameters in target modules: {total_params:,}\n")
                f.write(f"Pruned parameters: {pruned_params:,}\n")
                f.write(f"Pruning ratio: {pruning_ratio:.2%}\n")
                f.write(f"Original model size: {self.original_size:.2f} MB\n")
                f.write(f"Pruned model size: {self.pruned_size:.2f} MB\n")
                f.write(f"Size reduction: {(self.original_size - self.pruned_size) / self.original_size:.2%}\n")
            
            return True
        except Exception as e:
            logger.error(f"Error during pruning: {str(e)}")
            return False
    
    def benchmark(self, input_text="Hello, how are you?", num_runs=10, max_length=50):
        """Benchmark the pruned model against the original"""
        try:
            if self.model is None or self.pruned_model is None:
                logger.error("Both original and pruned models must be available for benchmarking")
                return False
            
            if self.tokenizer is None:
                logger.error("Tokenizer is required for benchmarking")
                return False
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            
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
            
            # Benchmark pruned model
            logger.info("Benchmarking pruned model...")
            pruned_times = []
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                with torch.no_grad():
                    self.pruned_model.generate(**inputs, max_length=max_length)
                end.record()
                
                torch.cuda.synchronize()
                pruned_times.append(start.elapsed_time(end))
            
            avg_pruned_time = sum(pruned_times) / len(pruned_times)
            
            # Calculate speedup
            speedup = avg_original_time / avg_pruned_time
            
            # Log results
            logger.info(f"Original model average time: {avg_original_time:.2f} ms")
            logger.info(f"Pruned model average time: {avg_pruned_time:.2f} ms")
            logger.info(f"Speedup: {speedup:.2f}x")
            logger.info(f"Original model size: {self.original_size:.2f} MB")
            logger.info(f"Pruned model size: {self.pruned_size:.2f} MB")
            logger.info(f"Size reduction: {(self.original_size - self.pruned_size) / self.original_size:.2%}")
            
            # Save benchmark results
            with open(os.path.join(self.output_dir, "benchmark_results.txt"), "w") as f:
                f.write(f"Original model average time: {avg_original_time:.2f} ms\n")
                f.write(f"Pruned model average time: {avg_pruned_time:.2f} ms\n")
                f.write(f"Speedup: {speedup:.2f}x\n")
                f.write(f"Original model size: {self.original_size:.2f} MB\n")
                f.write(f"Pruned model size: {self.pruned_size:.2f} MB\n")
                f.write(f"Size reduction: {(self.original_size - self.pruned_size) / self.original_size:.2%}\n")
            
            return True
        except Exception as e:
            logger.error(f"Error during benchmarking: {str(e)}")
            return False
