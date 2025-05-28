import os
import sys
import logging
from celery import Celery, shared_task
from typing import Dict, Any, List, Optional, Union

sys.path.append('.')

from src.optimized_llm import OptimizedLLM
from src.dynamic_quantization import DynamicQuantizer
from src.model_ensemble import ModelEnsemble
from src.model_pruning import ModelPruner
from src.tensorrt_optimization import TensorRTOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('multimodal_rag')
celery_app.config_from_object({
    'broker_url': os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    'result_backend': os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'enable_utc': True,
    'task_track_started': True,
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
})

celery_app.set_default()

@shared_task(ignore_result=False)
def optimize_model(model_path: str, optimization_type: str, output_dir: str, 
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Background task to optimize a model
    
    Args:
        model_path: Path to the model
        optimization_type: Type of optimization (quantize, prune, tensorrt)
        output_dir: Output directory for optimized model
        params: Additional parameters for optimization
    
    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info(f"Starting model optimization: {optimization_type} for {model_path}")
        
        if params is None:
            params = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        result = {
            "success": False,
            "model_path": model_path,
            "optimization_type": optimization_type,
            "output_dir": output_dir,
            "error": None
        }
        
        # Perform optimization based on type
        if optimization_type == "quantize":
            # Initialize quantizer
            quantizer = DynamicQuantizer(model_path=model_path)
            
            # Quantize model
            bits = params.get("bits", 8)
            quantization_type = params.get("quantization_type", "dynamic")
            
            success = quantizer.quantize(
                quantization_type=quantization_type,
                bits=bits,
                output_dir=output_dir
            )
            
            if success:
                result["success"] = True
                result["optimized_model_path"] = os.path.join(output_dir, f"model_quantized_{bits}bit")
                result["size_reduction"] = quantizer.size_reduction
        
        elif optimization_type == "prune":
            # Initialize pruner
            pruner = ModelPruner(model_path=model_path, output_dir=output_dir)
            
            # Prune model
            method = params.get("method", "magnitude")
            sparsity = params.get("sparsity", 0.3)
            target_modules = params.get("target_modules", ["query", "key", "value", "dense", "fc1", "fc2"])
            
            success = pruner.prune_model(
                method=method,
                sparsity=sparsity,
                target_modules=target_modules
            )
            
            if success:
                result["success"] = True
                result["optimized_model_path"] = os.path.join(output_dir, f"model_pruned_{method}_{int(sparsity*100)}pct")
                result["original_size"] = pruner.original_size
                result["pruned_size"] = pruner.pruned_size
                result["size_reduction"] = (pruner.original_size - pruner.pruned_size) / pruner.original_size
        
        elif optimization_type == "tensorrt":
            # Initialize TensorRT optimizer
            optimizer = TensorRTOptimizer(model_path=model_path)
            
            # Optimize with TensorRT
            precision = params.get("precision", "fp16")
            
            success = optimizer.optimize_with_tensorrt(
                precision=precision,
                output_dir=output_dir
            )
            
            if success:
                result["success"] = True
                result["optimized_model_path"] = os.path.join(output_dir, f"model_tensorrt_{precision}")
                result["speedup"] = optimizer.speedup
        
        else:
            result["error"] = f"Unknown optimization type: {optimization_type}"
        
        logger.info(f"Model optimization completed: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error during model optimization: {str(e)}")
        return {
            "success": False,
            "model_path": model_path,
            "optimization_type": optimization_type,
            "output_dir": output_dir,
            "error": str(e)
        }

@shared_task(ignore_result=False)
def batch_inference(model_path: str, model_type: str, prompts: List[str], 
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Background task to run batch inference on multiple prompts
    
    Args:
        model_path: Path to the model
        model_type: Type of model (optimized, quantized, tensorrt, ensemble)
        prompts: List of prompts to process
        params: Additional parameters for generation
    
    Returns:
        Dictionary with inference results
    """
    try:
        logger.info(f"Starting batch inference with {model_type} model for {len(prompts)} prompts")
        
        if params is None:
            params = {}
        
        result = {
            "success": False,
            "model_path": model_path,
            "model_type": model_type,
            "num_prompts": len(prompts),
            "results": [],
            "error": None
        }
        
        # Load model based on type
        if model_type == "quantized":
            # Initialize quantizer
            model = DynamicQuantizer(model_path=model_path)
            
            # Load quantized model
            success = model.load_quantized_model()
            
            if not success:
                result["error"] = "Failed to load quantized model"
                return result
        
        elif model_type == "tensorrt":
            # Initialize optimizer
            model = TensorRTOptimizer(model_path=model_path)
            
            # Load optimized model
            success = model.load_optimized_model()
            
            if not success:
                result["error"] = "Failed to load TensorRT optimized model"
                return result
        
        elif model_type == "ensemble":
            # Initialize ensemble
            model_paths = model_path.split(",")
            model = ModelEnsemble(model_paths=model_paths)
            
            # Load models
            use_8bit = params.get("use_8bit", False)
            success = model.load_models(use_8bit=use_8bit)
            
            if not success:
                result["error"] = "Failed to load ensemble models"
                return result
        
        else:  # Default to optimized LLM
            # Initialize optimized LLM
            model = OptimizedLLM(model_path=model_path)
            
            # Load model
            use_8bit = params.get("use_8bit", False)
            success = model.load_model(use_8bit=use_8bit)
            
            if not success:
                result["error"] = "Failed to load optimized model"
                return result
        
        # Extract generation parameters
        max_length = params.get("max_length", 100)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.9)
        top_k = params.get("top_k", 50)
        ensemble_method = params.get("ensemble_method", "mean")
        
        # Process each prompt
        for prompt in prompts:
            try:
                # Generate text
                if model_type == "ensemble":
                    output = model.ensemble_generate(
                        prompt=prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        ensemble_method=ensemble_method
                    )
                else:
                    # For other model types
                    output = model.generate(
                        prompt=prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
                
                # Add to results
                result["results"].append({
                    "prompt": prompt,
                    "generated_text": output,
                    "success": True,
                    "error": None
                })
            
            except Exception as e:
                # Add failed result
                result["results"].append({
                    "prompt": prompt,
                    "generated_text": None,
                    "success": False,
                    "error": str(e)
                })
        
        # Update overall success
        result["success"] = True
        
        logger.info(f"Batch inference completed for {len(prompts)} prompts")
        return result
    
    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}")
        return {
            "success": False,
            "model_path": model_path,
            "model_type": model_type,
            "num_prompts": len(prompts) if prompts else 0,
            "results": [],
            "error": str(e)
        }