#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Any, List, Optional

sys.path.append('.')

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.optimized_llm import OptimizedLLM
from src.dynamic_quantization import DynamicQuantizer
from src.model_ensemble import ModelEnsemble
from src.tensorrt_optimization import TensorRTOptimizer
from src.background_tasks import optimize_model, batch_inference
from celery.result import AsyncResult
from src.model_monitoring import model_metrics, get_gpu_memory_usage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
model_type = None
tokenizer = None

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Optimized Model API is running",
        "model_type": model_type,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API information"},
            {"path": "/generate", "method": "POST", "description": "Generate text with optimized model"},
            {"path": "/benchmark", "method": "POST", "description": "Benchmark model performance"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/tasks/optimize", "method": "POST", "description": "Start model optimization task"},
            {"path": "/tasks/batch_inference", "method": "POST", "description": "Start batch inference task"},
            {"path": "/tasks/<task_id>", "method": "GET", "description": "Get task status"},
            {"path": "/monitoring/metrics", "method": "GET", "description": "Get monitoring metrics for all models"},
            {"path": "/monitoring/gpu", "method": "GET", "description": "Get GPU status"},
            {"path": "/monitoring/export", "method": "POST", "description": "Export metrics to file"}
        ]
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text with optimized model"""
    try:
        # Get request data
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        prompt = data.get("prompt", "")
        max_length = data.get("max_length", 100)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Generate text
        start_time = time.time()
        
        if model_type == "ensemble":
            output = model.ensemble_generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ensemble_method=data.get("ensemble_method", "mean")
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
        
        end_time = time.time()
        
        # Return response
        return jsonify({
            "prompt": prompt,
            "generated_text": output,
            "model_type": model_type,
            "generation_time": end_time - start_time
        })
    
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Benchmark model performance"""
    try:
        # Get request data
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        prompt = data.get("prompt", "Hello, how are you?")
        max_length = data.get("max_length", 100)
        num_runs = data.get("num_runs", 10)
        
        # Run benchmark
        start_time = time.time()
        
        # Initialize results
        times = []
        
        for _ in range(num_runs):
            run_start = time.time()
            
            if model_type == "ensemble":
                model.ensemble_generate(
                    prompt=prompt,
                    max_length=max_length,
                    ensemble_method=data.get("ensemble_method", "mean")
                )
            else:
                # For other model types
                model.generate(
                    prompt=prompt,
                    max_length=max_length
                )
            
            run_end = time.time()
            times.append(run_end - run_start)
        
        end_time = time.time()
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Return response
        return jsonify({
            "model_type": model_type,
            "num_runs": num_runs,
            "total_time": end_time - start_time,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": num_runs / (end_time - start_time)
        })
    
    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    
    return jsonify({"status": "ok", "model_type": model_type})

@app.route('/tasks/optimize', methods=['POST'])
def start_optimization():
    """Start model optimization task"""
    try:
        # Get request data
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        model_path = data.get("model_path")
        optimization_type = data.get("optimization_type")
        output_dir = data.get("output_dir", "./optimized_models")
        params = data.get("params", {})
        
        if not model_path:
            return jsonify({"error": "No model_path provided"}), 400
        
        if not optimization_type:
            return jsonify({"error": "No optimization_type provided"}), 400
        
        # Start optimization task
        task = optimize_model.delay(
            model_path=model_path,
            optimization_type=optimization_type,
            output_dir=output_dir,
            params=params
        )
        
        # Return task ID
        return jsonify({
            "task_id": task.id,
            "status": "started",
            "model_path": model_path,
            "optimization_type": optimization_type
        })
    
    except Exception as e:
        logger.error(f"Error starting optimization task: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/tasks/batch_inference', methods=['POST'])
def start_batch_inference():
    """Start batch inference task"""
    try:
        # Get request data
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        model_path = data.get("model_path")
        model_type = data.get("model_type", "optimized")
        prompts = data.get("prompts", [])
        params = data.get("params", {})
        
        if not model_path:
            return jsonify({"error": "No model_path provided"}), 400
        
        if not prompts:
            return jsonify({"error": "No prompts provided"}), 400
        
        # Start batch inference task
        task = batch_inference.delay(
            model_path=model_path,
            model_type=model_type,
            prompts=prompts,
            params=params
        )
        
        # Return task ID
        return jsonify({
            "task_id": task.id,
            "status": "started",
            "model_path": model_path,
            "model_type": model_type,
            "num_prompts": len(prompts)
        })
    
    except Exception as e:
        logger.error(f"Error starting batch inference task: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get task status"""
    try:
        # Get task result
        task_result = AsyncResult(task_id)
        
        # Check if task exists
        if not task_result:
            return jsonify({"error": f"Task {task_id} not found"}), 404
        
        # Get task status
        if task_result.ready():
            if task_result.successful():
                # Task completed successfully
                return jsonify({
                    "task_id": task_id,
                    "status": "completed",
                    "result": task_result.result
                })
            else:
                # Task failed
                return jsonify({
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(task_result.result)
                })
        else:
            # Task still running
            return jsonify({
                "task_id": task_id,
                "status": "running"
            })
    
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/monitoring/metrics', methods=['GET'])
def get_monitoring_metrics():
    """Get monitoring metrics for all models"""
    try:
        # Get query parameters
        model_id = request.args.get('model_id')
        metric_name = request.args.get('metric_name')
        window_seconds = request.args.get('window_seconds')
        
        # Convert window_seconds to float if provided
        if window_seconds:
            try:
                window_seconds = float(window_seconds)
            except ValueError:
                return jsonify({"error": "Invalid window_seconds parameter"}), 400
        
        # Get metrics
        if model_id:
            if metric_name:
                # Get specific metric for specific model
                metrics = model_metrics.get_metrics(model_id, metric_name, window_seconds)
            else:
                # Get all metrics for specific model
                metrics = model_metrics.get_metrics(model_id, window_seconds=window_seconds)
        else:
            # Get metrics for all models
            metrics = model_metrics.get_all_metrics(window_seconds=window_seconds)
        
        return jsonify(metrics)
    
    except Exception as e:
        logger.error(f"Error getting monitoring metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/monitoring/gpu', methods=['GET'])
def get_gpu_status():
    """Get GPU status"""
    try:
        # Get GPU memory usage
        gpu_memory = get_gpu_memory_usage()
        
        return jsonify(gpu_memory)
    
    except Exception as e:
        logger.error(f"Error getting GPU status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/monitoring/export', methods=['POST'])
def export_metrics():
    """Export metrics to file"""
    try:
        # Get request data
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        model_id = data.get("model_id")
        metric_name = data.get("metric_name")
        format = data.get("format", "json")
        filename = data.get("filename")
        
        if not model_id:
            return jsonify({"error": "No model_id provided"}), 400
        
        # Export metrics
        if format.lower() == "csv":
            if not metric_name:
                return jsonify({"error": "metric_name is required for CSV export"}), 400
            
            model_metrics.export_csv(model_id, metric_name, filename)
            
            return jsonify({
                "success": True,
                "format": "csv",
                "model_id": model_id,
                "metric_name": metric_name,
                "filename": filename
            })
        else:
            # Default to JSON
            model_metrics.save_metrics(filename)
            
            return jsonify({
                "success": True,
                "format": "json",
                "filename": filename
            })
    
    except Exception as e:
        logger.error(f"Error exporting metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

def load_model(args):
    """Load the specified model"""
    global model, model_type, tokenizer
    
    try:
        if args.model_type == "quantized":
            logger.info(f"Loading quantized model from {args.model_path}")
            
            # Initialize quantizer
            quantizer = DynamicQuantizer(model_path=args.model_path)
            
            # Load quantized model
            success = quantizer.load_quantized_model()
            
            if not success:
                logger.error("Failed to load quantized model")
                return False
            
            model = quantizer
            tokenizer = quantizer.tokenizer
        
        elif args.model_type == "tensorrt":
            logger.info(f"Loading TensorRT optimized model from {args.model_path}")
            
            # Initialize optimizer
            optimizer = TensorRTOptimizer(model_path=args.model_path)
            
            # Load optimized model
            success = optimizer.load_optimized_model()
            
            if not success:
                logger.error("Failed to load TensorRT optimized model")
                return False
            
            model = optimizer
            tokenizer = optimizer.tokenizer
        
        elif args.model_type == "ensemble":
            logger.info(f"Loading ensemble models from {args.model_paths}")
            
            # Initialize ensemble
            ensemble = ModelEnsemble(model_paths=args.model_paths.split(","))
            
            # Load models
            success = ensemble.load_models(use_8bit=args.use_8bit)
            
            if not success:
                logger.error("Failed to load ensemble models")
                return False
            
            model = ensemble
            tokenizer = ensemble.tokenizers[0]  # Use first tokenizer
        
        else:  # Default to optimized LLM
            logger.info(f"Loading optimized model from {args.model_path}")
            
            # Initialize optimized LLM
            optimized_llm = OptimizedLLM(model_path=args.model_path)
            
            # Load model
            success = optimized_llm.load_model(use_8bit=args.use_8bit)
            
            if not success:
                logger.error("Failed to load optimized model")
                return False
            
            model = optimized_llm
            tokenizer = optimized_llm.tokenizer
        
        model_type = args.model_type
        logger.info(f"Model loaded successfully: {model_type}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Optimized Model API")
    
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the model")
    parser.add_argument("--model-type", type=str, default="optimized",
                        choices=["optimized", "quantized", "tensorrt", "ensemble"],
                        help="Type of model to load")
    parser.add_argument("--model-paths", type=str, default=None,
                        help="Comma-separated list of model paths for ensemble")
    parser.add_argument("--use-8bit", action="store_true",
                        help="Use 8-bit quantization for loading")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the API on")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the API on")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type != "ensemble" and not args.model_path:
        logger.error("--model-path is required for non-ensemble models")
        return 1
    
    if args.model_type == "ensemble" and not args.model_paths:
        logger.error("--model-paths is required for ensemble models")
        return 1
    
    # Load model
    success = load_model(args)
    
    if not success:
        logger.error("Failed to load model")
        return 1
    
    # Run Flask app
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
