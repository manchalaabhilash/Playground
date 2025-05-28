#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append('.')

from src.llm_training import LLMTrainer
from src.optimized_llm import OptimizedLLM
from src.peft_training import PeftTrainer
from src.aot_compiler import AotCompiler
from src.model_ensemble import ModelEnsemble
from src.model_distillation import ModelDistiller
from src.tensorrt_optimization import TensorRTOptimizer
from src.dynamic_quantization import DynamicQuantizer
from src.model_pruning import ModelPruner
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train and optimize LLMs for MultimodalRAG")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add ensemble command
    ensemble_parser = subparsers.add_parser("ensemble", help="Create and use model ensembles")
    ensemble_parser.add_argument("--models", type=str, nargs="+", required=True, 
                               help="Paths to models for ensemble")
    ensemble_parser.add_argument("--output", type=str, default="./ensemble_models", 
                               help="Output directory")
    ensemble_parser.add_argument("--prompt", type=str, default="Hello, how are you?", 
                               help="Prompt for generation")
    ensemble_parser.add_argument("--max-length", type=int, default=100, 
                               help="Maximum generation length")
    ensemble_parser.add_argument("--method", type=str, default="mean", choices=["mean", "max"], 
                               help="Ensemble method (mean or max)")
    ensemble_parser.add_argument("--use-8bit", action="store_true", 
                               help="Load models in 8-bit precision")
    
    # Add distillation command
    distill_parser = subparsers.add_parser("distill", help="Distill knowledge from teacher to student model")
    distill_parser.add_argument("--teacher", type=str, required=True, 
                               help="Path to teacher model")
    distill_parser.add_argument("--student", type=str, required=True, 
                               help="Path to student model")
    distill_parser.add_argument("--data", type=str, required=True, 
                               help="Path to training data")
    distill_parser.add_argument("--output", type=str, default="./distilled_models", 
                               help="Output directory")
    distill_parser.add_argument("--epochs", type=int, default=3, 
                               help="Number of training epochs")
    distill_parser.add_argument("--batch-size", type=int, default=4, 
                               help="Training batch size")
    distill_parser.add_argument("--learning-rate", type=float, default=5e-5, 
                               help="Learning rate")
    distill_parser.add_argument("--temperature", type=float, default=2.0, 
                               help="Distillation temperature")
    distill_parser.add_argument("--teacher-8bit", action="store_true", 
                               help="Load teacher model in 8-bit precision")
    distill_parser.add_argument("--student-8bit", action="store_true", 
                               help="Load student model in 8-bit precision")
    
    # Add TensorRT command
    tensorrt_parser = subparsers.add_parser("tensorrt", help="Optimize model with TensorRT")
    tensorrt_parser.add_argument("--model", type=str, required=True, 
                               help="Path to model")
    tensorrt_parser.add_argument("--output", type=str, default="./tensorrt_models", 
                               help="Output directory")
    tensorrt_parser.add_argument("--batch-size", type=int, default=1, 
                               help="Batch size for optimization")
    tensorrt_parser.add_argument("--sequence-length", type=int, default=128, 
                               help="Sequence length for optimization")
    tensorrt_parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], 
                               help="Precision for optimization")
    tensorrt_parser.add_argument("--benchmark", action="store_true", 
                               help="Run benchmarks after optimization")
    tensorrt_parser.add_argument("--benchmark-text", type=str, default="Hello, how are you?", 
                               help="Text for benchmarking")
    tensorrt_parser.add_argument("--benchmark-runs", type=int, default=10, 
                               help="Number of benchmark runs")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune an LLM on custom data")
    train_parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model to fine-tune")
    train_parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    train_parser.add_argument("--output", type=str, default="./trained_models", help="Output directory for trained model")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    train_parser.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="8bit", 
                             help="Quantization for training")
    
    # PEFT command
    peft_parser = subparsers.add_parser("peft", help="Parameter-efficient fine-tuning of an LLM")
    peft_parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model to fine-tune")
    peft_parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    peft_parser.add_argument("--output", type=str, default="./peft_models", help="Output directory for trained model")
    peft_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    peft_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    peft_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    peft_parser.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="8bit", 
                            help="Quantization for training")
    peft_parser.add_argument("--lora-r", type=int, default=8, help="LoRA attention dimension")
    peft_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    peft_parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout probability")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize an LLM for inference")
    optimize_parser.add_argument("--model", type=str, required=True, help="Model to optimize")
    optimize_parser.add_argument("--output", type=str, default="./optimized_models", help="Output directory")
    optimize_parser.add_argument("--quantize", choices=["none", "8bit", "4bit", "gptq"], default="8bit", 
                               help="Quantization method")
    optimize_parser.add_argument("--onnx", action="store_true", help="Export to ONNX format")
    optimize_parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark after optimization")
    
    # Add AOT compile command
    aot_parser = subparsers.add_parser("aot", help="Ahead-of-time compile a model")
    aot_parser.add_argument("--model", type=str, required=True, help="Model to compile")
    aot_parser.add_argument("--output", type=str, default="./compiled_models", help="Output directory")
    aot_parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes")
    aot_parser.add_argument("--batch-min", type=int, default=1, help="Minimum batch size for dynamic shapes")
    aot_parser.add_argument("--batch-max", type=int, default=32, help="Maximum batch size for dynamic shapes")
    aot_parser.add_argument("--cpp", action="store_true", help="Generate C++ inference code")
    
    # Add quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize model for efficient inference")
    quantize_parser.add_argument("--model", type=str, required=True, 
                               help="Path to model")
    quantize_parser.add_argument("--output", type=str, default="./quantized_models", 
                               help="Output directory")
    quantize_parser.add_argument("--type", type=str, default="dynamic", choices=["dynamic", "static"], 
                               help="Quantization type")
    quantize_parser.add_argument("--bits", type=int, default=8, choices=[4, 8], 
                               help="Quantization bit width")
    quantize_parser.add_argument("--benchmark", action="store_true", 
                               help="Run benchmarks after quantization")
    quantize_parser.add_argument("--benchmark-text", type=str, default="Hello, how are you?", 
                               help="Text for benchmarking")
    quantize_parser.add_argument("--benchmark-runs", type=int, default=10, 
                               help="Number of benchmark runs")
    
    # Add prune command
    prune_parser = subparsers.add_parser("prune", help="Prune model to reduce size and improve inference speed")
    prune_parser.add_argument("--model", type=str, required=True, 
                            help="Path to model")
    prune_parser.add_argument("--output", type=str, default="./pruned_models", 
                            help="Output directory")
    prune_parser.add_argument("--method", type=str, default="magnitude", 
                            choices=["magnitude", "random", "structured"], 
                            help="Pruning method")
    prune_parser.add_argument("--sparsity", type=float, default=0.3, 
                            help="Sparsity level (0.0-1.0)")
    prune_parser.add_argument("--target-modules", type=str, nargs="+", 
                            default=["query", "key", "value", "dense", "fc1", "fc2"], 
                            help="Target modules for pruning")
    prune_parser.add_argument("--benchmark", action="store_true", 
                            help="Run benchmarks after pruning")
    prune_parser.add_argument("--benchmark-text", type=str, default="Hello, how are you?", 
                            help="Text for benchmarking")
    prune_parser.add_argument("--benchmark-runs", type=int, default=10, 
                            help="Number of benchmark runs")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "train":
        # Train LLM on custom data
        logger.info(f"Training model {args.model} on data {args.data}")
        
        trainer = LLMTrainer(model_name=args.model, output_dir=args.output)
        
        # Load model with appropriate quantization
        use_8bit = args.quantize == "8bit"
        use_4bit = args.quantize == "4bit"
        trainer.load_model(use_8bit=use_8bit, use_4bit=use_4bit)
        
        # Prepare dataset
        dataset = trainer.prepare_dataset(args.data)
        if dataset is None:
            logger.error("Failed to prepare dataset")
            return 1
        
        # Train model
        success = trainer.train(
            dataset, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        if success:
            logger.info(f"Training completed successfully. Model saved to {args.output}/final")
            return 0
        else:
            logger.error("Training failed")
            return 1
    
    elif args.command == "peft":
        # Parameter-efficient fine-tuning
        logger.info(f"PEFT fine-tuning model {args.model} on data {args.data}")
        
        trainer = PeftTrainer(model_name=args.model, output_dir=args.output)
        
        # Load model with appropriate quantization
        use_8bit = args.quantize == "8bit"
        use_4bit = args.quantize == "4bit"
        trainer.load_model(use_8bit=use_8bit, use_4bit=use_4bit)
        
        # Apply PEFT adapter
        success = trainer.apply_peft(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        
        if not success:
            logger.error("Failed to apply PEFT adapter")
            return 1
        
        # Prepare dataset
        dataset = trainer.prepare_dataset(args.data)
        if dataset is None:
            logger.error("Failed to prepare dataset")
            return 1
        
        # Train model
        success = trainer.train(
            dataset, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        if success:
            logger.info(f"PEFT training completed successfully. Model saved to {args.output}/final")
            return 0
        else:
            logger.error("PEFT training failed")
            return 1
            
    elif args.command == "optimize":
        # Optimize LLM for inference
        logger.info(f"Optimizing model {args.model} with {args.quantize} quantization")
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize optimized LLM
        llm = OptimizedLLM(model_name=args.model)
        
        # Load with specified quantization
        success = llm.load_model(quantization=args.quantize)
        if not success:
            logger.error("Failed to load model for optimization")
            return 1
        
        # Export to ONNX if requested
        if args.onnx:
            logger.info("Exporting model to ONNX format")
            onnx_path = llm.export_to_onnx(output_dir=f"{args.output}/onnx")
            if not onnx_path:
                logger.error("Failed to export model to ONNX format")
            else:
                logger.info(f"Model exported to ONNX at {onnx_path}")
        
        # Run benchmark if requested
        if args.benchmark:
            logger.info("Running inference benchmark")
            
            import time
            
            # Warm-up
            llm.generate("Hello, world!", max_length=20)
            
            # Benchmark
            prompts = [
                "Explain the concept of machine learning in simple terms.",
                "What are the key differences between supervised and unsupervised learning?",
                "How does a neural network work?",
                "Explain the concept of transfer learning in natural language processing."
            ]
            
            total_time = 0
            total_tokens = 0
            
            for prompt in prompts:
                start_time = time.time()
                response = llm.generate(prompt, max_length=200)
                end_time = time.time()
                
                # Estimate token count (rough approximation)
                prompt_tokens = len(prompt.split())
                response_tokens = len(response.split())
                total_tokens += prompt_tokens + response_tokens
                
                # Calculate time
                elapsed = end_time - start_time
                total_time += elapsed
                
                logger.info(f"Prompt: {prompt[:30]}...")
                logger.info(f"Response time: {elapsed:.2f}s, Estimated tokens: {prompt_tokens + response_tokens}")
            
            # Calculate average throughput
            throughput = total_tokens / total_time
            logger.info(f"Average throughput: {throughput:.2f} tokens/second")
            
            # Save benchmark results
            with open(f"{args.output}/benchmark_results.txt", "w") as f:
                f.write(f"Model: {args.model}\n")
                f.write(f"Quantization: {args.quantize}\n")
                f.write(f"Average throughput: {throughput:.2f} tokens/second\n")
                f.write(f"Total time: {total_time:.2f}s\n")
                f.write(f"Total tokens: {total_tokens}\n")
            
            logger.info(f"Benchmark results saved to {args.output}/benchmark_results.txt")
    
    elif args.command == "aot":
        # AOT compilation
        logger.info(f"AOT compiling model {args.model}")
        
        try:
            # Load model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
            
            # Create example inputs
            example_inputs = (torch.randint(0, tokenizer.vocab_size, (1, 10)).to(model.device),)
            
            # Initialize compiler
            compiler = AotCompiler(model=model, example_inputs=example_inputs, output_dir=args.output)
            
            # Define dynamic shapes if requested
            dynamic_shapes = None
            if args.dynamic:
                dynamic_shapes = {
                    0: {
                        0: {
                            "name": "batch",
                            "min": args.batch_min,
                            "max": args.batch_max
                        }
                    }
                }
            
            # Compile model
            success = compiler.compile_model(dynamic_shapes=dynamic_shapes)
            
            if not success:
                logger.error("AOT compilation failed")
                return 1
            
            # Generate C++ inference code if requested
            if args.cpp:
                success = compiler.generate_cpp_inference_code()
                if not success:
                    logger.error("Failed to generate C++ inference code")
                    return 1
            
            logger.info(f"AOT compilation completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Error during AOT compilation: {str(e)}")
            return 1
    elif args.command == "ensemble":
        logger.info(f"Creating ensemble with {len(args.models)} models")
        
        try:
            # Initialize ensemble
            ensemble = ModelEnsemble(model_paths=args.models, output_dir=args.output)
            
            # Load models
            success = ensemble.load_models(use_8bit=args.use_8bit)
            if not success:
                logger.error("Failed to load ensemble models")
                return 1
            
            # Generate text
            logger.info(f"Generating text with prompt: {args.prompt}")
            generated_text = ensemble.ensemble_generate(
                args.prompt, 
                max_length=args.max_length, 
                ensemble_method=args.method
            )
            
            if generated_text:
                logger.info(f"Generated text: {generated_text}")
                
                # Save output
                os.makedirs(args.output, exist_ok=True)
                with open(os.path.join(args.output, "ensemble_output.txt"), "w") as f:
                    f.write(f"Prompt: {args.prompt}\n\n")
                    f.write(f"Generated text: {generated_text}\n")
                
                logger.info(f"Output saved to {os.path.join(args.output, 'ensemble_output.txt')}")
                return 0
            else:
                logger.error("Failed to generate text with ensemble")
                return 1
        except Exception as e:
            logger.error(f"Error during ensemble operation: {str(e)}")
            return 1
    elif args.command == "distill":
        logger.info(f"Distilling from teacher {args.teacher} to student {args.student}")
        
        try:
            # Initialize distiller
            distiller = ModelDistiller(
                teacher_model_name=args.teacher,
                student_model_name=args.student,
                output_dir=args.output
            )
            
            # Load models
            success = distiller.load_models(
                teacher_8bit=args.teacher_8bit,
                student_8bit=args.student_8bit
            )
            if not success:
                logger.error("Failed to load teacher and student models")
                return 1
            
            # Prepare dataset
            dataset = distiller.prepare_dataset(args.data)
            if dataset is None:
                logger.error("Failed to prepare dataset for distillation")
                return 1
            
            # Perform distillation
            success = distiller.distill(
                dataset=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                temperature=args.temperature
            )
            
            if success:
                logger.info(f"Distillation completed successfully")
                return 0
            else:
                logger.error("Distillation failed")
                return 1
        except Exception as e:
            logger.error(f"Error during distillation: {str(e)}")
            return 1
    elif args.command == "tensorrt":
        logger.info(f"Optimizing model {args.model} with TensorRT")
        
        try:
            # Initialize optimizer
            optimizer = TensorRTOptimizer(
                model_path=args.model,
                output_dir=args.output
            )
            
            # Optimize model
            success = optimizer.optimize_with_tensorrt(
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                precision=args.precision
            )
            
            if not success:
                logger.error("TensorRT optimization failed")
                return 1
            
            # Run benchmark if requested
            if args.benchmark:
                logger.info("Running benchmarks...")
                success = optimizer.benchmark(
                    input_text=args.benchmark_text,
                    num_runs=args.benchmark_runs
                )
                
                if not success:
                    logger.error("Benchmarking failed")
                    return 1
            
            logger.info(f"TensorRT optimization completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Error during TensorRT optimization: {str(e)}")
            return 1
    elif args.command == "quantize":
        logger.info(f"Quantizing model {args.model} with {args.bits}-bit {args.type} quantization")
        
        try:
            # Initialize quantizer
            quantizer = DynamicQuantizer(
                model_path=args.model,
                output_dir=args.output
            )
            
            # Quantize model
            success = quantizer.quantize(
                quantization_type=args.type,
                bits=args.bits
            )
            
            if not success:
                logger.error("Quantization failed")
                return 1
            
            # Run benchmark if requested
            if args.benchmark:
                logger.info("Running benchmarks...")
                success = quantizer.benchmark(
                    input_text=args.benchmark_text,
                    num_runs=args.benchmark_runs
                )
                
                if not success:
                    logger.error("Benchmarking failed")
                    return 1
            
            logger.info(f"Quantization completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Error during quantization: {str(e)}")
            return 1
    elif args.command == "prune":
        logger.info(f"Pruning model {args.model} with {args.method} method and {args.sparsity} sparsity")
        
        try:
            # Initialize pruner
            pruner = ModelPruner(
                model_path=args.model,
                output_dir=args.output
            )
            
            # Prune model
            success = pruner.prune_model(
                method=args.method,
                sparsity=args.sparsity,
                target_modules=args.target_modules
            )
            
            if not success:
                logger.error("Pruning failed")
                return 1
            
            # Run benchmark if requested
            if args.benchmark:
                logger.info("Running benchmarks...")
                success = pruner.benchmark(
                    input_text=args.benchmark_text,
                    num_runs=args.benchmark_runs
                )
                
                if not success:
                    logger.error("Benchmarking failed")
                    return 1
            
            logger.info(f"Pruning completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Error during pruning: {str(e)}")
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
