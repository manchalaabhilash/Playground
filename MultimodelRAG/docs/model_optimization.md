# Model Optimization Guide

This document provides detailed information about the various model optimization techniques available in the MultimodalRAG system.

## Dynamic Quantization

Dynamic quantization converts weights to INT8 precision while keeping activations in floating-point. This reduces model size and improves inference speed with minimal accuracy loss.

### Usage

```python
from src.dynamic_quantization import DynamicQuantizer

# Initialize quantizer
quantizer = DynamicQuantizer(model_path="path/to/model")

# Apply 8-bit dynamic quantization
quantizer.quantize(quantization_type="dynamic", bits=8)

# Benchmark against original model
quantizer.benchmark(input_text="Your test prompt")
```

### CLI Usage

```bash
python scripts/train_optimize_llm.py quantize --model "path/to/model" --type dynamic --bits 8 --benchmark
```

## Model Ensembling

Model ensembling combines predictions from multiple models to improve accuracy and robustness.

### Usage

```python
from src.model_ensemble import ModelEnsemble

# Initialize ensemble with multiple models
ensemble = ModelEnsemble(model_paths=["model1", "model2", "model3"])

# Load models
ensemble.load_models()

# Generate text using ensemble
generated_text = ensemble.ensemble_generate(
    "Your prompt here", 
    max_length=100, 
    ensemble_method="mean"
)
```

### CLI Usage

```bash
python scripts/train_optimize_llm.py ensemble --models "model1" "model2" "model3" --prompt "Your test prompt" --method mean
```

## Knowledge Distillation

Knowledge distillation transfers knowledge from a larger teacher model to a smaller student model.

### Usage

```python
from src.model_distillation import ModelDistiller

# Initialize distiller
distiller = ModelDistiller(
    teacher_model_name="large-model",
    student_model_name="small-model"
)

# Load models
distiller.load_models()

# Prepare dataset
dataset = distiller.prepare_dataset("path/to/data.txt")

# Perform distillation
distiller.distill(
    dataset=dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    temperature=2.0
)
```

### CLI Usage

```bash
python scripts/train_optimize_llm.py distill --teacher "large-model" --student "small-model" --data "path/to/data.txt" --epochs 3
```

## TensorRT Optimization

TensorRT optimization accelerates inference by optimizing the model for NVIDIA GPUs.

### Usage

```python
from src.tensorrt_optimization import TensorRTOptimizer

# Initialize optimizer
optimizer = TensorRTOptimizer(model_path="path/to/model")

# Optimize with TensorRT
optimizer.optimize_with_tensorrt(precision="fp16")

# Benchmark against original model
optimizer.benchmark(input_text="Your test prompt")
```

### CLI Usage

```bash
python scripts/train_optimize_llm.py tensorrt --model "path/to/model" --precision fp16 --benchmark
```

## AOT Compilation

AOT (Ahead-of-Time) compilation prepares models for deployment in non-Python environments.

### Usage

```python
from src.aot_compiler import AotCompiler
import torch

# Create example inputs
example_inputs = (torch.randn(1, 10),)

# Initialize compiler
compiler = AotCompiler(model=your_model, example_inputs=example_inputs)

# Define dynamic shapes (optional)
dynamic_shapes = {
    0: {
        0: {
            "name": "batch",
            "min": 1,
            "max": 64
        }
    }
}

# Compile model
compiler.compile_model(dynamic_shapes=dynamic_shapes)
```

### CLI Usage

```bash
python scripts/train_optimize_llm.py compile --model "path/to/model" --dynamic
```

## Model Pruning

Model pruning removes less important weights to reduce model size and improve inference speed.

### Usage

```python
from src.model_pruning import ModelPruner

# Initialize pruner
pruner = ModelPruner(model_path="path/to/model")

# Prune model using magnitude-based pruning
pruner.prune_model(
    method="magnitude",  # Options: "magnitude", "random", "structured"
    sparsity=0.3,        # 30% of weights will be pruned
    target_modules=["query", "key", "value", "dense"]  # Target specific modules
)

# Benchmark against original model
pruner.benchmark(input_text="Your test prompt")
```

### CLI Usage

```bash
python scripts/train_optimize_llm.py prune --model "path/to/model" --method magnitude --sparsity 0.3 --benchmark
```

### Pruning Methods

1. **Magnitude Pruning**: Removes weights with smallest absolute values
2. **Random Pruning**: Randomly removes weights based on sparsity level
3. **Structured Pruning**: Removes entire neurons/channels based on their importance

### Target Modules

For transformer-based models, common target modules include:
- `query`, `key`, `value`: Attention mechanism components
- `dense`: Output projection in attention blocks
- `fc1`, `fc2`: Feed-forward network components

## Performance Comparison

| Optimization Technique | Size Reduction | Speed Improvement | Accuracy Impact |
|------------------------|----------------|-------------------|-----------------|
| Dynamic Quantization (8-bit) | ~75% | 1.5-2x | Minimal |
| Dynamic Quantization (4-bit) | ~87% | 2-3x | Moderate |
| Knowledge Distillation | Varies | Varies | Moderate |
| Model Ensembling | Negative | Negative | Improved |
| TensorRT Optimization | None | 2-5x | None |
| AOT Compilation | None | 1.5-3x | None |
| Magnitude Pruning (30%) | ~30% | 1.2-1.5x | Minimal |
| Magnitude Pruning (50%) | ~50% | 1.5-2x | Moderate |
| Structured Pruning (30%) | ~30% | 1.3-1.8x | Moderate |
