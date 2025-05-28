import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger(__name__)

class AotCompiler:
    """Module for AOT (Ahead-of-Time) compilation of PyTorch models"""
    
    def __init__(self, model=None, example_inputs=None, output_dir="./compiled_models"):
        self.model = model
        self.example_inputs = example_inputs
        self.output_dir = output_dir
        self.compiled_model_path = None
    
    def compile_model(self, dynamic_shapes=None, optimization_level=2):
        """Compile model using torch.export and AOT compilation"""
        try:
            if self.model is None or self.example_inputs is None:
                logger.error("Both model and example_inputs must be provided")
                return False
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set model to eval mode
            self.model.eval()
            
            logger.info("Starting AOT compilation process...")
            
            # Define dynamic dimensions if provided
            dynamic_dims = {}
            if dynamic_shapes:
                for input_idx, shape_dict in dynamic_shapes.items():
                    for dim_idx, dim_info in shape_dict.items():
                        name = dim_info.get("name", f"dim_{input_idx}_{dim_idx}")
                        min_val = dim_info.get("min", 1)
                        max_val = dim_info.get("max", 1024)
                        dynamic_dims[f"{input_idx}.{dim_idx}"] = torch.export.Dim(
                            name, min=min_val, max=max_val
                        )
            
            # Export the model
            logger.info("Exporting model...")
            exported_model = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_dims if dynamic_dims else None
            )
            
            # Save the exported model
            exported_path = os.path.join(self.output_dir, "exported_model.pt")
            torch.save(exported_model, exported_path)
            logger.info(f"Exported model saved to {exported_path}")
            
            # Compile the model using AOT
            logger.info("Compiling model with AOT...")
            
            # Define AOT compilation options
            options = {
                "aot_inductor.output_path": os.path.join(self.output_dir, "model.so"),
                "aot_inductor.optimization_level": optimization_level
            }
            
            # Compile the model
            self.compiled_model_path = torch._export.aot_compile(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_dims if dynamic_dims else None,
                options=options
            )
            
            logger.info(f"Model compiled successfully to {self.compiled_model_path}")
            
            # Generate C++ example code
            self._generate_cpp_example()
            
            return True
        except Exception as e:
            logger.error(f"Error during AOT compilation: {str(e)}")
            return False
    
    def _generate_cpp_example(self):
        """Generate example C++ code for using the compiled model"""
        try:
            # Determine if CUDA is available
            cuda_available = torch.cuda.is_available()
            device_type = "CUDA" if cuda_available else "CPU"
            at_device = "at::kCUDA" if cuda_available else "at::kCPU"
            header_file = "aoti_model_container_runner_cuda.h" if cuda_available else "aoti_model_container_runner.h"
            runner_class = "AOTIModelContainerRunnerCuda" if cuda_available else "AOTIModelContainerRunner"
            
            # Create C++ example file
            cpp_example_path = os.path.join(self.output_dir, "inference_example.cpp")
            
            with open(cpp_example_path, "w") as f:
                f.write(f"""#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/{header_file}>

int main() {{
    c10::InferenceMode mode;

    // Load and run the compiled model
    torch::inductor::{runner_class} runner("{os.path.basename(self.compiled_model_path)}");
    
    // Create input tensor(s)
    std::vector<torch::Tensor> inputs = {{torch::randn({{8, 10}}, {at_device})}};
    
    // Run inference
    std::vector<torch::Tensor> outputs = runner.run(inputs);
    
    // Print results
    std::cout << "Inference result:" << std::endl;
    std::cout << outputs[0] << std::endl;

    return 0;
}}
""")
            
            # Create CMakeLists.txt
            cmake_path = os.path.join(self.output_dir, "CMakeLists.txt")
            
            with open(cmake_path, "w") as f:
                f.write(f"""cmake_minimum_required(VERSION 3.18)
project(model_inference)

find_package(Torch REQUIRED)

add_executable(inference_example inference_example.cpp)
target_link_libraries(inference_example ${{TORCH_LIBRARIES}})

# Set C++ standard
set_property(TARGET inference_example PROPERTY CXX_STANDARD 17)
""")
            
            # Create README with build instructions
            readme_path = os.path.join(self.output_dir, "README.md")
            
            with open(readme_path, "w") as f:
                f.write(f"""# AOT Compiled Model

This directory contains an AOT-compiled model for {device_type} inference.

## Files

- `model.so`: The compiled model shared library
- `inference_example.cpp`: Example C++ code for inference
- `CMakeLists.txt`: CMake configuration for building the example

## Building and Running

1. Install PyTorch C++ libraries
2. Build the example:

```bash
mkdir build
cd build
cmake ..
make
```

3. Run the example:

```bash
./inference_example
```

## Notes

- The model expects input tensors of the shape used during compilation
- For dynamic shapes, refer to the exported model's metadata
""")
            
            logger.info(f"Generated C++ example code in {self.output_dir}")
            return True
        except Exception as e:
            logger.error(f"Error generating C++ example: {str(e)}")
            return False
