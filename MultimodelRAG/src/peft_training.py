import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)

class PeftTrainer:
    """Module for parameter-efficient fine-tuning of LLMs"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", output_dir="./peft_models"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, use_8bit=False, use_4bit=False):
        """Load the base model with optional quantization"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure quantization options
            quantization_config = {}
            if use_8bit:
                quantization_config["load_in_8bit"] = True
            elif use_4bit:
                quantization_config["load_in_4bit"] = True
                quantization_config["bnb_4bit_compute_dtype"] = torch.float16
                
            # Load model with quantization if specified
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                **quantization_config
            )
            
            # Prepare model for k-bit training if using quantization
            if use_8bit or use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def apply_peft(self, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        """Apply LoRA adapter for parameter-efficient fine-tuning"""
        try:
            # Define LoRA configuration
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            # Apply LoRA adapter
            self.model = get_peft_model(self.model, peft_config)
            
            # Print trainable parameters info
            self.model.print_trainable_parameters()
            
            logger.info("LoRA adapter applied successfully")
            return True
        except Exception as e:
            logger.error(f"Error applying PEFT: {str(e)}")
            return False
    
    def prepare_dataset(self, data_path):
        """Prepare dataset from custom data"""
        try:
            # Load and format data
            # This is a simplified example - adapt to your data format
            with open(data_path, 'r') as f:
                data = [line.strip() for line in f.readlines()]
            
            # Create dataset
            dataset = Dataset.from_dict({"text": data})
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            return tokenized_dataset
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            return None
    
    def train(self, dataset, epochs=3, batch_size=4, learning_rate=2e-4):
        """Fine-tune the model on custom dataset"""
        if self.model is None or self.tokenizer is None:
            logger.error("Model and tokenizer must be loaded before training")
            return False
        
        try:
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=f"{self.output_dir}/logs",
                logging_steps=100,
                save_strategy="epoch",
                fp16=torch.cuda.is_available()
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer
            )
            
            # Train model
            trainer.train()
            
            # Save model
            self.model.save_pretrained(f"{self.output_dir}/final")
            self.tokenizer.save_pretrained(f"{self.output_dir}/final")
            
            logger.info(f"PEFT model trained and saved to {self.output_dir}/final")
            return True
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False