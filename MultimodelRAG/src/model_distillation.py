import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class ModelDistiller:
    """Module for knowledge distillation from teacher to student models"""
    
    def __init__(self, teacher_model_name, student_model_name, output_dir="./distilled_models"):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.output_dir = output_dir
        self.teacher_model = None
        self.student_model = None
        self.teacher_tokenizer = None
        self.student_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_models(self, teacher_8bit=False, student_8bit=False):
        """Load teacher and student models"""
        try:
            # Load teacher model
            logger.info(f"Loading teacher model: {self.teacher_model_name}")
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_8bit=teacher_8bit
            )
            self.teacher_model.eval()  # Teacher is always in eval mode
            
            # Load student model
            logger.info(f"Loading student model: {self.student_model_name}")
            self.student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.student_model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_8bit=student_8bit
            )
            
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def prepare_dataset(self, data_path):
        """Prepare dataset for distillation"""
        try:
            # Load data
            with open(data_path, 'r') as f:
                data = [line.strip() for line in f.readlines()]
            
            # Create dataset
            dataset = Dataset.from_dict({"text": data})
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.student_tokenizer(
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
    
    def distill(self, dataset, epochs=3, batch_size=4, learning_rate=5e-5, temperature=2.0):
        """Perform knowledge distillation from teacher to student"""
        if self.teacher_model is None or self.student_model is None:
            logger.error("Teacher and student models must be loaded before distillation")
            return False
        
        try:
            # Define custom loss function for distillation
            class DistillationTrainer(Trainer):
                def __init__(self, teacher_model, temperature=2.0, alpha=0.5, **kwargs):
                    super().__init__(**kwargs)
                    self.teacher_model = teacher_model
                    self.temperature = temperature
                    self.alpha = alpha  # Weight for distillation loss vs standard loss
                
                def compute_loss(self, model, inputs, return_outputs=False):
                    # Standard loss calculation
                    outputs = model(**inputs)
                    student_logits = outputs.logits
                    
                    # Calculate standard cross-entropy loss
                    labels = inputs.get("labels")
                    if labels is not None:
                        loss_ce = F.cross_entropy(
                            student_logits.view(-1, student_logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
                    else:
                        loss_ce = 0.0
                    
                    # Get teacher logits
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(**inputs)
                        teacher_logits = teacher_outputs.logits
                    
                    # Calculate distillation loss (KL divergence)
                    loss_kd = F.kl_div(
                        F.log_softmax(student_logits / self.temperature, dim=-1),
                        F.softmax(teacher_logits / self.temperature, dim=-1),
                        reduction="batchmean"
                    ) * (self.temperature ** 2)
                    
                    # Combine losses
                    loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd
                    
                    return (loss, outputs) if return_outputs else loss
            
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
            
            # Initialize distillation trainer
            trainer = DistillationTrainer(
                teacher_model=self.teacher_model,
                temperature=temperature,
                alpha=0.5,  # Equal weight to CE and KD loss
                model=self.student_model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.student_tokenizer
            )
            
            # Train (distill) model
            trainer.train()
            
            # Save distilled model
            self.student_model.save_pretrained(f"{self.output_dir}/final")
            self.student_tokenizer.save_pretrained(f"{self.output_dir}/final")
            
            logger.info(f"Distillation completed successfully. Model saved to {self.output_dir}/final")
            return True
        except Exception as e:
            logger.error(f"Error during distillation: {str(e)}")
            return False