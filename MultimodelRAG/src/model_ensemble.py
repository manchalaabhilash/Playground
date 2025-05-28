import os
import torch
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelEnsemble:
    """Module for creating and using ensembles of language models"""
    
    def __init__(self, model_paths: List[str], output_dir: str = "./ensemble_models"):
        self.model_paths = model_paths
        self.output_dir = output_dir
        self.models = []
        self.tokenizers = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_models(self, use_8bit: bool = False) -> bool:
        """Load all models in the ensemble"""
        try:
            logger.info(f"Loading {len(self.model_paths)} models for ensemble")
            
            for i, model_path in enumerate(self.model_paths):
                logger.info(f"Loading model {i+1}/{len(self.model_paths)}: {model_path}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.tokenizers.append(tokenizer)
                
                # Load model
                if use_8bit:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        load_in_8bit=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto"
                    )
                
                # Set to eval mode
                model.eval()
                self.models.append(model)
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def ensemble_generate(self, prompt: str, max_length: int = 100, 
                         ensemble_method: str = "mean", temperature: float = 1.0,
                         top_p: float = 0.9, top_k: int = 50) -> Optional[str]:
        """Generate text using ensemble of models"""
        try:
            if not self.models or not self.tokenizers:
                logger.error("Models not loaded. Call load_models() first.")
                return None
            
            logger.info(f"Generating text with ensemble using {ensemble_method} method")
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Tokenize prompt for each model
            inputs = []
            for tokenizer in self.tokenizers:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                inputs.append(input_ids)
            
            # Generate from each model separately
            individual_outputs = []
            individual_texts = []
            
            for i, (model, tokenizer, input_ids) in enumerate(zip(self.models, self.tokenizers, inputs)):
                logger.info(f"Generating with model {i+1}/{len(self.models)}")
                
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_length=max_length,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                individual_outputs.append(output)
                
                # Decode output
                output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                individual_texts.append(output_text)
                
                logger.info(f"Model {i+1} output: {output_text[:50]}...")
            
            # Save individual outputs
            with open(os.path.join(self.output_dir, "individual_outputs.txt"), "w") as f:
                for i, text in enumerate(individual_texts):
                    f.write(f"Model {i+1} ({self.model_paths[i]}):\n")
                    f.write(f"{text}\n\n")
            
            # Perform ensemble generation
            if ensemble_method == "voting":
                # Simple voting-based ensemble (most common next token)
                ensemble_output = self._voting_ensemble(individual_outputs, prompt)
            elif ensemble_method == "max":
                # Max confidence ensemble
                ensemble_output = self._max_confidence_ensemble(individual_outputs, prompt)
            else:  # Default to mean
                # Mean logits ensemble
                ensemble_output = self._mean_logits_ensemble(individual_outputs, prompt)
            
            logger.info(f"Ensemble output: {ensemble_output[:50]}...")
            
            # Save ensemble output
            with open(os.path.join(self.output_dir, "ensemble_output.txt"), "w") as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Ensemble method: {ensemble_method}\n\n")
                f.write(f"Ensemble output:\n{ensemble_output}\n")
            
            return ensemble_output
        except Exception as e:
            logger.error(f"Error during ensemble generation: {str(e)}")
            return None
    
    def _voting_ensemble(self, individual_outputs, prompt):
        """Ensemble method based on token voting"""
        # This is a simplified implementation
        # In practice, you would need to align tokens across different tokenizers
        
        # For simplicity, we'll just use the most common output
        all_texts = []
        for i, output in enumerate(individual_outputs):
            text = self.tokenizers[i].decode(output.sequences[0], skip_special_tokens=True)
            all_texts.append(text)
        
        # Remove prompt from the beginning of each text
        for i in range(len(all_texts)):
            if all_texts[i].startswith(prompt):
                all_texts[i] = all_texts[i][len(prompt):]
        
        # Simple approach: split into sentences and vote
        all_sentences = []
        for text in all_texts:
            sentences = text.split('. ')
            all_sentences.append(sentences)
        
        # Find the maximum number of sentences
        max_sentences = max(len(sentences) for sentences in all_sentences)
        
        # Vote for each sentence position
        ensemble_sentences = []
        for i in range(max_sentences):
            sentence_candidates = []
            for sentences in all_sentences:
                if i < len(sentences):
                    sentence_candidates.append(sentences[i])
            
            # Count occurrences of each sentence
            sentence_counts = {}
            for sentence in sentence_candidates:
                if sentence in sentence_counts:
                    sentence_counts[sentence] += 1
                else:
                    sentence_counts[sentence] = 1
            
            # Get the most common sentence
            most_common_sentence = max(sentence_counts.items(), key=lambda x: x[1])[0]
            ensemble_sentences.append(most_common_sentence)
        
        # Join sentences
        ensemble_text = '. '.join(ensemble_sentences)
        if not ensemble_text.endswith('.'):
            ensemble_text += '.'
        
        return prompt + ensemble_text
    
    def _max_confidence_ensemble(self, individual_outputs, prompt):
        """Ensemble method based on maximum confidence scores"""
        # This implementation selects tokens with highest confidence
        
        # For simplicity, we'll use the model with highest average confidence
        avg_scores = []
        for output in individual_outputs:
            # Calculate average log probability
            scores = output.scores
            avg_score = sum(torch.max(score, dim=-1)[0].mean().item() for score in scores) / len(scores)
            avg_scores.append(avg_score)
        
        # Get the model with highest average confidence
        best_model_idx = np.argmax(avg_scores)
        best_output = individual_outputs[best_model_idx]
        
        # Decode output from best model
        best_text = self.tokenizers[best_model_idx].decode(
            best_output.sequences[0], 
            skip_special_tokens=True
        )
        
        return best_text
    
    def _mean_logits_ensemble(self, individual_outputs, prompt):
        """Ensemble method based on averaging logits"""
        # This is a more complex implementation that requires
        # generating text token-by-token with averaged logits
        
        # For simplicity in this demo, we'll implement a basic version
        # that generates a new sequence using the tokenizer from the first model
        
        tokenizer = self.tokenizers[0]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Maximum length for generation
        max_new_tokens = 100
        
        # Start with input ids
        current_ids = input_ids
        
        for _ in range(max_new_tokens):
            # Get logits from each model for the current sequence
            all_logits = []
            
            for model in self.models:
                with torch.no_grad():
                    outputs = model(current_ids)
                    # Get logits for the last token
                    logits = outputs.logits[:, -1, :]
                    all_logits.append(logits)
            
            # Average the logits
            mean_logits = torch.mean(torch.stack(all_logits), dim=0)
            
            # Apply temperature and sampling
            temperature = 0.7
            mean_logits = mean_logits / temperature
            
            # Sample from the distribution
            probs = torch.nn.functional.softmax(mean_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # Check if we've generated an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Decode the generated sequence
        output_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        return output_text
