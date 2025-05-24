"""
LLM interaction module for the Multimodal RAG system.
Handles communication with multimodal LLMs.
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any, Optional
import base64

from src.config import LLM_MODEL, LLM_API_KEY

class MultimodalLLM:
    """
    Multimodal LLM interaction class.
    Supports text and image inputs for generating responses.
    """
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the Multimodal LLM"""
        self.model_name = model_name or LLM_MODEL
        self.api_key = api_key or LLM_API_KEY or os.environ.get("LLM_API_KEY")
        
        # Set API endpoint based on model
        if "gpt" in self.model_name.lower():
            self.api_endpoint = "https://api.openai.com/v1/chat/completions"
        elif "claude" in self.model_name.lower():
            self.api_endpoint = "https://api.anthropic.com/v1/messages"
        else:
            # Default to OpenAI
            self.api_endpoint = "https://api.openai.com/v1/chat/completions"
    
    def generate_response(self, question: str, relevant_texts: List[Any] = None, relevant_images: List[Any] = None) -> str:
        """Generate a response using the multimodal LLM"""
        if not self.api_key:
            return "Error: LLM API key not provided. Please set the LLM_API_KEY environment variable."
        
        # Prepare the prompt with context
        prompt = self._prepare_prompt(question, relevant_texts, relevant_images)
        
        # Prepare the API request
        if "gpt" in self.model_name.lower():
            response = self._call_openai_api(prompt, relevant_images)
        elif "claude" in self.model_name.lower():
            response = self._call_anthropic_api(prompt, relevant_images)
        else:
            # Default to OpenAI
            response = self._call_openai_api(prompt, relevant_images)
        
        return response
    
    def _prepare_prompt(self, question: str, relevant_texts: List[Any] = None, relevant_images: List[Any] = None) -> str:
        """Prepare the prompt with context"""
        prompt = "You are a helpful assistant that can understand both text and images.\n\n"
        
        # Add text context if available
        if relevant_texts and len(relevant_texts) > 0:
            prompt += "Here is some relevant information from documents:\n\n"
            for i, doc in enumerate(relevant_texts):
                content = getattr(doc, "page_content", str(doc))
                prompt += f"Document {i+1}:\n{content}\n\n"
        
        # Add image context if available
        if relevant_images and len(relevant_images) > 0:
            prompt += f"I'm also providing {len(relevant_images)} relevant images.\n\n"
        
        # Add the question
        prompt += f"Question: {question}\n\n"
        prompt += "Please provide a comprehensive answer based on the provided information."
        
        return prompt
    
    def _call_openai_api(self, prompt: str, relevant_images: List[Any] = None) -> str:
        """Call the OpenAI API with the prompt and images"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prepare the messages
        messages = [{"role": "system", "content": "You are a helpful assistant that can understand both text and images."}]
        
        # Add images if available
        if relevant_images and len(relevant_images) > 0:
            content = []
            
            # Add text part
            content.append({"type": "text", "text": prompt})
            
            # Add image parts
            for img in relevant_images:
                image_path = img.get("image_path", "")
                if os.path.exists(image_path):
                    try:
                        with open(image_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            })
                    except Exception as e:
                        print(f"Error loading image {image_path}: {str(e)}")
            
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": prompt})
        
        # Prepare the request data
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            response_json = response.json()
            
            if response.status_code == 200:
                return response_json["choices"][0]["message"]["content"]
            else:
                error_message = response_json.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_message}"
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    def _call_anthropic_api(self, prompt: str, relevant_images: List[Any] = None) -> str:
        """Call the Anthropic API with the prompt and images"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Prepare the messages
        messages = []
        
        # Add images if available
        if relevant_images and len(relevant_images) > 0:
            content = []
            
            # Add text part
            content.append({"type": "text", "text": prompt})
            
            # Add image parts
            for img in relevant_images:
                image_path = img.get("image_path", "")
                if os.path.exists(image_path):
                    try:
                        with open(image_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            })
                    except Exception as e:
                        print(f"Error loading image {image_path}: {str(e)}")
            
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        
        # Prepare the request data
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            response_json = response.json()
            
            if response.status_code == 200:
                return response_json["content"][0]["text"]
            else:
                error_message = response_json.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_message}"
        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"
