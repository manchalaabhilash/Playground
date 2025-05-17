import ollama
import time
import os
from src.config import OLLAMA_HOST, OLLAMA_MODEL

class LocalLLM:
    def __init__(self, model_name=None, max_retries=5, retry_delay=2):
        self.model_name = model_name or OLLAMA_MODEL
        self.default_max_length = 2048
        self.default_temperature = 0.7
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure Ollama client
        if OLLAMA_HOST:
            ollama.set_host(OLLAMA_HOST)
        
        # Ensure model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Make sure the model is available, with retries for container startup"""
        for attempt in range(self.max_retries):
            try:
                # Check if model exists
                models = ollama.list()
                model_exists = any(model['name'] == self.model_name for model in models.get('models', []))
                
                if not model_exists:
                    print(f"Model {self.model_name} not found, pulling...")
                    ollama.pull(self.model_name)
                
                return True
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt+1}/{self.max_retries}: Ollama not ready yet. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to connect to Ollama after {self.max_retries} attempts: {e}")
                    # Continue anyway, we'll retry when generating responses
        
        return False

    def generate_response(self, prompt, max_length=None, temperature=None, system_message=None):
        """Generate a response from the LLM"""
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': prompt})

        # Try to generate a response with retries
        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        'num_predict': max_length if max_length is not None else self.default_max_length,
                        'temperature': temperature if temperature is not None else self.default_temperature,
                    }
                )
                return response['message']['content']
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt+1}/{self.max_retries}: Error generating response. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    return f"I'm having trouble connecting to the language model. Please try again later. Error: {str(e)}"
