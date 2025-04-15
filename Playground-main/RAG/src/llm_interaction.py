import ollama

class LocalLLM:
    def __init__(self, model_name="llama3", default_max_length=512, default_temperature=0.7):
        self.model_name = model_name
        self.default_max_length = default_max_length
        self.default_temperature = default_temperature

    def generate_response(self, prompt, max_length=None, temperature=None, system_message=None):
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': prompt})

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
        except ollama.OllamaAPIError as e:
            return f"Error communicating with Ollama: {e}"