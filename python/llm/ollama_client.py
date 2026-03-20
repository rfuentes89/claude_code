import ollama

class OllamaClient:
    def __init__(self, model="deepseek-coder"):
        self.model = model

    def ask(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in MRI scientist, Python, MATLAB, and Julia."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]