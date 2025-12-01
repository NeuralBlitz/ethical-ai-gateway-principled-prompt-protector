from src.processor import PromptProcessor

class PrincipledPromptProtector:
    _instance = None

    def __new__(cls, model_path="models/", tokenizer_path="models/"):
        if cls._instance is None:
            cls._instance = super(PrincipledPromptProtector, cls).__new__(cls)
            cls._instance.processor = PromptProcessor(model_path=model_path, tokenizer_path=tokenizer_path)
        return cls._instance

    def assess_prompt(self, prompt_text: str, risk_threshold: float = 0.5) -> dict:
        return self.processor.detect_prompt_risk(prompt_text, risk_threshold)
