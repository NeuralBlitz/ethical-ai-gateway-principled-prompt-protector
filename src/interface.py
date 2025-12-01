# src/interface.py
from src.processor import PromptProcessor

class PrincipledPromptProtector:
    _instance = None # Singleton pattern for efficient model loading

    def __new__(cls, model_path="models/", tokenizer_path="models/"):
        """
        Implements a singleton pattern to ensure the model is loaded only once.
        model_path should be a directory for HuggingFace model loading.
        """
        if cls._instance is None:
            cls._instance = super(PrincipledPromptProtector, cls).__new__(cls)
            # PromptProcessor now expects a directory path for model_name_or_path
            # if loading from push_to_hub saved directory.
            # So, model_path is passed to ContextualRiskClassifier directly.
            cls._instance.processor = PromptProcessor(model_path=model_path, tokenizer_path=tokenizer_path)
        return cls._instance

    def assess_prompt(self, prompt_text: str, risk_threshold: float = 0.5) -> dict:
        """
        Assesses a given prompt for ethical risks using the Principled Prompt Protector.

        Args:
            prompt_text (str): The user's prompt to be assessed.
            risk_threshold (float): The score (0.0-1.0) above which a category is flagged.

        Returns:
            dict: Assessment results including risk scores and guidance.
        """
        return self.processor.detect_prompt_risk(prompt_text, risk_threshold)
