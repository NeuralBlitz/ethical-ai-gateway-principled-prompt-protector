# src/processor.py
import torch
from src.model import ContextualRiskClassifier

class PromptProcessor:
    def __init__(self, model_path="models/crc_model_v1.pth", tokenizer_path="models/"):
        # Load model with potentially custom tokenizer path
        # Assuming model_path also contains the config.json and tokenizer files if push_to_hub used save_pretrained
        try:
            # Try loading directly from path if it's a HF-compatible directory
            self.crc_classifier = ContextualRiskClassifier(model_name_or_path=model_path, num_labels=5)
            # If loaded from save_pretrained dir, weights are already there
        except Exception:
            # Fallback for custom .pth file loading
            self.crc_classifier = ContextualRiskClassifier(model_name_or_path=tokenizer_path, num_labels=5)
            self.crc_classifier.load_weights(model_path) # Load specific .pth weights
        
        self.tokenizer = self.crc_classifier.get_tokenizer()
        self.model = self.crc_classifier.get_model()
        self.id2label = self.crc_classifier.id2label # Get the label mappings

    def detect_prompt_risk(self, prompt_text: str, risk_threshold: float = 0.5) -> dict:
        """
        Analyzes a user prompt for ethical risks based on the loaded CRC model.

        Args:
            prompt_text (str): The raw text of the user's prompt.
            risk_threshold (float): The score above which a category is considered "flagged".

        Returns:
            dict: A dictionary containing assessment results.
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        self.model.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()

        # Map probabilities to ethical category names
        risk_scores = {self.id2label[i]: prob for i, prob in enumerate(probabilities)}
        
        # Calculate overall risk (e.g., the maximum score across all problematic categories)
        overall_risk = max(risk_scores.values())

        # Determine which categories exceed the risk threshold
        flagged_categories = [cat for cat, score in risk_scores.items() if score > risk_threshold]
        
        # Placeholder for more advanced keyword/explainability integration
        flagged_keywords = [] 

        # Generate simplified guidance based on the highest risk category
        guidance = "Prompt assessed as ethically compliant."
        if overall_risk > risk_threshold:
            highest_risk_category = max(risk_scores, key=risk_scores.get)
            if "harm_prevention" in highest_risk_category:
                guidance = "Warning: This prompt may lead to harmful content. Consider rephrasing to be constructive and safe (Φ1)."
            elif "fairness_discrimination" in highest_risk_category:
                guidance = "Warning: This prompt may exhibit bias. Please ensure language is inclusive and non-discriminatory (Φ7)."
            elif "privacy_violation" in highest_risk_category:
                guidance = "Warning: This prompt may violate privacy. Avoid requesting sensitive Personally Identifiable Information (PII) (Φ10)."
            elif "transparency_deception" in highest_risk_category:
                guidance = "Warning: This prompt may promote deception. Ensure clear distinction between AI and human content (Φ4)."
            elif "accountability_misuse" in highest_risk_category:
                guidance = "Warning: This prompt may encourage irresponsible AI use. Ensure ethical application of AI capabilities (Φ5)."
            # Add general guidance if multiple categories flagged
            if len(flagged_categories) > 1:
                guidance = "Multiple ethical concerns detected. Review flagged categories for responsible rephrasing."

        return {
            "prompt": prompt_text,
            "overall_risk_score": overall_risk,
            "flagged_categories": flagged_categories,
            "risk_details": risk_scores,
            "suggested_guidance": guidance,
            "is_flagged": overall_risk > risk_threshold
        }
