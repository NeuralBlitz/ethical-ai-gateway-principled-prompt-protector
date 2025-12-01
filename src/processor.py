import torch
from src.model import ContextualRiskClassifier

class PromptProcessor:
    def __init__(self, model_path="models/crc_model_v1.pth", tokenizer_path="models/"):
        try:
            self.crc_classifier = ContextualRiskClassifier(model_name_or_path=model_path, num_labels=5)
        except Exception:
            self.crc_classifier = ContextualRiskClassifier(model_name_or_path=tokenizer_path, num_labels=5)
            self.crc_classifier.load_weights(model_path)
        self.tokenizer = self.crc_classifier.get_tokenizer()
        self.model = self.crc_classifier.get_model()
        self.id2label = self.crc_classifier.id2label

    def detect_prompt_risk(self, prompt_text: str, risk_threshold: float = 0.5) -> dict:
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
        risk_scores = {self.id2label[i]: prob for i, prob in enumerate(probabilities)}
        overall_risk = max(risk_scores.values())
        flagged_categories = [cat for cat, score in risk_scores.items() if score > risk_threshold]
        guidance = "Prompt assessed as ethically compliant."
        if overall_risk > risk_threshold:
            guidance = "Potential ethical concern detected; please review and revise your prompt."
        return {
            "prompt": prompt_text,
            "overall_risk_score": overall_risk,
            "flagged_categories": flagged_categories,
            "risk_details": risk_scores,
            "suggested_guidance": guidance,
            "is_flagged": overall_risk > risk_threshold
        }
