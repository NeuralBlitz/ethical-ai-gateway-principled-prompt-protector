from transformers import AutoModelForSequenceClassification, AutoTokenizer, PretrainedConfig
import torch

class ContextualRiskClassifier:
    def __init__(self, model_name_or_path="distilroberta-base", num_labels=5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.id2label = {
            0: "harm_prevention_score",
            1: "fairness_discrimination_score",
            2: "privacy_violation_score",
            3: "transparency_deception_score",
            4: "accountability_misuse_score"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        config = PretrainedConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        config.id2label = self.id2label
        config.label2id = self.label2id
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        self.model.eval()
    
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
    
    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_label_mappings(self):
        return self.id2label, self.label2id
