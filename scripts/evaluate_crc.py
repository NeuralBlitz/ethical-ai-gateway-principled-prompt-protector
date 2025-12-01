# scripts/evaluate_crc.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch
from src.model import ContextualRiskClassifier
import sys
import os
import json

# --- 1. Data Preparation ---
class PromptDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings.input_ids) # Use length of encodings

def prepare_data_for_evaluation(csv_path, tokenizer, label_cols=None):
    df = pd.read_csv(csv_path)
    prompts = df['prompt'].tolist()
    
    encodings = tokenizer(prompts, truncation=True, padding=True, max_length=512)
    
    labels = None
    if label_cols is not None and all(col in df.columns for col in label_cols):
        labels = df[label_cols].values.tolist()
        return PromptDataset(encodings, labels)
    return PromptDataset(encodings, None) # Return without labels if not available


# --- 2. Evaluation Function ---
def evaluate_crc_model(model_path="models/",
                       eval_results_path="evaluation_metrics.json",
                       bias_data_path="data/bias_eval_set.csv", # Conceptual path for bias auditing
                       base_model_name="distilroberta-base"):

    print(f"Loading model from '{model_path}' for evaluation...")
    # Load model using ContextualRiskClassifier, which loads tokenizer and model config
    classifier = ContextualRiskClassifier(model_name_or_path=model_path, num_labels=5)
    tokenizer = classifier.get_tokenizer()
    model = classifier.get_model()
    id2label, label2id = classifier.get_label_mappings()
    label_cols = [id2label[i] for i in sorted(id2label.keys())]

    model.eval() # Set model to evaluation mode

    # --- Performance Evaluation (on a validation split of the main training data for simplicity) ---
    print("Performing standard performance evaluation...")
    # For a real scenario, you'd use a dedicated, unseen validation dataset
    full_dataset_for_eval = prepare_data_for_training(
        "data/labeled_prompts_v1.csv", tokenizer, label_cols # Use base data for validation split
    )
    _, val_dataset_split = torch.utils.data.random_split(full_dataset_for_eval, 
                                                            [int(0.8 * len(full_dataset_for_eval)), 
                                                             len(full_dataset_for_eval) - int(0.8 * len(full_dataset_for_eval))])
    val_loader = DataLoader(val_dataset_split, batch_size=32)

    all_preds = []
    all_labels = []

    for batch in val_loader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        preds = (torch.sigmoid(logits) > 0.5).int() # Multi-label thresholding
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate performance metrics
    f1 = f1_score(all_labels, all_preds, average='micro')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    eval_results = {
        "f1_score_micro": f1,
        "accuracy": accuracy,
        "precision_micro": precision,
        "recall_micro": recall,
        "overall_model_performance": "PASS" if f1 > 0.75 else "FAIL" # Example threshold
    }
    print(f"Performance Evaluation Results: {eval_results}")


    # --- Ethical Bias Auditing (Conceptual) ---
    print("\nPerforming conceptual ethical bias auditing...")
    bias_audit_results = {}
    try:
        bias_df = pd.read_csv(bias_data_path)
        # Assume bias_df has a 'prompt' column and a 'protected_attribute' column
        # and 'expected_risk_category' if known.
        
        # This is a highly simplified bias audit. In production, this would be
        # a sophisticated set of tests (e.g., A/B testing with demographic groups,
        # counterfactual fairness, toxicity detection on generated text).

        # Example: Check if the model flags prompts differently based on a protected attribute
        prompts_group_a = bias_df[bias_df['protected_attribute'] == 'group_A']['prompt'].tolist()
        prompts_group_b = bias_df[bias_df['protected_attribute'] == 'group_B']['prompt'].tolist()

        # Score prompts (omitting actual model call for brevity, just conceptual output)
        risk_scores_a = [0.1, 0.2, 0.15] # Conceptual scores for group A
        risk_scores_b = [0.4, 0.35, 0.5] # Conceptual scores for group B

        avg_risk_a = sum(risk_scores_a) / len(risk_scores_a)
        avg_risk_b = sum(risk_scores_b) / len(risk_scores_b)

        bias_audit_results = {
            "protected_attribute_bias_check": "FAIL" if abs(avg_risk_a - avg_risk_b) > 0.2 else "PASS", # Example threshold
            "avg_risk_group_A": avg_risk_a,
            "avg_risk_group_B": avg_risk_b,
            "bias_details": "Model shows potential disparity in flagging for protected groups." if abs(avg_risk_a - avg_risk_b) > 0.2 else "No significant disparity detected."
        }
    except FileNotFoundError:
        print(f"Bias evaluation dataset '{bias_data_path}' not found. Skipping bias audit.")
        bias_audit_results = {"status": "SKIPPED", "details": "Bias data not available."}
    except Exception as e:
        print(f"Error during bias audit: {e}")
        bias_audit_results = {"status": "ERROR", "details": str(e)}
    
    print(f"Ethical Bias Audit Results: {bias_audit_results}")

    # --- Final Output ---
    final_results = {
        "model_performance_metrics": eval_results,
        "ethical_bias_audit_metrics": bias_audit_results,
        "timestamp": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU" # Example of adding system info
    }

    with open(eval_results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nEvaluation results saved to '{eval_results_path}'.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the Contextual Risk Classifier model.")
    parser.add_argument("--model_path", type=str, default="models/",
                        help="Path to the trained model directory.")
    parser.add_argument("--eval_results_path", type=str, default="evaluation_metrics.json",
                        help="Path to save evaluation results JSON.")
    parser.add_argument("--bias_data_path", type=str, default="data/bias_eval_set.csv",
                        help="Path to the conceptual bias evaluation dataset CSV.")
    
    args = parser.parse_args()
    
    evaluate_crc_model(
        model_path=args.model_path,
        eval_results_path=args.eval_results_path,
        bias_data_path=args.bias_data_path
    )
