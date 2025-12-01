# scripts/train_crc.py
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
from src.model import ContextualRiskClassifier
import sys
import os

# --- 1. Data Preparation ---
class PromptDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def prepare_data_for_training(csv_path, tokenizer, label_cols):
    df = pd.read_csv(csv_path)
    prompts = df['prompt'].tolist()
    labels = df[label_cols].values.tolist() # Use the dynamic label_cols
    
    encodings = tokenizer(prompts, truncation=True, padding=True, max_length=512)
    return PromptDataset(encodings, labels)

# --- 2. Training Function ---
def train_crc_model(data_path="data/labeled_prompts_v1.csv",
                    model_output_dir="models/",
                    epochs=3,
                    batch_size=16,
                    learning_rate=2e-5,
                    model_name_base="distilroberta-base"):

    # Ensure model_output_dir exists
    os.makedirs(model_output_dir, exist_ok=True)

    # Initialize our custom classifier to get tokenizer and model with correct labels
    classifier = ContextualRiskClassifier(model_name_or_path=model_name_base, num_labels=5) 
    tokenizer = classifier.get_tokenizer()
    model = classifier.get_model()
    id2label, label2id = classifier.get_label_mappings()
    
    # Define label columns based on the classifier's id2label mapping
    label_cols = [id2label[i] for i in sorted(id2label.keys())]


    # Split data for training and validation
    full_dataset = prepare_data_for_training(data_path, tokenizer, label_cols)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=learning_rate,
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # Use loss as a simple metric for this conceptual example
        report_to="none", # Disable reporting to external services like W&B for CI/CD simplicity
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        # compute_metrics function can be added for more detailed evaluation like F1, etc.
        # For simplicity, we'll rely on default eval_loss for this conceptual step.
    )

    print("Starting CRC model training...")
    trainer.train()
    print("CRC model training complete.")
    
    # Save the fine-tuned model weights and tokenizer in HF format
    classifier.save_pretrained(model_output_dir)
    print(f"Model saved to '{model_output_dir}'.")

if __name__ == '__main__':
    # Example command-line arguments parsing
    import argparse
    parser = argparse.ArgumentParser(description="Train the Contextual Risk Classifier model.")
    parser.add_argument("--data_path", type=str, default="data/labeled_prompts_v1.csv",
                        help="Path to the labeled CSV dataset.")
    parser.add_argument("--model_output_dir", type=str, default="models/",
                        help="Directory to save the trained model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    
    args = parser.parse_args()
    
    train_crc_model(
        data_path=args.data_path,
        model_output_dir=args.model_output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
