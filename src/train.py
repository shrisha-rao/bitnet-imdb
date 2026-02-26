import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import torch
from utils import replace_linear_with_bitlinear

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)  # Higher LR for quantized training
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Use a subset of data for quick experimentation")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("imdb")
    if args.max_samples:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(args.max_samples))
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(args.max_samples // 5))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Replace linear layers with BitLinear
    replace_linear_with_bitlinear(model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./bitnet-imdb-finetuned")
    tokenizer.save_pretrained("./bitnet-imdb-finetuned")

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

if __name__ == "__main__":
    main()
