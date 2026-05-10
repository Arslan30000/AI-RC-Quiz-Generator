import json

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in text.strip().split('\n')]
    })

add_markdown("""
# AI Reading Comprehension: Deep Learning Text Generation
## Fine-Tuning T5 for Question Generation
This notebook is designed to be run on **Google Colab** using the T4 GPU. 
It fulfills the project requirement of training a Neural Network sequence-to-sequence model and evaluating it using BLEU, ROUGE, and METEOR.

**Instructions:**
1. Go to `Runtime > Change runtime type` and select **T4 GPU**.
2. Upload your `train.csv` and `val.csv` files to the Colab files panel on the left.
3. Click `Runtime > Run all`.
""")

add_code("""
!pip install transformers datasets evaluate rouge_score nltk accelerate meteor
""")

add_code("""
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import nltk

nltk.download('wordnet')
nltk.download('punkt')
""")

add_markdown("""
## 1. Load the RACE Dataset
*Note: We load a subset of the data (5,000 rows) so that the notebook finishes in a reasonable time on Colab. You can increase this if you want to train longer!*
""")

add_code("""
# Load dataset
train_df = pd.read_csv('train.csv').head(5000) 
val_df = pd.read_csv('val.csv').head(500)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print(f"Training on {len(train_dataset)} examples.")
""")

add_markdown("""
## 2. Preprocess Data for T5
T5 requires a specific prompt format. We prepend `generate question:` to the article.
""")

add_code("""
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # Prefix the input with a task description for T5
    inputs = ["generate question: " + str(doc) for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    # Setup targets
    labels = tokenizer(text_target=[str(q) for q in examples["question"]], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)
""")

add_markdown("""
## 3. Define Evaluation Metrics (BLEU, ROUGE, METEOR)
Your teacher requested these exact metrics for text generation.
""")

add_code("""
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute metrics
    result_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
    # BLEU and METEOR require specific formatting
    # BLEU expects references to be a list of lists
    bleu_refs = [[ref] for ref in decoded_labels]
    result_bleu = bleu.compute(predictions=decoded_preds, references=bleu_refs)
    
    result_meteor = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "rougeL": result_rouge["rougeL"],
        "bleu": result_bleu["bleu"],
        "meteor": result_meteor["meteor"],
    }
""")

add_markdown("""
## 4. Train the Neural Network (with Checkpointing)
This block automatically saves checkpoints to your Google Drive folder so you don't lose progress if Colab crashes.
""")

add_code("""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments WITH CHECKPOINTING
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_rc_checkpoints",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    save_strategy="epoch",
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Fast training on GPU
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting Deep Learning Training...")
trainer.train()
""")

add_markdown("""
## 5. View Final Metrics!
Once training is complete, the final metrics will be printed above. You can screenshot those metrics (BLEU, ROUGE, METEOR) and put them directly in your Final Report!
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/Neural_Question_Generation_T5.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Successfully generated notebooks/Neural_Question_Generation_T5.ipynb")
