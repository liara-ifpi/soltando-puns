import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from sklearn import metrics
from sklearn.metrics import classification_report


dataset = load_dataset('Superar/Puntuguese')
dataset = dataset.remove_columns('labels')

# checkpoint = 'google-bert/bert-base-multilingual-cased'
checkpoint = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",  # Save checkpoints at the end of each epoch
    logging_strategy="epoch",
    save_total_limit=3,  # Keep only the last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    logging_dir='logs',
    report_to=[]
)

# Define Trainer object for training the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

torch.cuda.empty_cache()
# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('model')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities
    logits = torch.tensor(logits)
    probs = logits.softmax(dim=-1)
    # Get predictions from probabilities
    predictions = probs.argmax(axis=-1)
    print(classification_report(labels, predictions))

    f1 = metrics.f1_score(labels, predictions, zero_division = 0, average='macro')
    recall = metrics.recall_score(labels, predictions, zero_division = 0, average='macro')
    precision = metrics.precision_score(labels, predictions, zero_division = 0, average='macro')
    acc = metrics.accuracy_score(labels, predictions)

    probs = probs[:, 1].numpy()  # Get probabilities for the positive class
    auc = metrics.roc_auc_score(labels, probs)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

# Define Trainer with test dataset
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

eval_results = trainer.evaluate()

# Print the evaluation results
print("Evaluation results:")
for key, value in eval_results.items():
    print(f"{key}: {value}")