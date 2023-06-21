import pandas as pd 
    
# making dataframe 
df = pd.read_csv("citation_sentiment")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased") 

enumClass = {"n":0,"p":1,"o":2}
df["labels"] = [enumClass[i] for i in list(df["labels"])]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["labels"], test_size=0.15, random_state=42)


train = pd.DataFrame()
train["text"] = X_train
train["labels"] = y_train

test = pd.DataFrame()


test["text"] = X_test
test["labels"] = y_test

from datasets import Dataset, DatasetDict

train_ds = Dataset.from_pandas(train)
test_ds = Dataset.from_pandas(test)

ds = DatasetDict({"train":train_ds,"test":test_ds})


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


tokens = ds.map(preprocess_function, batched=True)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


import evaluate

accuracy = evaluate.load("accuracy")


import numpy as np


def compute_metrics(eval_pred):
    print("hellooo")
    predictions, labels = eval_pred
    print(predictions)
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "n", 1: "p", 2:"o"}
label2id = {"n": 0, "p": 1,"o":2}


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased", num_labels=3, id2label=id2label, label2id=label2id
)


training_args = TrainingArguments(
    output_dir="sci_sentiment_neg",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokens["train"],
    eval_dataset=tokens["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
