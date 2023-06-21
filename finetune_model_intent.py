#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install transformers datasets evaluate')


# In[ ]:

from datasets import Dataset, DatasetDict

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import pandas as pd

# In[2]:

import datasets

#imdb = load_dataset("imdb")


#print("=================")
#print(type(imdb))

import json
import pandas as pd

file = 'scicite/train.jsonl'

with open(file) as f:
    train = pd.DataFrame(json.loads(line) for line in f)

file = 'scicite/dev.jsonl'

with open(file) as f:
    test = pd.DataFrame(json.loads(line) for line in f)

train = train.rename(columns={'label': 'labels','string':'text'})
test = test.rename(columns={"label":"labels",'string':'text'})

train = train[['text', 'labels']]
test = test[['text','labels']]


enumClass = {"result":0,"method":1,"background":2}
train["labels"] = [enumClass[i] for i in list(train["labels"])]
test["labels"] = [enumClass[i] for i in list(test["labels"])]

train_ds = Dataset.from_pandas(train)
test_ds = Dataset.from_pandas(test)
ds = DatasetDict({"train":train_ds,"test":test_ds})
print(ds)

# In[3]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


# In[4]:


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


# In[5]:


tokens = ds.map(preprocess_function, batched=True)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("================ after data collector ===================")

# In[7]:

#print(tokenized_train)

import evaluate

accuracy = evaluate.load("accuracy")


# In[23]:


import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# In[9]:

id2label = {0:"result",1:"method",2:"background"}
label2id = {"result":0,"method":1,"background":2}


# In[13]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased", num_labels=3, id2label=id2label, label2id=label2id
)


# In[ ]:


training_args = TrainingArguments(
    output_dir="sci_intent_classify",
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

#trainer.push_to_hub()
