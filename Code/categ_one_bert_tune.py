from collections import defaultdict
import numpy as np
import tabulate
from sklearn.metrics import f1_score
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import logging
import csv
csv.field_size_limit(100000000)
from sklearn.model_selection import train_test_split
import random
import json

# STEP ONE: BERT MODEL
from transformers import BertTokenizer, BertForSequenceClassification
import torch

categories = ["cond-mat.soft", "cond-mat.supr-con", "cond-mat.mtrl-sci", "cond-mat.str-el", "cond-mat.mes-hall", "cond-mat.dis-nn", "cond-mat.other", "cond-mat.stat-mech", "cond-mat.quant-gas"]

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))

label_mapping = {label: index for index, label in enumerate(categories)}

cond_filtered8 = []
# Read the entire JSON file as a single string
with open('cond_filtered8.json', 'r') as f:
    json_data = f.read()
# Parse the JSON string
data = json.loads(json_data)
# Append each entry (nested dictionary) to the list
for entry in data.values():
    cond_filtered8.append(entry)

class Datum:
    def __init__(self, id, title, categories, abstract, journal, category_one):
        self.id = id
        self.title = title
        self.categories = categories
        self.abstract = abstract
        self.journal = journal
        self.category_one = category_one

    def getid(self):
        return self.id
    
    def gettitle(self):
        return self.title
    
    def getcategories(self):
        return self.categories
    
    def getabstract(self):
        return self.abstract
    
    def getjournal(self):
        return self.journal
    
    def getcategory_one(self):
        return self.category_one
    
    def setid(self, id):
        self,id = id

    def settitle(self, title):
        self.title = title

    def setcategories(self):
         self.categories = categories
    
    def setabstract(self):
         self.abstract = abstract
    
    def setjournal(self):
        self.journal = journal

    def setcategory_one(self):
        self.category_one = category_one

arxiv_data_objects = []
for entry in cond_filtered8:
    if isinstance(entry, dict):
        datum = Datum(entry['id'], entry['title'], entry['categories'], entry['abstract'], entry.get('journal-ref', ''), entry.get('category_one', '')) 
        arxiv_data_objects.append(datum)
    else:
        print("Skipping entry because it's not a dictionary:", entry)

import random


# Shuffle the data with a fixed seed for reproducibility
random.seed(0)
random.shuffle(arxiv_data_objects)

# Calculate the sizes of train, dev, and test sets
train_size = int(0.8 * len(arxiv_data_objects))
dev_size = int(0.1 * len(arxiv_data_objects)) 
test_size = len(arxiv_data_objects) - train_size - dev_size

# Split the data into train, dev, and test sets
train_data = arxiv_data_objects[:train_size]
dev_data = arxiv_data_objects[train_size:train_size + dev_size]
test_data = arxiv_data_objects[train_size + dev_size:]

# Verify the sizes of each set
print("Train set size:", len(train_data))
print("Dev set size:", len(dev_data))
print("Test set size:", len(test_data))

# Pre-processing abstracts
import re
from urllib.parse import urlparse

def clean_text(text):
    # Remove punctuation using regex
    text = re.sub(r"@[A-Za-z_-]+", 'USR', text)
    text = re.sub(r"https?\S+", 'URL', text)

    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

    text = text.strip()

    text = text.lower()

    return text

# Runnign the Bert toeknizer on our abstracts
import string

def analyse_tokens(train_data):
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Initialize the counter
    counter = defaultdict(int)

    # Loop over the data
    for entry in tqdm(train_data[:10000]):
        # Remove punctuation and URLs from the abstract
        abstract = entry.getabstract()
        abstract = clean_text(abstract)
        
        # Tokenize the abstract and convert token ids to tokens
        encoded_abstract = tokenizer(
            abstract, 
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        decoded_abstract = tokenizer.convert_ids_to_tokens(encoded_abstract['input_ids'][0])

        # Loop over the tokens
        current = []
        for token in decoded_abstract:
            if token.startswith('##'):  
                current.append(token[2:])
            else:
                if current:
                    # If current is not empty, join the tokens and add to the counter
                    counter[tuple(current)] += 1
                current = [token]  
        
        # Add the last word to the counter
        if current:
            counter[tuple(current)] += 1

    # Sort the counter by size and frequency
    counter = sorted(counter.items(), key=lambda x: (len(x[0]), x[1]), reverse=True)

    # Print the top 10 largest words
    print('Top 10 largest words:')
    for i in counter[:10]:
        print(i)

    # Sort the counter by frequency and print the 100 smallest tokens
    counter = sorted(counter, key=lambda x: x[1])
    print('100 smallest tokens:')
    for i in counter[:100]:
        print(i)

# Call the function with your train_data
analyse_tokens(train_data)

class Config:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

label_mapping = {label: index for index, label in enumerate(categories)}

# Trial to optimize; 
import torch.nn.functional as F

logging.set_verbosity_error()

class BERTClassifier(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(BERTClassifier, self).__init__()
        self.llm = BertModel.from_pretrained('bert-base-uncased').to(Config.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_prob).to(Config.device)
        self.linear_1 = nn.Linear(768, int(768 / 3)).to(Config.device)
        self.relu = nn.ReLU().to(Config.device)
        self.linear_2 = nn.Linear(int(768/3), 9).to(Config.device)

        # Fine-tune more layers of BERT
        frozen_layer = 9
        modules = [self.llm.embeddings, *self.llm.encoder.layer[:frozen_layer]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = output.pooler_output

        linear_1_output = self.linear_1(pooler_output)
        relu_output = self.relu(linear_1_output)
        dropout_output = self.dropout(relu_output)
        output = self.linear_2(dropout_output)
        return output
    
def custom_collate(batch):
    return batch

from torch.utils.data import DataLoader
# Create a function called train_epoch

def train_epoch(train_data, classifier, optimizer, criterion, batch_size):
    total_loss = 0
    classifier.train()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    # Loop over the train_data
    for batch in tqdm(train_loader):
        input_ids = []
        attention_masks = []
        targets = []

        for datum in batch:
            abstract = datum.getabstract()
            abstract = clean_text(abstract)
            encoded_abstract = tokenizer(
                abstract,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            )

            input_ids.append(encoded_abstract['input_ids'].to(Config.device).squeeze(0))
            attention_masks.append(encoded_abstract['attention_mask'].to(Config.device).squeeze(0))
            target_category = datum.getcategory_one()
            target_category_id = label_mapping[target_category]
            targets.append(target_category_id)

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        targets = torch.tensor(targets, dtype=torch.long, device=Config.device)

        optimizer.zero_grad()
        outputs = classifier(input_ids=input_ids, attention_mask=attention_masks)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (len(train_data) / batch_size)

# Train classifier funciton that trains multiple epochs
def train_classifier(train_data, classifier, optimizer, criterion, n_epochs, batch_size):
    total_loss = 0
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_data, classifier, optimizer, criterion, batch_size)
        print('Average loss for epoch %d: %.4f' % (epoch, loss))
        total_loss += loss
    return total_loss / n_epochs

import torch
import random
from tqdm import tqdm
from sklearn.metrics import classification_report


# Test classifier applies the trained classifier to the test data
def test_classifier(test_data, classifier, criterion, tokenizer, batch_size):
    total_loss = 0
    all_outputs = []
    all_targets = []

    classifier.eval()  
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    with torch.no_grad():
        # Loop over the test_data
        for batch in tqdm(test_loader):
            input_ids = []
            attention_masks = []
            targets = []
            for datum in batch:
                abstract = datum.getabstract()
                abstract = clean_text(abstract)
                encoded_abstract = tokenizer(
                    abstract,
                    padding='max_length',
                    max_length=512,  
                    truncation=True,
                    return_tensors='pt',
                    return_token_type_ids=False
                )
                input_ids.append(encoded_abstract['input_ids'].to(Config.device).squeeze(0))
                attention_masks.append(encoded_abstract['attention_mask'].to(Config.device).squeeze(0))
                target_category = datum.getcategory_one()
                target_category_id = label_mapping[target_category]
                targets.append(target_category_id)

            input_ids = torch.stack(input_ids).to(Config.device)
            attention_masks = torch.stack(attention_masks).to(Config.device)
            targets = torch.tensor(targets, dtype=torch.long, device=Config.device)

            outputs = classifier(input_ids=input_ids, attention_mask=attention_masks)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predicted_categories = torch.argmax(outputs, dim=1)
            all_outputs.extend(predicted_categories.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    print(f'Average loss on test data: {avg_loss:.4f}')
    print(classification_report(all_targets, all_outputs))
    return avg_loss  # Return average loss    

# Subset of train_data to make the process faster: 
import random

subset_size = 2000
subset_train_data = random.sample(train_data, subset_size)

#Hyper-tuning the parameters:
import itertools
learning_rate_list = [1e-5, 5e-5, 3e-5]
epochs_list = [1, 2, 3]
batch_size_list = [16, 32]

best_hyperparams = {}
best_loss = float('inf')

# Iterate through all combinations of hyperparameters
for learning_rate, epochs, batch_size in itertools.product(learning_rate_list, epochs_list, batch_size_list):
    print(f"Iteration: Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}")

    # Initialize classifier, criterion, and optimizer
    classifier = BERTClassifier().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Train the classifier
    avg_loss = train_classifier(subset_train_data, classifier, optimizer, criterion, epochs, batch_size)

    # Evaluate the classifier
    dev_loss = test_classifier(dev_data, classifier, criterion, tokenizer, batch_size)

    # Update best hyperparameters if test loss improves
    if dev_loss < best_loss:
        best_loss = dev_loss
        best_hyperparams = {'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': batch_size}

print("Best hyperparameters:", best_hyperparams)

# Testing on the test data:
best_learning_rate = best_hyperparams['learning_rate']
best_epochs = best_hyperparams['epochs']
best_batch_size = best_hyperparams['batch_size']

# Initialize classifier, criterion, and optimizer with best hyperparameters
classifier = BERTClassifier().to(Config.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=best_learning_rate)

# Train the classifier
train_classifier(train_data, classifier, optimizer, criterion, best_epochs, best_batch_size)

# Evaluate the classifier
test_classifier(test_data, classifier, criterion, tokenizer, best_batch_size)

# STEP TWO: SCIBERT MODEL
class Config:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Pre-processing abstracts
import re
from urllib.parse import urlparse

def clean_text(text):
    # Remove punctuation using regex
    text = re.sub(r"@[A-Za-z_-]+", 'USR', text)
    text = re.sub(r"https?\S+", 'URL', text)

    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

    text = text.strip()

    text = text.lower()

    return text

from collections import defaultdict
from tqdm import tqdm
from transformers import BertTokenizer

def analyse_tokens(train_data):
    # Initialize the SciBERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # Initialize the counter
    counter = defaultdict(int)

    # Loop over the data
    for entry in tqdm(train_data[:10000]):
        # Remove punctuation and URLs from the abstract
        abstract = entry.getabstract()
        abstract = clean_text(abstract)
        
        # Tokenize the abstract and convert token ids to tokens
        encoded_abstract = tokenizer(
            abstract, 
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        decoded_abstract = tokenizer.convert_ids_to_tokens(encoded_abstract['input_ids'][0])

        # Loop over the tokens
        current = []
        for token in decoded_abstract:
            if token.startswith('##'):
                current.append(token[2:])
            else:
                if current:
                    # If current is not empty, join the tokens and add to the counter
                    counter[tuple(current)] += 1
                current = [token]
        
        # Add the last word to the counter
        if current:
            counter[tuple(current)] += 1

    # Sort the counter by size and frequency
    counter = sorted(counter.items(), key=lambda x: (len(x[0]), x[1]), reverse=True)

    # Print the top 10 largest words
    print('Top 10 largest words:')
    for i in counter[:10]:
        print(i)

    # Sort the counter by frequency and print the 100 smallest tokens
    counter = sorted(counter, key=lambda x: x[1])
    print('100 smallest tokens:')
    for i in counter[:100]:
        print(i)

# Call the function with your train_data
analyse_tokens(train_data)

import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Set the tokenizer for defining the scibert class
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

class sciBERTClassifier(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(sciBERTClassifier, self).__init__()
        # Initialize the BERT model
        self.llm = BertModel.from_pretrained('allenai/scibert_scivocab_uncased').to(Config.device)
        self.tokenizer = tokenizer 
        self.dropout = nn.Dropout(dropout_prob).to(Config.device)

        hidden_size = self.llm.config.hidden_size
        self.linear_1 = nn.Linear(hidden_size, int(hidden_size / 3)).to(Config.device)
        self.relu = nn.ReLU().to(Config.device)
        self.linear_2 = nn.Linear(int(hidden_size / 3), 9).to(Config.device)

        frozen_layer = 9
        modules = [self.llm.embeddings, *self.llm.encoder.layer[:frozen_layer]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = output.pooler_output

        linear_1_output = self.linear_1(pooler_output)
        relu_output = self.relu(linear_1_output)
        dropout_output = self.dropout(relu_output)
        output = self.linear_2(relu_output)
        return output
    
def custom_collate(batch):
    return batch

from torch.utils.data import DataLoader
# Create a function called train_epoch
def train_epoch(train_data, classifier, optimizer, criterion, batch_size):
    total_loss = 0
    classifier.train()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    # Loop over the train_data
    for batch in tqdm(train_loader):
        input_ids = []
        attention_masks = []
        targets = []

        for datum in batch:
            abstract = datum.getabstract()
            abstract = clean_text(abstract)
            encoded_abstract = tokenizer(
                abstract,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False
            )

            input_ids.append(encoded_abstract['input_ids'].to(Config.device).squeeze(0))
            attention_masks.append(encoded_abstract['attention_mask'].to(Config.device).squeeze(0))
            target_category = datum.getcategory_one()
            target_category_id = label_mapping[target_category]
            targets.append(target_category_id)

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        targets = torch.tensor(targets, dtype=torch.long, device=Config.device)

        optimizer.zero_grad()
        outputs = classifier(input_ids=input_ids, attention_mask=attention_masks)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (len(train_data) / batch_size)

# Train classifier performs train_epoch multiple time:  
def train_classifier(train_data, classifier, optimizer, criterion, n_epochs, batch_size):
    total_loss = 0
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_data, classifier, optimizer, criterion, batch_size)
        print('Average loss for epoch %d: %.4f' % (epoch, loss))
        total_loss += loss
    return total_loss / n_epochs

import torch
import random
from tqdm import tqdm
from sklearn.metrics import classification_report

# Test_classifier classifier tests the train classifier on test data
def test_classifier(test_data, classifier, criterion, tokenizer, batch_size):
    total_loss = 0
    all_outputs = []
    all_targets = []

    classifier.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    with torch.no_grad():
        # Loop over the test_data
        for batch in tqdm(test_loader):
            input_ids = []
            attention_masks = []
            targets = []
            for datum in batch:
                abstract = datum.getabstract()
                abstract = clean_text(abstract)
                encoded_abstract = tokenizer(
                    abstract,
                    padding='max_length',
                    max_length=512,
                    truncation=True,
                    return_tensors='pt',
                    return_token_type_ids=False
                )
                input_ids.append(encoded_abstract['input_ids'].to(Config.device).squeeze(0))
                attention_masks.append(encoded_abstract['attention_mask'].to(Config.device).squeeze(0))
                target_category = datum.getcategory_one()
                target_category_id = label_mapping[target_category]
                targets.append(target_category_id)

            input_ids = torch.stack(input_ids).to(Config.device)
            attention_masks = torch.stack(attention_masks).to(Config.device)
            targets = torch.tensor(targets, dtype=torch.long, device=Config.device)

            outputs = classifier(input_ids=input_ids, attention_mask=attention_masks)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predicted_categories = torch.argmax(outputs, dim=1)
            all_outputs.extend(predicted_categories.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    print(f'Average loss on test data: {avg_loss:.4f}')
    print(classification_report(all_targets, all_outputs))
    return avg_loss  

# Running the train epochs, train classifier, and test classifier
import random

subset_size = 2000
subset_train_data = random.sample(train_data, subset_size)

#Hyper-tuning the parameters:
import itertools
learning_rate_list = [1e-5, 5e-5, 3e-5]
epochs_list = [1, 2, 3]
batch_size_list = [16, 32]

best_hyperparams = {}
best_loss = float('inf')

# Iterate through all combinations of hyperparameters
for learning_rate, epochs, batch_size in itertools.product(learning_rate_list, epochs_list, batch_size_list):
    print(f"Iteration: Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}")

    # Initialize classifier, criterion, and optimizer
    classifier = sciBERTClassifier().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Train the classifier
    avg_loss = train_classifier(subset_train_data, classifier, optimizer, criterion, epochs, batch_size)

    # Evaluate the classifier
    dev_loss = test_classifier(dev_data, classifier, criterion, tokenizer, batch_size)

    # Update best hyperparameters if test loss improves
    if dev_loss < best_loss:
        best_loss = dev_loss
        best_hyperparams = {'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': batch_size}

print("Best hyperparameters:", best_hyperparams)


# Testing on the test data:
best_learning_rate = best_hyperparams['learning_rate']
best_epochs = best_hyperparams['epochs']
best_batch_size = best_hyperparams['batch_size']

# Initialize classifier, criterion, and optimizer with best hyperparameters
classifier = sciBERTClassifier().to(Config.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=best_learning_rate)

# Train the classifier
train_classifier(train_data, classifier, optimizer, criterion, best_epochs, best_batch_size)

# Evaluate the classifier
test_classifier(test_data, classifier, criterion, tokenizer, best_batch_size)
