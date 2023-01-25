import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
import sys
import csv

# use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define directory
#directory = '/kaggle/input/col774-2022/'
directory = str(sys.argv[1])

#directory_out = '/kaggle/working/'
directory_out = '.'

# read data
dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv'))
dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv'))

dataframe_val_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv'))
dataframe_val_y = pd.read_csv(os.path.join(directory, 'non_comp_test_y.csv'))

dataframe_test_x = pd.read_csv(os.path.join(directory,'comp_test_x.csv'))

# concatenate train and non_comp test data
sentences = np.hstack((dataframe_x['Title'].values,  dataframe_val_x['Title'].values))
labels = np.hstack((dataframe_y['Genre'].values, dataframe_val_y['Genre'].values))
labels = torch.tensor(labels)

# pre-trained model to be used
pretrained_model_name = 'bert-large-uncased'

# parameters
batch_size = 50
learning_rate = 5e-5
num_epochs = 5
max_length = 64

# tokenize and encode sequences in the training set
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(sent,  add_special_tokens = True,  max_length = max_length, truncation=True,  padding='max_length', return_attention_mask = True, return_tensors = 'pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# split the dataset into training and validation set
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# return model
def getModel():
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, 
        num_labels = 30, 
        output_attentions = False, 
        output_hidden_states = False,
    )
    return model

# return optimizer
def getOptim(model):
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr = learning_rate, 
                      eps = 1e-8 
                    )
    return optimizer

# return dataloader
def getDataloader():
    train_dataloader = DataLoader(
                train_dataset, 
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
            )
    return train_dataloader,validation_dataloader

# return scheduler
def getScheduler(dataloader,optimizer):
    epochs = num_epochs
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    return scheduler

# compute accuracy of predictions
def computeAccuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training 
print('Training model ...')
model = getModel()
model.to(device)

train_dataloader, validation_dataloader = getDataloader()
optimizer = getOptim(model)
scheduler = getScheduler(train_dataloader,optimizer)

#seeding to minimize randomness
seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)

training_stats = []
for epoch_i in range(0, num_epochs):

    print(f'Epoch {epoch_i + 1}')
    total_train_loss = 0

    model.train()

    # iterating over batches of train data
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
        loss, logits = outputs['loss'], outputs['logits']
        total_train_loss += loss.item()
        avg_train_loss = loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    # Average loss over all batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            

    print("Average training loss: {0:.2f}".format(avg_train_loss))

    model.eval()

    total_train_accuracy = 0
    total_train_loss = 0

    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                                  token_type_ids=None, 
                                  attention_mask=b_input_mask,
                                  labels=b_labels)
            loss, logits = outputs['loss'], outputs['logits']

        total_train_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_train_accuracy += computeAccuracy(logits, label_ids)

    # Average training accuracy over all batches.
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    print("Training Accuracy: {0:.9f}".format(avg_train_accuracy))

    total_val_accuracy = 0
    total_val_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                                  token_type_ids=None, 
                                  attention_mask=b_input_mask,
                                  labels=b_labels)
            loss, logits = outputs['loss'], outputs['logits']

        total_val_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_val_accuracy += computeAccuracy(logits, label_ids)

    # Average validation accuracy over all batches
    avg_val_accuracy = total_val_accuracy / len(validation_dataloader)
    print("Validation Accuracy: {0:.9f}".format(avg_val_accuracy))

    avg_val_loss = total_val_loss / len(validation_dataloader)

    print("Average validation Loss: {0:.9f}".format(avg_val_loss))

    # Save all stats on this epoch
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy
        }
    )

print("Model Trained!")

# Testing
print("Testing model ...")
input_ids = []
attention_masks = []

sentences = dataframe_test_x['Title'].values
labels = torch.tensor(dataframe_test_x['Id'].values)

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(sent,  add_special_tokens = True,  max_length = max_length, truncation=True,  padding='max_length', return_attention_mask = True, return_tensors = 'pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

dataset_test = TensorDataset(input_ids, attention_masks, labels)

test_dataloader = DataLoader(
            dataset_test,
            sampler = SequentialSampler(dataset_test),
            batch_size = batch_size
        )

preds = []
for batch in test_dataloader:
    batch_input_ids = batch[0].to(device)
    batch_input_mask = batch[1].to(device)
    with torch.no_grad():        
        outputs = model(batch_input_ids, 
                              token_type_ids=None, 
                              attention_mask=batch_input_mask,
                              labels=None)
        logits = outputs['logits']

    logits = logits.detach().cpu().numpy()

    # compute predictions
    pred_flat = np.argmax(logits, axis=1).flatten()
    preds += pred_flat.tolist()

predicted_vals = []
iter = 0

for pred in preds:
    predicted_vals.append((iter, pred))
    iter += 1

# write to csv file
header = ['Id','Genre']

with open(os.path.join(directory_out,'comp_test_y.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(predicted_vals)

print('Generated predictions!')