import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import get_linear_schedule_with_warmup
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#directory = '/kaggle/input/col774-2022/'
directory = str(sys.argv[1])
dataframe_x = pd.read_csv(os.path.join(directory,'train_x.csv'))
dataframe_y = pd.read_csv(os.path.join(directory, 'train_y.csv'))
dataframe_val_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv'))
dataframe_val_y = pd.read_csv(os.path.join(directory, 'non_comp_test_y.csv'))

batch_size = 50

tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')

input_ids = []
attention_masks = []

sentences = np.hstack((dataframe_x['Title'].values,  dataframe_val_x['Title'].values))
labels = np.hstack((dataframe_y['Genre'].values, dataframe_val_y['Genre'].values))

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(sent,  add_special_tokens = True,  max_length = 64,  pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

learning_rate = 5e-5

def ret_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-large-uncased', 
        num_labels = 30, 
        output_attentions = False, 
        output_hidden_states = False,
    )
    return model

def ret_optim(model):
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr = learning_rate, 
                      eps = 1e-8 
                    )
    return optimizer

def ret_dataloader():
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader,validation_dataloader

num_epochs = 5

def ret_scheduler(dataloader,optimizer):
    epochs = num_epochs
    total_steps = len(dataloader) * epochs
    #allowing very low learning rates for some initial steps - mainly used in attention networks
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    return scheduler

import random
import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#training starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ret_model()
model.to(device)

train_dataloader,validation_dataloader = ret_dataloader()
optimizer = ret_optim(model)
scheduler = ret_scheduler(train_dataloader,optimizer)

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

training_stats = []
epochs = num_epochs

for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
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

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            

    # Measure how long this epoch took.

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("")
    print("Training accuracy...")

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

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

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_train_accuracy = total_eval_accuracy / len(train_dataloader)
    print("  Accuracy: {0:.9f}".format(avg_train_accuracy))

    print("Validation accuracy")
    # Evaluate data for one epoch
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

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

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)


    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.9f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    print("  Validation Loss: {0:.9f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy
        }
    )

print("")
print("Training complete!")

import csv

dataframe_val_x = pd.read_csv(os.path.join(directory,'comp_test_x.csv'))

input_ids = []
attention_masks = []

sentences = dataframe_val_x['Title'].values
labels = dataframe_val_x['Id'].values

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(sent,  add_special_tokens = True,  max_length = 64,  pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

dataset2 = TensorDataset(input_ids, attention_masks, labels)

test_dataloader = DataLoader(
            dataset2, # The validation samples.
            sampler = SequentialSampler(dataset2), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

lis = []
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    with torch.no_grad():        
        outputs = model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask,
                              labels=None)
        logits = outputs['logits']

    # Accumulate the validation loss.
    total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    pred_flat = np.argmax(logits, axis=1).flatten()
    lis += pred_flat.tolist()

predicted_vals = []
iter = 0

for x in lis:
    predicted_vals.append((iter, x))
    iter += 1

header = ['Id','Genre']

directory_out = '/kaggle/working/'

with open(os.path.join(directory,'comp_test_y.csv'), 'w') as f:
#with open(os.path.join(directory_out,'output.csv'), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(header)
    # write a row to the csv file
    writer.writerows(predicted_vals)