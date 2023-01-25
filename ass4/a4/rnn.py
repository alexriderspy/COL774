import csv
import sys
import os
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext.data import get_tokenizer

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

dataframe_test_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv'))
dataframe_test_y = pd.read_csv(os.path.join(directory, 'non_comp_test_y.csv'))

# parameters
num_epochs = 50
learning_rate = 0.01
batch_size = 20 

# architecture 
sentence_length = 15
embedding_dim = 300
input_dim = 300 
hidden_dim = 128 
layers = 1 
outputs = 30

# tokenize and encode sequences in the training set
glove = torchtext.vocab.GloVe(name='6B', dim = embedding_dim)
tokenizer = get_tokenizer("basic_english")

class TextLoader(Dataset):
    def __init__(self, dataframe_x, dataframe_y, sentence_length, embedding_dim):
        self.dataframe_x = dataframe_x
        self.dataframe_y = dataframe_y
        self.sentence_length = sentence_length

        def to_embedding(sentence):
            tokens = tokenizer(sentence)
            tokens = (tokens +[""] * (self.sentence_length-len(tokens))) if len(tokens)<self.sentence_length else tokens[:self.sentence_length] 
            return glove.get_vecs_by_tokens(tokens) 

        titles = dataframe_x['Title'].to_numpy()
        embeddings = torch.zeros(len(titles), self.sentence_length, embedding_dim).to(device)
        
        for i in range(len(titles)):
            embeddings[i] = to_embedding(titles[i]).to(device)

        self.embedded_x = embeddings
        

    def __len__(self):
        return len(self.dataframe_x)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        embeddings = self.embedded_x[idx].to(device)
        labelKey = self.dataframe_y.iloc[idx, 1]
        label = (torch.tensor(int(labelKey)).to(device))

        return embeddings, label

def get_output_shape(model, input_dim):
    rand_input = torch.rand(1, input_dim).to(device)
    return model(rand_input)[0].shape

class RecurrentNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, layers, outputs):
        super(RecurrentNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim , hidden_dim, layers, bidirectional=True, batch_first=True).to(device)

        rnn_out = get_output_shape(self.rnn, input_dim)
        flattened_size = np.prod(list(rnn_out))

        self.fc1 = nn.Linear(flattened_size,hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, outputs).to(device)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        x = torch.tanh(self.fc1(out[:, -1, :])).to(device)
        x = (self.fc2(x)).to(device)
        return x
    
model = RecurrentNet(input_dim, hidden_dim, layers, outputs).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

dataset = TextLoader(dataframe_x = dataframe_x, dataframe_y = dataframe_y, sentence_length = sentence_length, embedding_dim = embedding_dim)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = 0)

for epoch in (range(num_epochs)):
    for i, (titles, labels) in enumerate(dataloader):
        titles = titles.to(device)
        labels = labels.to(device)
        outputs = model(titles).to(device)
        loss = criterion(outputs, labels).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

predicted_vals = []
        
dataset_test = TextLoader(dataframe_x = dataframe_test_x, dataframe_y = dataframe_test_y, sentence_length=sentence_length, embedding_dim=embedding_dim)
dataloader_test = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle=False, num_workers = 0)

with torch.no_grad():
    n_correct = 0
    n_samples = 0

    iter = 0
    for text, labels in dataloader_test:
        text = text.to(device, dtype=torch.float)
        labels = labels
        outputs = model(text)
        _, predicted = torch.max(outputs,1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(predicted)):
            
            pred = predicted[i]
            predicted_vals.append([iter,pred.item()])
            iter += 1

with torch.no_grad():
    correct = 0
    samples = 0

    iter = 0
    for text, labels in dataloader:
        text = text.to(device, dtype=torch.float)
        labels = labels.to(device)
        outputs = model(text)
        _, predicted = torch.max(outputs,1)

        samples += labels.size(0)
        correct += (predicted == labels).sum().item()

header = ['Id','Genre']
with open(os.path.join(directory_out,'non_comp_test_pred_y.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(predicted_vals)

print("Accuracy of train = ", str(correct/samples))
print("Accuracy of non comp test = ", str(n_correct/n_samples))
