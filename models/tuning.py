import sys
import os

from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import torch.nn as nn
import torch
import torch.optim as optim
import csv

from collections import defaultdict

sys.path.append('../utils')

from models import CNN, LSTM
from data_loader import Dataset
#from pytorch_transformers import AdamW, BertForSequenceClassification

os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing.sequence import pad_sequences

from training import *


def train(model_name="LSTM", params=None):

    # Parameters to tune
    print(params)
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    if model_name == "LSTM":
        learning_rate = params["learning_rate"]
        hidden_dim = params["hidden_dim"]
        num_layers = params["num_layers"]

    embedding_dim = 300
    embedding = "Random"  # "Glove" # "Random" # #Both

    # Constants
    test_percentage = 0.1
    val_percentage = 0.2

    # Load data
    torch.manual_seed(42)
    dataset = Dataset("../data/cleaned_tweets_orig.csv", use_embedding=embedding, embedd_dim=embedding_dim)
    train_data, val_test_data = split_dataset(dataset, test_percentage + val_percentage )
    val_data, test_data = split_dataset(val_test_data, test_percentage/(test_percentage + val_percentage) )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size , collate_fn= my_collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size , collate_fn= my_collate)

    # Define model
    if model_name == "CNN":
        vocab_size = len(dataset.vocab)
        model = CNN(vocab_size, embedding_dim=embedding_dim, combine=params["combine"],
                n_filters=params["filters"])
    elif model_name == "LSTM":
        vocab_size = len(dataset.vocab)
        model = LSTM(vocab_size, embedding_dim, batch_size=batch_size, hidden_dim=hidden_dim, lstm_num_layers=num_layers)
    elif model_name == "Bert":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   collate_fn=bert_collate)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 collate_fn=bert_collate)

    if not model_name=="Bert":
        model.embedding.weight.data.copy_(dataset.vocab.vectors)

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimiser
    optimizer = optim.Adam(model.parameters())
    if model_name=="Bert":
        optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
        # todo: Add scheduler
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    # weighted cross entropy loss, by class counts of other classess
    weights = torch.tensor([0.9414, 0.2242, 0.8344], device = device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(num_epochs):
        # train
        epoch_loss, epoch_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # realtime feel
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {epoch_loss:.5f} | Train Acc: {epoch_acc*100:.2f}%')

    # Compute F1 score on validation set - this is what we optimise during tuning
    loss, acc, predictions, ground_truth = evaluate_epoch(model, val_loader, criterion, device, is_final=True)
    val_f1 = f1_score(y_true=ground_truth, y_pred=predictions, average="macro")
    print("Done")
    return val_f1

def tune_lstm():

    output_fle = open("lstm_tuning.txt", 'w')
    file_writer = csv.writer(output_fle)
    grid = {"learning_rate": [0.005, 0.01, 0.05, 0.1, 0.5], #[2e-4, 2e-5],
            "batch_size": [16, 32],
            "num_epochs": [1, 5],
            "hidden_dim": [64, 128, 256],
            "num_layers": [1,2,3]}
    best_val_f1 = 0.0
    best_params = None
    for params in ParameterGrid(grid):
        val_f1 = train("LSTM", params)
        file_writer.writerow(str(params) + " " + str(val_f1))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params
    print("Best parameters have validation F1: %f" % val_f1)
    print(best_params)

    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    num_epochs = params["num_epochs"]
    hidden_dim = params["hidden_dim"]
    num_layers = params["num_layers"]

def tune_cnn():

    output_fle = open("cnn_tuning.txt", 'w')
    file_writer = csv.writer(output_fle)
    grid = {"learning_rate": [0.005, 0.01, 0.05, 0.1, 0.5], #[2e-4, 2e-5],
            "num_epochs": [1, 5],
            #"embedding_dim": [64, 128, 256],
            "combine": [False],
            "batch_size": [16, 32],
            "filters": [20,50,100]}
    best_val_f1 = 0.0
    best_params = None
    for params in ParameterGrid(grid):
        val_f1 = train("CNN", params)
        file_writer.writerow([str(params), str(val_f1)])
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params
    print("Best parameters have validation F1: %f" % val_f1)
    print(best_params)

    # batch_size = params["batch_size"]
    # learning_rate = params["learning_rate"]
    # num_epochs = params["num_epochs"]
    # hidden_dim = params["hidden_dim"]
    # num_layers = params["num_layers"]


if __name__ == '__main__':
    # Specify model as command line argument
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "LSTM"

    if model == "LSTM":
        print(model)
        tune_lstm()
    elif model == "CNN":
        print(model)
        tune_cnn()
