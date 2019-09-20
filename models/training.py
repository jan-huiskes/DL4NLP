import sys
import os

from sklearn.metrics import confusion_matrix, classification_report
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

sys.path.append('../utils')

from models import CNN, LSTM
from data_loader import Dataset
from pytorch_transformers import AdamW, BertForSequenceClassification

os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing.sequence import pad_sequences




def split_dataset(dataset, test_percentage=0.1):
    """
    Split a dataset in a train and test set.

    Parameters
    ----------
    dataset : dataset.Data
        Custom dataset object.
    test_percentage : float, optional
        Percentage of the data to be assigned to the test set.
    """
    test_size = round(len(dataset) * test_percentage)
    train_size = len(dataset) - test_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

def get_loss_weights(dataset):
    count = torch.zeros((3,1))
    for __, y in dataset:
        count[int(y[0].item())]+=1
    total_count = torch.sum(count)
    weights =  torch.tensor([(count[1]+count[2])/total_count, (count[0]+count[2])/total_count, (count[1]+count[0])/total_count])
    print(weights)
    return weights

def accuracy(predictions, targets):
    predictions = torch.argmax(predictions, dim=1)
    return (predictions== targets).sum().to(torch.float)/predictions.shape[0]

def train_epoch(model, loader, optimizer, criterion, device):
    epoch_loss, epoch_acc = 0, 0

    model.train()
    for batch in loader:
        optimizer.zero_grad()
        x, y = batch[0].to(device), batch[1].to(torch.long).to(device)
        if type(model) is BertForSequenceClassification:
            loss, logits = model(x, labels=y)
            acc = accuracy(logits, y)
        else:
            predictions = model(x).squeeze(1)
            loss = criterion(predictions, y[:, 0])
            acc = accuracy(predictions, y[:, 0])

        #print(loss.item())
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / (len(loader)), epoch_acc / (len(loader))


def evaluate_epoch(model, loader, criterion, device, is_final = False):
    eval_loss, eval_acc = 0, 0
    if is_final:
        prediction_list = []
        ground_truth = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(torch.long).to(device)
            if type(model) is BertForSequenceClassification:
                loss, logits = model(x, labels=y)
                acc = accuracy(logits, y)
            else:
                predictions = model(x).squeeze(1)
                loss = criterion(predictions, y[:, 0])
                acc = accuracy(predictions, y[:, 0])
            eval_loss += loss.item()
            eval_acc += acc.item()
            if is_final:
                prediction_list+= list(torch.argmax(predictions, dim=1).detach().cpu().numpy())
                ground_truth+= list(y[:, 0].detach().cpu().numpy())
    if is_final:
        return  eval_loss / (len(loader)), eval_acc / (len(loader)), prediction_list, ground_truth
    return eval_loss / (len(loader)), eval_acc / (len(loader))

def my_collate(batch):
    """
    Collate function for dataloader.

    Parameters
    ----------
    batch : (x,y) tuple, where x is a tensor with indices and y is a tensor with labels
    ----------
    Returns: (x,y) , where x has shape [sentence_length, batch_size] and y has shape [batch_size, number_of_labels]
    """

    #y to be returned.
    new_batch = {"y": torch.Tensor().new_empty((0, 5))}
    #get labels and max length
    max_length = 0
    for x, y in batch:

        max_length = max(max_length, len(x))
        new_batch['y'] = torch.cat((new_batch['y'], y.view(-1,5)), 0)

    #get x tensors with the same length with shape [len, batch_size]
    new_batch["x"]= torch.Tensor().new_empty((max_length, 0))
    for x,y in batch:

        x = x.to(torch.float).permute(1,0)
        x = nn.functional.pad(x, (0, max_length - x.shape[1]), mode='constant', value=0).permute(1,0)
        new_batch['x'] = torch.cat((new_batch['x'], x), 1)

    new_batch['x'] = new_batch['x'].to(dtype=torch.long)

    return new_batch['x'] , new_batch['y']

def bert_collate(batch):
    """
    Collate function for Bert.

    Parameters
    ----------
    batch : (x,y) tuple, where x is a tensor with input ids and y is a tensor with labels
    ----------
    Returns: (x,y) , where x has shape [sentence_length, batch_size] and y has shape [batch_size, number_of_labels]
    """
    # Set the maximum sequence length. In the original paper, the authors used a length of 512.
    MAX_LEN = 64

    inputs = [s[0] for s in batch]
    labels = [s[1][0] for s in batch]

    inputs_padded = pad_sequences(inputs, maxlen=MAX_LEN, dtype="long", truncating="post",
                                  padding="post")
    #inputs_padded = pad_sequence(inputs)

    x_ids = torch.tensor(inputs_padded)
    y = torch.tensor(labels)
    return x_ids, y





def save_plot(data, name, directory, is_val=False):
    plt.figure()
    plt.plot(data, label=f"{name}")
    plt.legend()
    plt.savefig(f"{directory}/{name}.png")
    plt.close()

def save_data(data_subset, set):
    '''
    Helper function to get the data back into csv format (so that it can be fed to SVM)
    '''

    alldata = []

    lbls = ['cleaned_tweet', 'cls', 'count', 'hate_speech', 'off_lang', 'neither']
    for i in data_subset:
        datalist = [" ".join(i[0])] + i[1].tolist()
        alldata.append(datalist)

    df = pd.DataFrame(alldata, columns=lbls)
    df.to_csv(f'../data/{set}.csv')



def main():
    torch.manual_seed(42)
    #some params
    experiment_number = 1
    test_percentage = 0.1
    val_percentage = 0.2
    batch_size= 10
    num_epochs = 5
    embedding_dim=300
    model_name = "CNN" #"CNN" #"Bert"
    embedding = "Both"#"Glove" # "Random" #
    soft_labels = False
    # Bert parameter
    if model_name == "Bert":
        embedding = "None"
    if embedding == "Both":
        combine = True
        embedding = "Random"
    else:
        combine =False
    learning_rate = 2e-5

    # load data
    dataset = Dataset("../data/cleaned_tweets_orig.csv", use_embedding=embedding, embedd_dim=embedding_dim, combine = combine)
    train_data, val_test_data = split_dataset(dataset, test_percentage + val_percentage )
    val_data, test_data = split_dataset(val_test_data, test_percentage/(test_percentage + val_percentage) )

    #save_data(train_data, 'train')
    #save_data(test_data, 'test')

    #define loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size , collate_fn= my_collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size , collate_fn= my_collate)

    #define model
    vocab_size = len(dataset.vocab)
    if model_name == "CNN":
        model = CNN(vocab_size, embedding_dim, combine=combine)
    elif model_name == "LSTM":
        model = LSTM(vocab_size, embedding_dim, batch_size = batch_size, combine=combine)
    elif model_name == "Bert":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   collate_fn=bert_collate)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 collate_fn=bert_collate)

    if not model_name=="Bert":
        model.embedding.weight.data.copy_(dataset.vocab.vectors)
        if combine:
            model.embedding_glove.weight.data.copy_(dataset.glove.vectors)
    #cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #optimiser
    optimizer = optim.Adam(model.parameters())
    if model_name=="Bert":
        optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
        # todo: Add scheduler
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    #weighted cross entropy loss, by class counts of other classess
    weights = torch.tensor([0.9414, 0.2242, 0.8344], device = device)
    #weights = torch.tensor([1.0, 1.0, 1.0], device = device) #get_loss_weights(train_data).to(device) # not to run again
    criterion = nn.CrossEntropyLoss(weight=weights)

    plot_log = defaultdict(list)
    for epoch in range(num_epochs):
        #train and validate
        epoch_loss, epoch_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        #save for plotting
        for name, point in zip(["train_loss", "train_accuracy", "val_loss", "val_accuracy"],[epoch_loss, epoch_acc, val_loss, val_acc]):
            plot_log[f'{name}'] = point
        #realtime feel
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {epoch_loss:.5f} | Train Acc: {epoch_acc*100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.5f} |  Val. Acc: {val_acc*100:.2f}%')
    #save plot
    results_directory = f'plots/{experiment_number}'
    os.makedirs(results_directory, exist_ok=True)
    for name, data in plot_log.items():
        save_plot(data, name, results_directory)
    #save model
    torch.save(model, os.path.join(results_directory, 'model_cnn.pth'))
    #confusion matrix and all that fun
    loss, acc, predictions, ground_truth = evaluate_epoch(model, val_loader, criterion, device, is_final=True)
    print(predictions, ground_truth)
    conf_matrix = confusion_matrix(ground_truth, predictions)
    class_report = classification_report(ground_truth, predictions)
    print('\nFinal Loss and Accuracy\n----------------\n')
    print(f'\t Val. Loss: {loss:.5f} |  Val. Acc: {acc*100:.2f}%')
    print('\nCONFUSION MATRIX\n----------------\n')
    print(conf_matrix)
    print('\nCLASSSIFICATION REPORT\n----------------------\n')
    print(class_report)


if __name__ == '__main__':

    main()