import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences

# following https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

tweets = ['here it is', 'my vocabulary of tweets', 'and another tweet', 'i like tweeting sooo much']
labels = [0, 1, 1, 2]

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# these tokens are needed(?)
tweets = ["[CLS] " + tweet + " [SEP]" for tweet in tweets]

# tokenize the tweets
tokens = [tokenizer.tokenize(tweet) for tweet in tweets]

indexed_tokens = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokens],
                          maxlen=30, dtype="long", truncating="post", padding="post")

segments_ids = [[1] * len(txt) for txt in indexed_tokens]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long)
segments_tensors = torch.tensor(segments_ids, dtype=torch.long)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)

print ("Number of layers:", len(encoded_layers))
layer_i = 0

print ("Number of batches:", len(encoded_layers[layer_i]))
batch_i = 0

print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))
