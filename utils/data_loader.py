import torch
import torch.utils.data as data
import pandas as pd
import torchtext as text
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS

class Dataset(data.Dataset):
    """
    Dataset representing https://github.com/tpawelski/hate-speech-detection
    """

    def __init__(self, csv_path, use_cleaned=True, use_embedding="None", embedd_dim=300,
                 rm_stop_words=False):
        """
        :param csv_path: Path to csv file
        :param use_cleaned: Returns tweets without punctuation and converted to lower-case
        :param use_embedding: "None", "Glove", "Random"
            If "None": Tweets are returned as a list of strings (tokens)
            If "Glove": Tweets are returned as a list of indices. The Glove vocabulary can
            be accessed using the .vocab property
            If "Random": Tweets are returned as a list of indices. A new vocabulary is built
        :param embedd_dim: size of embeddings
        :param rm_stop_words: Whether stop words shall be removed from tweets
        """
        self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        self.use_cleaned = use_cleaned
        self.use_embedding = use_embedding
        self.rm_stop_words = rm_stop_words
        self.textProcesser = text.data.Field(sequential=True, use_vocab=True, init_token=None,
                                             eos_token=None, fix_length=None, dtype=torch.int64,
                                             preprocessing=None, postprocessing=None, lower=False,
                                             tokenize=None, tokenizer_language='en',
                                             include_lengths=False, batch_first=False,
                                             pad_token='<pad>', unk_token='<unk>', pad_first=False,
                                             truncate_first=False, stop_words=None,
                                             is_target=False)

        if use_embedding == "None":
            self._vocab = None
        elif use_embedding == "Glove":
            # Multiple versions of the GloVe embedding are available
            # Check https://pytorch.org/text/vocab.html#torchtext.vocab.Vocab.load_vectors
            self._vocab = self._build_pretrained_vocab(self.textProcesser, self.df,
                                                       dim=embedd_dim,
                                                       vectors=f"glove.6B.{embedd_dim}d")
        elif use_embedding == "Random":
            # Todo: Only use training data, not entire vocab
            self._vocab = self._build_rnd_vocab(self.textProcesser, self.df, dim=embedd_dim)
        else:
            raise AttributeError("Value for attribute 'use_embedding' is not supported.")

        self.textProcesser.vocab = self._vocab

    def __getitem__(self, index):
        """
        index: the unique identifier of the tweet
        tweet: the tweet, in textual form
        clean_tweet: the text of the tweet after removing punctuation and converting to lower-case
        class: the majority label (0: hate speech, 1: offensive language, 2: neither)
        count: the total # of CrowdFlower users who labeled the tweet
        hate_speech: the # of users who labeled the tweet as hate speech
        offensive_language: the # of users who labeled the tweet as offensive language
        neither: the # of users who labeled the Tweet as neither hate speech nor offensive language

        Returns the tweet (list of strings if use_embedding="None", or list of vectors otherwise)
        and the label (torch.tensor: class, count, hate_speach, off_lang, neither)
        """
        index, count, hate_speach, off_lang, neither, cls, tweet, clean_tweet = self.df.iloc[index]
        lbl = torch.Tensor([cls, count, hate_speach, off_lang, neither])

        if self.use_cleaned:
            example = clean_tweet
        else:
            example = tweet

        # Tokenize (Convert it to list of strings)
        example = self.textProcesser.preprocess(example)

        # Remove Stop Words
        if self.rm_stop_words:
            example = [x for x in example if x not in STOP_WORDS]

        # Numericalize (Convert it to list of indices)
        if self.vocab:
            example = self.textProcesser.numericalize([example])

        return example, lbl

    def __len__(self):
        return len(self.df)

    @property
    def vocab(self):
        """
        The torchtext.vocab object that can be used to init the embedding layer.
        Is None, if the dataset object was initiliased with use_embedding="None".
        """
        return self._vocab

    @staticmethod
    def _build_rnd_vocab(text_processer: text.data.Field, df, dim):
        # Let's only use the cleaned tweets for building the vocabulary
        clean_tweets = df["clean_tweet"].values

        # Tokenize them
        tokenized = [text_processer.preprocess(x) for x in clean_tweets]

        text_processer.build_vocab(tokenized, max_size=None, min_freq=1,
                                   specials=['<unk>', '<pad>'], vectors=None, unk_init=None,
                                   vectors_cache=None, specials_first=True)
        text_processer.vocab.dim = dim

        # Init vectors randomly
        # std = 0.05 is based on the norm of average GloVE 100-dim word vectors
        n = torch.distributions.Normal(0, 0.05)
        text_processer.vocab.vectors = n.sample((len(text_processer.vocab), dim))
        return text_processer.vocab

    @staticmethod
    def _build_pretrained_vocab(text_processer: text.data.Field, df, dim, vectors):
        # Let's only use the cleaned tweets for building the vocabulary
        clean_tweets = df["clean_tweet"].values

        # Tokenize them
        tokenized = [text_processer.preprocess(x) for x in clean_tweets]
        text_processer.build_vocab(tokenized, max_size=None, min_freq=1,
                                   specials=['<unk>', '<pad>'], vectors=vectors,
                                   unk_init=None, vectors_cache=None, specials_first=True)
        text_processer.vocab.dim = dim
        return text_processer.vocab

    def split_train_test_scikit(self):

        """
        Split data into train and test (also separating tweet and label i.e. X and y) for use in scikit
        """

        if self.use_cleaned:
            X = self.df['clean_tweet']
        else:
            X = self.df['tweet']

        y = self.df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test

    def split_train_test_torch(self):
        #todo: finish

        """
        Split data into train and test (also separating tweet and label i.e. X and y) for use with torch
        """

        if self.use_cleaned:
            X = self.df['clean_tweet']
        else:
            X = self.df['tweet']

        y = self.df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test

def get_loader(csv_path, use_cleaned=True, batch_size=100):
    """
    Returns the PyTorch DataLoader
    :param csv_path:
    :param use_cleaned:
    :param batch_size:
    """
    dataset = Dataset(csv_path, use_cleaned)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return data_loader


if __name__ == '__main__':
    # Example of using the dataset class (with cleaned tweets)
    ds = Dataset("../data/cleaned_tweets_orig.csv", use_cleaned=True)
    print(f"Dataset contains {len(ds)} examples.")
    print("Datapoint at index 15:")
    example, label = ds[15]
    print("Tweet is a list of tokens: ", example)
    print("Label: ", label)
    print("=" * 50)

    # Let's try using the dataset with Glove embeddings
    ds_glove = Dataset("../data/cleaned_tweets_orig.csv", use_embedding="Glove")
    example, label = ds_glove[15]
    print("Tweet is now a list of indices:", example)
    print("Label: ", label)
    # We can access the Glove vocab:
    vocab = ds_glove.vocab
    # use it to get the index of a word
    print("index of hello: ", vocab.stoi["hello"])
    # or get the corresponding word of an index
    print("Word behind index 56: ", vocab.itos[56])
    print("=" * 50)

    # Let's use the dataset class with our own embeddings:
    ds = Dataset("../data/cleaned_tweets_orig.csv", use_embedding="Random")
    print("Size of new vocabulary: ", len(ds.vocab))
    print("Some entries of the vocabulary: ", ds.vocab.itos[:10])
    print("Index of <pad> is: ", ds.vocab.stoi[" <pad>"])

    # How to init embedding layer with vocab
    import torch.nn as nn

    embed = nn.Embedding(len(ds.vocab), embedding_dim=ds.vocab.dim)
    embed.weight.data.copy_(ds.vocab.vectors)
