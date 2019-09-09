import torch
import torch.utils.data as data
import pandas as pd


class Dataset(data.Dataset):
    """
    Dataset representing https://github.com/tpawelski/hate-speech-detection
    """

    def __init__(self, csv_path, use_cleaned=True):
        """
        Reads tweets from csv file
        """
        self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        self.use_cleaned = use_cleaned

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

        Returns the tweet (string) and
        label (torch.tensor: class, count, hate_speach, off_lang, neither)
        """
        index, count, hate_speach, off_lang, neither, cls, tweet, clean_tweet = self.df.iloc[index]
        label = torch.Tensor([cls, count, hate_speach, off_lang, neither])

        if self.use_cleaned:
            return clean_tweet, label
        else:
            return tweet, label

    def __len__(self):
        return len(self.df)


def get_loader(csv_path, use_cleaned=True, batch_size=100):
    """
    Returns the PyTorch DataLoader
    :param csv_path:
    :param use_cleaned:
    :param batch_size:
    """
    dataset = Dataset(csv_path, use_cleaned)

    # Todo: implement custom collate_fn
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return data_loader


if __name__ == '__main__':
    ds = Dataset("../data/cleaned_tweets_orig.csv")
    t = ds[15]
    pass
