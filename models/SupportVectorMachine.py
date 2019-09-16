import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report


sys.path.append('../utils')

from data_loader import Dataset

class BaselineClassifier:

    """
    Calculate tf-idf on test data and classifies using an SVM.

    Parameters
    ==========

    """

    def __init__(self, ngram_range=2, max_vocab_f=15000,
        alpha=0.0001, min_df=3):

        self.vectorizer = None
        self.model = None

        # parameters for tf-idf vectorizer
        self.ngram_range = (1, ngram_range)
        self.max_vocab_f = max_vocab_f
        self.min_df = min_df

        # parameter for SVM
        self.alpha = alpha

    def train(self, x_train, y_train):

        """
        Trains a logistic classifier.
        """

        tweets = list(x_train)

        vec = TfidfVectorizer(ngram_range=self.ngram_range,
                            max_features=self.max_vocab_f,
                            strip_accents='unicode')

        # generate term document matrix (model inputs)
        X = vec.fit_transform(tweets)

        # SVM classifier
        classifier = SGDClassifier(loss="hinge", alpha=self.alpha).fit(X, y_train)

        # save the model and vectorizer
        self.model = classifier
        self.vectorizer = vec

    def predict(self, tweets):

        """
        Generates predictions from the trained classifier.
        """

        tweets = list(tweets)

        # get vectorizer and determine tfidf for tweets
        vec = self.vectorizer
        X = vec.transform(tweets)

        # get the classifier
        classifier = self.model

        # score tweets
        scores = classifier.predict(X)

        return scores


if __name__ == '__main__':

    # load data
    dataset = Dataset("../data/cleaned_tweets_orig.csv")
    x_train, x_test, y_train, y_test = dataset.split_train_test_scikit()

    classifier = BaselineClassifier()
    classifier.train(x_train, y_train)
    scores = classifier.predict(x_test)

    conf_matrix = confusion_matrix(y_test, scores)
    class_report = classification_report(y_test, scores)

    print('\nCONFUSION MATRIX\n----------------\n')
    print(conf_matrix)

    print('\nCLASSSIFICATION REPORT\n----------------------\n')
    print(class_report)
