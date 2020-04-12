import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import torch

from app.models import Pickle_model, Fake_news_classifier_model

class Preprocessor_controller():
    def __init__(self, vectorizer_path, device='cpu'):
        self.vectorizer_path = vectorizer_path
        self.device = device

        self.__init_stemmer()
        self.__init_vectorizer()

    def get_vectorizer_vocab_size(self):
        return len(self.vectorizer.vocabulary)

    def __init_stemmer(self):
        nltk.download('stopwords')

        stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        stop.update(punctuation)

        self.stopwords = stop
        self.stemmer = PorterStemmer()

    def __init_vectorizer(self):
        pickle_model = Pickle_model()

        vocabulary = pickle_model.get_pickle(path=self.vectorizer_path)
        self.vectorizer = CountVectorizer(stop_words='english', vocabulary=vocabulary)
        self.vectorizer._validate_vocabulary()

    def __stem_text(self, text):
        final_text = [self.stemmer.stem(word.strip()) for word in text.split() if word.strip().lower() not in self.stopwords]

        return " ".join(final_text)

    def __vectorize_text(self, text):
        return self.vectorizer.transform(text)

    def __csr_to_tensor(self, csr):
        coo = csr.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def transform(self, text_articles, article_titles):
        text = text_articles if isinstance(text_articles, list) else [text_articles]
        titles = article_titles if isinstance(article_titles, list) else [article_titles]

        inputs = [title + " " + text for (title, text) in zip(titles, text)]

        stemmed_text = [self.__stem_text(text) for text in inputs]
        print("Stemmed_text: {}".format(stemmed_text))
        vector = self.__vectorize_text(stemmed_text)
        print("Vector: {}".format(vector))
        tensor = self.__csr_to_tensor(vector)
        print("Tensor: {}".format(tensor))
        return tensor.to(self.device)

class Fake_news_classifier_controller():
    def __init__(self, model_path, in_features, device='cpu'):
        self.model_path = model_path
        self.model = Fake_news_classifier_model(in_features=in_features).to(device)
        self.model.load_model(path=model_path, device=device)
        self.model.eval()

    def __predict_tensor(self, tensor):
        pred = self.model(tensor)
        return pred

    def predict(self, tensor):
        prob_list = self.__predict_tensor(tensor).tolist()

        print("Confidence: {}".format(prob_list))

        return [{'real':{'label':'Real', 'value':real_prob}, 'fake':{'label':'Fake', 'value':fake_prob}} for real_prob, fake_prob in prob_list]