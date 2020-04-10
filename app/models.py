import pickle

import torch
import torch.nn as nn

class Pickle_model():
    def get_pickle(self, path):
        return pickle.load(open(path, 'rb'))

class Fake_news_classifier_model(nn.Module):
    def __init__(self, in_features):
        super(Fake_news_classifier_model, self).__init__()

        self.dense1 = nn.Linear(in_features, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 16)
        self.dense5 = nn.Linear(16, 2)

    def load_model(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return torch.softmax(x, dim=-1)