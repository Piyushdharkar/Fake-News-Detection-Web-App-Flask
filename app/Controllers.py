import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Fake_news_classifier_controller():
    def __init__(self):
        pass

    def predict(self, text):
        return pd.DataFrame({'class':['Real', 'Fake'], 'probability': [0.3, 0.7]})
