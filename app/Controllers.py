import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Fake_news_classifier_controller():
    def __init__(self):
        pass

    def predict(self, text):
        return pd.DataFrame({'class':['Real', 'Fake'], 'probability': [0.3, 0.7]})

class Bar_plot_controller():
    def __init__(self):
        pass


    def fit(self, dataset, x, y, fig_size=(15, 15)):
        self.dataset = dataset
        self.x = x
        self.y = y
        self.fig_size = fig_size

    def draw(self):
        fig = plt.figure(figsize=self.fig_size)
        sns.barplot(x=self.x, y=self.y, data=self.dataset)
        return fig