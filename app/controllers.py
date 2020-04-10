import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Fake_news_classifier_controller():
    def __init__(self):
        pass

    def predict(self, text):
        return {'Real':0.7, 'Fake':0.3}
