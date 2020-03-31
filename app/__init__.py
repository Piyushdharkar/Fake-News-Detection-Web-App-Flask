from flask import Flask
from app.Controllers import Fake_news_classifier_controller, Bar_plot_controller

app = Flask(__name__)

app.fake_news_classifier = Fake_news_classifier_controller()
app.bar_plot = Bar_plot_controller()

from app import views