from flask import Flask
from app.controllers import Fake_news_classifier_controller

app = Flask(__name__)

app.fake_news_classifier = Fake_news_classifier_controller()

from app import views