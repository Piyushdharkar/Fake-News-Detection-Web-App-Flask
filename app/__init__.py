from flask import Flask
from app.controllers import Preprocessor_controller, Fake_news_classifier_controller

app = Flask(__name__)

app.config['MODEL_DIR'] = 'C:/Users/Piyush/PycharmProjects/FakeNewsDetection/app/static/model/'
app.config['TORCH_DEVICE'] = 'cpu'

preprocessor_controller = Preprocessor_controller(vectorizer_path=app.config['MODEL_DIR'] + 'vectorizer_vocab.pkl', device=app.config['TORCH_DEVICE'])
app.preprocessor_controller = preprocessor_controller
app.fake_news_classifier_controller = Fake_news_classifier_controller(model_path=app.config['MODEL_DIR'] + 'model_state_dict.pth', in_features=preprocessor_controller.get_vectorizer_vocab_size(), device=app.config['TORCH_DEVICE'])

from app import views