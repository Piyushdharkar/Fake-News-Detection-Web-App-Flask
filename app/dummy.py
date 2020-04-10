from app.controllers import Preprocessor_controller

path = 'C:/Users/Piyush/PycharmProjects/FakeNewsDetection/app/static/model/'
pp = Preprocessor_controller(vectorizer_path=path + 'vectorizer_vocab.pkl')
tensor = pp.transform('Hello! I hope you are well.')

