from app import app
from flask import render_template, request, current_app, jsonify
from app.exceptions import Invalid_usage

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template('/public/index.html')

@app.errorhandler(Invalid_usage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/predict', methods=['POST'])
def predict():
    request_dict = request.values.to_dict()

    if request_dict is None:
        raise Invalid_usage('Cannot read json object.')
    print("Request dict: {}".format(request_dict))
    if 'text' not in request_dict or 'title' not in request_dict:
        raise Invalid_usage('Cannot find key in json object.')
    news_article_text = request_dict['text']
    news_article_title = request_dict['title']
    if not isinstance(news_article_text, str) or not isinstance(news_article_title, str):
        raise Invalid_usage('Posted data is not string.')
    print('text: {}'.format(news_article_text))
    print('title: {}'.format(news_article_title))
    if news_article_text.strip() == '' or news_article_title.strip() == '':
        raise Invalid_usage('Posted data is empty')

    preprocessor_controller = current_app.preprocessor_controller
    tensor = preprocessor_controller.transform(text_articles=news_article_text, article_titles=news_article_title)

    fake_news_classifier_controller = current_app.fake_news_classifier_controller
    result = fake_news_classifier_controller.predict(tensor=tensor)
    print("Response json: {}".format(result))

    return jsonify(result)

