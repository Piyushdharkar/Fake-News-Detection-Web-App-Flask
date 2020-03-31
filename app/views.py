from app import app
from flask import render_template, request, current_app

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template('/public/index.html')

@app.route('/output', methods=['GET', 'POST'])
def get_output():
    news_article_text =  request.form.get('news_article_text')
    print('text: ' + news_article_text)
    fake_news_classifier_controller = current_app.fake_news_classifier
    real_prob, fake_prob = fake_news_classifier_controller.predict(news_article_text)

    print(real_prob, fake_prob)

    return render_template('/public/output.html')
