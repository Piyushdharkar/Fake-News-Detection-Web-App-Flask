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
    result = fake_news_classifier_controller.predict(text=news_article_text)
    print(result)
    bar_plot_controller = current_app.bar_plot
    bar_plot_controller.fit(dataset=result, x='class', y='probability', fig_size=(15, 15))
    fig = bar_plot_controller.draw()
    print(fig)

    return render_template('/public/output.html')
