from app import app
from flask import render_template, request

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template('/public/index.html')

@app.route('/output', methods=['GET', 'POST'])
def get_output():
    print(request.form.get('news_article_text'))

    return render_template('/public/output.html')
