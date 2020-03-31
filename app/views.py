from app import app

@app.route('/', methods=['GET'])
def home():
    return 'Test home page'
