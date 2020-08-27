from flask import request
from flask import jsonify
from flask import Flask

#create instance of flask' class
app = Flask(__name__)

#specify app decorator
@app.route('/hello',methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting': 'Hello, ' + name + '!'
    }
    return jsonify(response)