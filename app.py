import flask
from flask import render_template
import pickle
import sklearn
from sklearn.multioutput import MultiOutputRegressor

app = flask.Flask(__name__, template_folder='src/templates', static_folder='src')


@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('index.html')
    if flask.request.method == 'POST':
        with open('src/models/mor_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        input_data = flask.request.form[['iw', 'if', 'vw', 'fp']]
        y_pred = loaded_model.predict([[input_data]])
        
        return render_template('index.html', result = y_pred)
    
if __name__ == '__main__':
    app.run()