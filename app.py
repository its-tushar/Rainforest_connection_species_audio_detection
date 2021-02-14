from flask import Flask, jsonify, request
import numpy as np
import pickle
#from keras.models import model_from_json
import pandas as pd
import datetime

from final_file import final
# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

final_object=final()
###################################################

###################################################


@app.route('/',methods=['GET'])
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    index=request.form.to_dict()['Enter_index']
    id,duration,predicted,time=final_object.predict(int(index))
    return flask.render_template('new.html',Recording_id=id,Audio_Duration=duration,predicted=predicted,time=time)

if __name__ == '__main__':
    app.run(debug=True)
