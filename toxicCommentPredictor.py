# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

# from _future_ import print_function
import io
import os
import json
import pickle
from keras.models import load_model
from io import StringIO
import sys
import signal
import traceback
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
from sklearn.externals import joblib
import flask
from keras.preprocessing import text, sequence


import pandas as pd

prefix = '/opt/ml/'
model_path = prefix;

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = load_model('./toxicCommentPred.h5')
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        jdata = json.loads(flask.request.data)
        df = pd.DataFrame(jdata)
        numpArr = np.array([df.Value.values[0]])
        print(numpArr)
        max_features = 30000
        maxlen = 100
        embed_size = 300
        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(numpArr))
        print(tokenizer.word_index)
        x_test = tokenizer.texts_to_sequences(numpArr)
        x_test = np.array(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        x_test = np.reshape(x_test, (100, 1)).T
        data = x_test    
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))
    # Do the prediction
    predictions = ScoringService.predict(data)
    # Convert from numpy back to CSV
    resp = pd.Series(predictions.flatten()).to_json(orient='values')
    return flask.Response(response=resp, status=200, mimetype='application/json')