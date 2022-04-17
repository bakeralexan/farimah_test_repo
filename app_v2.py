from flask import Flask, render_template, request
import numpy as np
import joblib
from joblib import load
import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
import uuid
from input_v2 import Model


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        model = joblib.load("model.pkl")
        

        
        # Get values through input bars
        ticker = request.form.get("Ticker")
        SPF = Model()
        data = Model.extract_data(ticker)
        X = Model.reshape()


        # Get prediction
        prediction = model.predict(X)
        
    else:
        prediction = ""
        
    return render_template("index.html", output = prediction)


# Running the app
if __name__ == '__main__':
    app.run(debug = True)