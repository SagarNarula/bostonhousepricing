import pickle
from flask import Flask,render_template,request,app,jsonify,url_for
import numpy as np
import pandas as pd

app=Flask(__name__)

### Load the pickle file
model=pickle.load(open("regmodel.pkl","rb"))
scale_pkl=pickle.load(open("scaling.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scale_pkl.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scale_pkl.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Predicted house price is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
    
