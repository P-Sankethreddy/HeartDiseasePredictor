import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

from sklearn.preprocessing import StandardScaler

stdScaler=StandardScaler()


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


# dataset[columns_for_scaling]=stdScaler.fit_transform(dataset[columns_for_scaling])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])    
def predict():
    features=[]
    trans=[]
    features.append(request.form.get('age'))
    features.append(request.form.get('trestbps'))
    features.append(request.form.get('chol'))
    features.append(request.form.get('thalach'))
    features.append(request.form.get('oldpeak'))
    features.append(request.form.get('thal'))
  
  
        
    if request.form.get('sex') is "male":
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(1)
    if request.form.get('cp') is 0:
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
    elif request.form.get('cp') is 1:
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
    elif request.form.get('cp') is 2:
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
    if request.form.get('fbs') is 0:
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(1)
    if request.form.get('restecg') is 0:
        features.append(1)
        features.append(0)
        features.append(0)
        
    elif request.form.get('restecg') is 1:
        features.append(0)
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(0)
        features.append(1)
    if request.form.get('exang') is 0:
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(1)  
    if request.form.get('slope') is 0:
        features.append(1)
        features.append(0)
        features.append(0)
    elif request.form.get('slope') is 1:
        features.append(0)
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(0)
        features.append(1)  
    if request.form.get('ca') is 0:
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
    elif request.form.get('ca') is 1:
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
    elif request.form.get('ca') is 2:
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
    elif request.form.get('ca') is 3:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)           
            


    
    final_features=[np.array(features)]
   
    
    prediction=model.predict(final_features)
   
    return render_template('index.html',prediction_text="Patient have Heart disease" if prediction == 1  else "Patient not have heart disease")        



      




if __name__=="__main__":
    app.run(debug=True)

