from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def churn_prediction():
    CreditScore = int(request.form.get('creditscore'))
    Geography = int(request.form.get('geography'))  
    Gender = int(request.form.get('gender'))  
    Age = int(request.form.get('age'))
    Tenure = int(request.form.get('tenure'))
    Balance = float(request.form.get('balance'))
    NumOfProducts = int(request.form.get('numofproducts'))
    HasCrCard = float(request.form.get('hascrcard'))
    IsActiveMember = float(request.form.get('isactivemember'))
    EstimatedSalary = float(request.form.get('estimatedsalary'))

    # Prediction
    result = model.predict(np.array([CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]).reshape(1,10))
   
    if result[0] == 1:
        result = 'Exited'
    else:
        result = 'Not Exited'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(debug = True)