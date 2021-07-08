import joblib
from Encoding.dummies import *
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('models/Model.h5')
scaler = joblib.load('models/scaler.h5')



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    all_data       = request.args
    SeniorCitizen  = int(all_data['SeniorCitizen'])
    tenure         = int(all_data['tenure'])
    MonthlyCharges = float(all_data['MonthlyCharges'])
    TotalCharges   = float(all_data['TotalCharges'])

    
    gender           = Gender_dumm[all_data['gender']]
    Partner          = Partner_dumm[all_data['Partner']]
    Dependents       = Dependents_dumm[all_data['Dependents']]
    PhoneService     = PhoneService_dumm[all_data['PhoneService']]
    MultipleLines    = MultipleLines_dumm[all_data['MultipleLines']]
    InternetService  = InternetService_dumm[all_data['InternetService']]
    OnlineSecurity   = OnlineSecurity_dumm[all_data['OnlineSecurity']]
    OnlineBackup     = OnlineBackup_dumm[all_data['OnlineBackup']]
    DeviceProtection = DeviceProtection_dumm[all_data['DeviceProtection']]
    TechSupport      = TechSupport_dumm[all_data['TechSupport']]
    StreamingTV      = StreamingTV_dumm[all_data['StreamingTV']]
    StreamingMovies  = StreamingMovies_dumm[all_data['StreamingMovies']]
    Contract         = Contract_dumm[all_data['Contract']]
    PaperlessBilling = PaperlessBilling_dumm[all_data['PaperlessBilling']]
    PaymentMethod    = PaymentMethod_dumm[all_data['PaymentMethod']]





    x = [SeniorCitizen, tenure, MonthlyCharges, TotalCharges, gender , Partner , Dependents , PhoneService]
    x += MultipleLines + InternetService + OnlineSecurity + OnlineBackup + DeviceProtection 
    x +=  TechSupport + StreamingTV + StreamingMovies + Contract + PaperlessBilling + PaymentMethod


    x = scaler.transform([x])
    churn  = model.predict(x)[0]

    return render_template('prediction.html', Churn = churn)






if __name__ == '__main__':
    app.run(debug = True , host="127.0.0.1")
