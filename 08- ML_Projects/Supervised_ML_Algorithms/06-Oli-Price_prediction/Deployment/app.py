import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('models/Model.h5')
scaler = joblib.load('models/scaler.h5')



@app.route('/', methods=['GET'])

def home():

    return render_template("index.html")




@app.route('/predict', methods=['GET'])

def predict():



    all_data = request.args

    Years = int(all_data['date'].split('-')[0])
    Moth  = int(all_data['date'].split('-')[1])
    Day   = int(all_data['date'].split('-')[2])
    
    inp_data = [Years , Moth , Day]
        
    Oil_Price = model.predict(scaler.transform([inp_data]))[0]
    
    return render_template("index.html" , Oil_Price = Oil_Price)















if __name__ == '__main__':
    app.run(debug = True , host="127.0.0.1")
