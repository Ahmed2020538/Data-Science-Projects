from flask import Flask , render_template , request
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load('models/Model.h5')
scaler = joblib.load('models/tfidf.h5')




@app.route("/" , methods=["GET"])
def home():
    return render_template("index.html")




@app.route("/predict" , methods =["GET"])
def predict() :

    all_data = request.args
    
    inp_data = all_data
        
    News = model.predict(scaler.transform(inp_data))[0]
    
    return render_template("index.html" , News = News)
    











if __name__ == '__main__':
    app.run(debug = True , host="127.0.0.1")

