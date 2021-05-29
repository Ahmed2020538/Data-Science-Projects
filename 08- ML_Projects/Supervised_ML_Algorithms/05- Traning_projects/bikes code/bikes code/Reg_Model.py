import joplib

Model      = joplib.load("Model1.h5") 
Scaller    = joplib.load("Scaller.h5")

cus_data   = np.array([])

cus_data   = Scaller.transform(cus_data)

prediction = Model.predict(cus_data)

print(f"The Prediction :: {prediction[0]}")
