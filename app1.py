from flask import Flask, render_template, request 
import pickle 
import numpy as np


app=Flask(__name__)
model=pickle.load(open("villa_price_prediction.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    form = request.form
    # Create a feature dict with all columns set to 0
    features_dict = {col: 0 for col in model_columns}
    # Fill in values from form
    features_dict['beds'] = float(form.get('beds', 0))
    features_dict['bathrooms'] = float(form.get('bathroom', 0))
    features_dict['bedrooms'] = float(form.get('bedrooms', 0))
    features_dict['latitude'] = float(form.get('lat', 0))
    features_dict['longitude'] = float(form.get('long', 0))
    features_dict['has_wireless_internet'] = 1 if form.get('has_wireless_internet') == '1' else 0
    features_dict['has_air_conditioning'] = 1 if form.get('has_air_conditioning') == '1' else 0
    # Build feature array in correct order
    final_features = np.array([ [features_dict[col] for col in model_columns] ])
    prediction = model.predict(final_features)
    actual_price= np.exp(prediction[0])
    return render_template('index.html', prediction_text=f'Estimated Villa Prediction Price: {actual_price:,.2f}')
    

    


if __name__=="__main__":
    app.run(debug=True)
