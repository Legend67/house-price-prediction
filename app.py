from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_price(AveRooms, AveBedrms, AveOccup):

    input_data = pd.DataFrame([[AveRooms, AveBedrms, AveOccup]],
                            columns=['AveRooms', 'AveBedrms', 'AveOccup'])
    

    input_scaled = scaler.transform(input_data)
    

    predicted_price = model.predict(input_scaled)[0]
    

    return round(predicted_price, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collecting the data
        AveRooms = float(request.form['AveRooms'])
        AveBedrms = float(request.form['AveBedrms'])
        AveOccup = float(request.form['AveOccup'])
        

        predicted_price = predict_price(AveRooms, AveBedrms, AveOccup)
        
        # Render result template with prediction
        return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
