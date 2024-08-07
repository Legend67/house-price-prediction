import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the California housing data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# Select features and target
X = df[['AveRooms', 'AveBedrms', 'AveOccup']]
y = df['Target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Save the trained model and scaler to files
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predict on test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

def predict_price(AveRooms, AveBedrms, AveOccup):
    # Load the trained model and scaler
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame([[AveRooms, AveBedrms, AveOccup]],
                            columns=['AveRooms', 'AveBedrms', 'AveOccup'])
    
    # Standardize the input features
    input_scaled = scaler.transform(input_data)
    
    # Predict the price using the model
    predicted_price = model.predict(input_scaled)[0]
    
    # Round the predicted price to 2 decimal places
    return round(predicted_price, 4)
