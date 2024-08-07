# House Price Prediction Web Application Using Flask

## Overview
This project is a web application designed to predict house prices based on user input using machine learning models. Built with Flask, the application leverages various machine learning algorithms to estimate housing prices based on features such as average number of rooms, bedrooms, and occupancy. The application is user-friendly, with a clean interface that allows users to input their data and receive predictions in real-time.

## Project Structure
```bash
house-price-prediction/
│
├── app.py                   # Flask application code
├── model.py                 # Model training and prediction logic
├── house_price_model.pkl    # Saved machine learning model
├── scaler.pkl               # Saved feature scaler
├── requirements.txt         # Python dependencies
├── static/
│   └── style.css            # CSS for styling the web pages
├── templates/
│   ├── index.html           # HTML form for user input
│   └── result.html          # HTML page to display prediction results
└── README.md                # Project documentation
```

## Features
- Input Form: Users can enter features such as average number of rooms, bedrooms, and occupancy to predict house prices.
- Prediction: Utilizes trained machine learning models to estimate house prices based on the provided features.
- Result Display: Shows the predicted price on a separate page in a user-friendly format.

## Machine Learning Model Used
### Random Forest Regressor
- Description: An ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees.
- Usage: Offers robust performance by reducing overfitting and handling complex relationships between features and target variable.

## Setup Instructions
### Prerequisites
- Python 3.6 or higher
- pip for installing Python packages

### Installation
1. Clone the Repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```
2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Download or Train the Model

If the model files (**house_price_model.pkl** and **scaler.pkl**) are not provided, you can train and save them by running:

```bash
python model.py
```

This will create the necessary model and scaler files.

## Running the Application
1. Start the Flask Server
```bash
python app.py
```

2. Access the Web Application

Open your web browser and go to http://127.0.0.1:5000/. You will see the input form.

3. Make Predictions

Enter values for the average number of rooms, bedrooms, and occupancy, then submit the form to receive a predicted house price.

## Code Overview
#### `app.py`

This file contains the Flask application code, including routes for displaying the input form and handling predictions. It integrates with the trained machine learning model and scaler to provide predictions.

#### `model.py`

This script handles the training of machine learning models using the California housing dataset. It saves the trained models and scalers to files for use in the Flask application.

##### `static/style.css`

Contains styling for the web application to ensure a modern and user-friendly interface.

#### `templates/`
- **`index.html:`** HTML form for users to input their house features.
- **`result.html:`** Displays the predicted house price.

## Screenshots

1. ### Server
![image](https://github.com/user-attachments/assets/559a1860-d5f7-428d-99e6-9c2d6d19ceb2)

2. ### Home Page
![image](https://github.com/user-attachments/assets/a08f0aee-a7b0-4d34-b235-91a261602985)

3. ### Sample Data 
![image](https://github.com/user-attachments/assets/e6cc5603-3c47-49b3-b25f-afb090f5bd31)

4. ### Prediction Result
![image](https://github.com/user-attachments/assets/accf0a45-14d0-40de-959e-cedb72d690cc)

## Dependencies
Key packages used:

- **`Flask`** for the web framework
- **`scikit-learn`** for machine learning algorithms
- **`pandas`** for data manipulation
- **`joblib`** for saving and loading models
  
Dependencies are listed in requirements.txt.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- **California Housing Dataset:** Used for training the machine learning models.
- **Scikit-Learn:** Provides machine learning algorithms and tools.
- **Flask:** Framework for building the web application.
