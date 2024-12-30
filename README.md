# **House Price Prediction**

## **Overview**
This project demonstrates how to build a simple **House Price Prediction** model using **Python** and **scikit-learn**. The model predicts house prices based on features such as the size of the house (in square feet), the location (urban/rural), and the number of rooms. It uses a regression algorithm to understand the relationships between these features and the price.

---

## **Features**
- **Exploratory Data Analysis (EDA):** Analyzing and visualizing the dataset.
- **Model Training:** Using Linear Regression to train a predictive model.
- **Prediction:** Predicting house prices for new data.
- **Dataset:** A small dataset with fictional house details for simplicity.

---

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:**
  - `pandas` for data manipulation
  - `scikit-learn` for machine learning
  - `matplotlib` for data visualization
  - `joblib` for saving and loading the trained model

---

## **Folder Structure**
```plaintext
house-price-prediction/
├── data/
│   └── housing.csv       # Dataset file
├── scripts/
│   └── train_model.py    # Script to train the model
│   └── predict.py        # Script to make predictions
├── models/
│   └── house_price_model.pkl  # Saved trained model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

## **Dataset**
The dataset is saved as data/housing.csv and contains the following columns:

size: The size of the house in square feet.
location: Whether the house is located in an urban or rural area.
rooms: The number of rooms in the house.
price: The price of the house in dollars.

Sample Dataset
csv
size,location,rooms,price
1200,urban,3,200000
800,rural,2,100000
1500,urban,4,250000
950,rural,3,120000
1800,urban,5,300000

## **Setup and Installation**

-->Clone the repository:
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

-->Install the required dependencies:
pip install -r requirements.txt

-->Ensure the dataset (housing.csv) is in the data/ folder.

##**How to Run the Project**
1. Train the Model
Run the train_model.py script to train the regression model:

python scripts/train_model.py
Output Example:
Mean Squared Error: 250000000.0
Model saved to '../models/house_price_model.pkl'

2. Make Predictions
Run the predict.py script to predict house prices for new data:

python scripts/predict.py
Output Example:
Predicted Price: $220,000.00

##**Project Workflow**

-->Data Preprocessing:
Load the dataset.
Encode categorical data (location) using one-hot encoding.
Split the dataset into training and testing sets.

-->Model Training:
Train a Linear Regression model using the training data.
Evaluate the model using the test set and calculate the Mean Squared Error (MSE).

-->Prediction:
Use the trained model to predict prices for new house data.

##**Results**
The trained model achieves reasonable accuracy on the test set, with a low mean squared error (depending on the dataset).
It can accurately predict house prices for given inputs.
