# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

# Step 1: Load dataset
data = pd.read_csv(r'C:/Users/mabbi/Desktop/PROJECTS/HOUSE PRICE/data/housing.csv')


# Step 2: Preprocessing
X = data[['size', 'location', 'rooms']]  # Features
y = data['price']  # Target

# Convert 'location' to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['location'], drop_first=True)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

save_dir = 'C:/Users/mabbi/Desktop/PROJECTS/HOUSE PRICE/models'


# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Step 6: Save the model
joblib.dump(model, 'C:/Users/mabbi/Desktop/PROJECTS/HOUSE PRICE/models/house_price_model.pkl')
print("Model saved to 'C:/Users/mabbi/Desktop/PROJECTS/HOUSE PRICEmodels/house_price_model.pkl'")
