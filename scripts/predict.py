# scripts/predict.py

import pandas as pd
import joblib

# Step 1: Load the trained model
model = joblib.load('C:/Users/mabbi/Desktop/PROJECTS/HOUSE PRICE/models/house_price_model.pkl')



# Step 2: Prepare new data for prediction
new_data = pd.DataFrame({
    'size': [1400],  # Example size in sqft
    'location': ['urban'],  # Location (urban or rural)
    'rooms': [3]  # Number of rooms
})
expected_features = model.feature_names_in_

# Check if 'location_urban' is in the expected features
for feature in expected_features:
    if feature not in new_data.columns:
        new_data[feature] = 0
# Convert 'location' to numerical (one-hot encoding)
new_data = pd.get_dummies(new_data, columns=['location'], drop_first=True)

# Step 3: Predict house price
predicted_price = model.predict(new_data)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
