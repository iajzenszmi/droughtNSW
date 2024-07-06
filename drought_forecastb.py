import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to simulate data
def simulate_data(start_date, end_date):
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n_samples = len(dates)
    precipitation = np.random.uniform(0, 300, n_samples)  # in mm
    temperature = np.random.uniform(10, 35, n_samples)  # in Celsius
    soil_moisture = np.random.uniform(0, 100, n_samples)  # in %
    
    # Drought index (simulated using a more realistic approach)
    drought_index = (precipitation - np.mean(precipitation)) / np.std(precipitation) - \
                    (temperature - np.mean(temperature)) / np.std(temperature) + \
                    (soil_moisture - np.mean(soil_moisture)) / np.std(soil_moisture)
    
    data = pd.DataFrame({
        'Date': dates,
        'Precipitation': precipitation,
        'Temperature': temperature,
        'Soil_Moisture': soil_moisture,
        'Drought_Index': drought_index
    })
    return data

# Simulate data
data = simulate_data("2000-01-01", "2023-12-31")

# Prepare the dataset
X = data[['Precipitation', 'Temperature', 'Soil_Moisture']]
y = data['Drought_Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Get the test set dates
test_dates = data.iloc[y_test.index]['Date'].values

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test.values, label='Actual Drought Index', color='b')
plt.plot(test_dates, y_pred, label='Predicted Drought Index', color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Drought Index')
plt.title('Drought Forecasting for NSW')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
