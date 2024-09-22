


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Load the dataset
df = pd.read_csv('sp500_daily_data_last_year.csv')

# Select features and target
features = df[['Open', 'High', 'Low', 'Volume']]  # Example features
target = df['Close'].values.reshape(-1, 1)  # Target variable

# Scale the features and target
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target)

# Define the Encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Initialize the model, loss function, and optimizer
input_dim = features_scaled.shape[1]  # Number of features
hidden_dim = 128
encoded_dim = 128

model = Encoder(input_dim, hidden_dim, encoded_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert features_scaled to tensor
input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# Train the model
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    output, encoded = model(input_tensor)
    loss = criterion(output, input_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Extract the thought_vector from the trained model
thought_vector = encoded.detach().numpy()
print("Thought vector shape:", thought_vector.shape)

# Scale thought_vector to ensure consistent scaling
thought_vector_scaler = MinMaxScaler()
thought_vector_scaled = thought_vector_scaler.fit_transform(thought_vector)

# Prepare target variable for fitting (shifted to align with input)
target_scaled_input = target_scaled[1:].ravel()  # Flatten to 1D
thought_vector_scaled_input = thought_vector_scaled[:-1]  # All but the last

# Fit ensemble models using the scaled features
ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
bagging = BaggingRegressor(n_estimators=500)
et = ExtraTreesRegressor(n_estimators=500)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=500)

# Fit the models
ada.fit(thought_vector_scaled_input, target_scaled_input)
bagging.fit(thought_vector_scaled_input, target_scaled_input)
et.fit(thought_vector_scaled_input, target_scaled_input)
gb.fit(thought_vector_scaled_input, target_scaled_input)
rf.fit(thought_vector_scaled_input, target_scaled_input)

# Evaluate the models and make predictions
predictions_ada = ada.predict(thought_vector_scaled_input)
predictions_bagging = bagging.predict(thought_vector_scaled_input)
predictions_et = et.predict(thought_vector_scaled_input)
predictions_gb = gb.predict(thought_vector_scaled_input)
predictions_rf = rf.predict(thought_vector_scaled_input)

# Plotting predictions against actual values
plt.figure(figsize=(14, 7))
plt.plot(target_scaled_input, label='Actual Prices', color='black')
plt.plot(predictions_ada, label='AdaBoost Predictions', color='blue', alpha=0.6)
plt.plot(predictions_rf, label='Random Forest Predictions', color='red', alpha=0.6)
plt.legend()
plt.title('Model Predictions vs Actual Prices')
plt.show()

# Length check
print(f"Length of true values: {len(target_scaled_input)}")
print(f"Length of AdaBoost predictions: {len(predictions_ada)}")
print(f"Length of Bagging predictions: {len(predictions_bagging)}")
print(f"Length of Extra Trees predictions: {len(predictions_et)}")
print(f"Length of Gradient Boosting predictions: {len(predictions_gb)}")
print(f"Length of Random Forest predictions: {len(predictions_rf)}")

# Align lengths
n = min(len(target_scaled_input), len(predictions_ada))  # Use the smallest length

# Define the evaluation function
def evaluate_predictions(true, preds):
    mae = np.mean(np.abs(true - preds))
    mse = np.mean((true - preds) ** 2)
    r2 = 1 - (np.sum((true - preds) ** 2) / np.sum((true - np.mean(true)) ** 2))
    return mae, mse, r2

# Evaluate AdaBoost predictions
ada_mae, ada_mse, ada_r2 = evaluate_predictions(target_scaled_input[:n], np.array(predictions_ada[:n]))
print(f"AdaBoost - MAE: {ada_mae}, MSE: {ada_mse}, R²: {ada_r2}")

# Evaluate Bagging predictions
bagging_mae, bagging_mse, bagging_r2 = evaluate_predictions(target_scaled_input[:n], np.array(predictions_bagging[:n]))
print(f"Bagging - MAE: {bagging_mae}, MSE: {bagging_mse}, R²: {bagging_r2}")

# Evaluate Extra Trees predictions
et_mae, et_mse, et_r2 = evaluate_predictions(target_scaled_input[:n], np.array(predictions_et[:n]))
print(f"Extra Trees - MAE: {et_mae}, MSE: {et_mse}, R²: {et_r2}")

# Evaluate Gradient Boosting predictions
gb_mae, gb_mse, gb_r2 = evaluate_predictions(target_scaled_input[:n], np.array(predictions_gb[:n]))
print(f"Gradient Boosting - MAE: {gb_mae}, MSE: {gb_mse}, R²: {gb_r2}")

# Evaluate Random Forest predictions
rf_mae, rf_mse, rf_r2 = evaluate_predictions(target_scaled_input[:n], np.array(predictions_rf[:n]))
print(f"Random Forest - MAE: {rf_mae}, MSE: {rf_mse}, R²: {rf_r2}")



# Function to predict the next 30 days
def predict_next_days(model, last_data, days=30):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    input_tensor = torch.tensor(last_data, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(days):
            output, encoded = model(input_tensor)
            predicted_value = output[-1, 0].numpy().reshape(1, 1)  # Ensure shape is (1, 1)
            predictions.append(predicted_value)

            # Create a new feature vector for the next day prediction
            new_feature_vector = np.concatenate((input_tensor[0, 1:].numpy(), predicted_value.flatten()))  # Keep all but the first feature

            # Update input tensor with the new feature vector (to predict the next day)
            input_tensor = torch.tensor(new_feature_vector, dtype=torch.float32).reshape(1, -1)  # Reshape to maintain dimensions

    return np.array(predictions).reshape(-1, 1)  # Ensure the final output is 2D

# Load the last data point from your dataset
last_data = features_scaled[-1].reshape(1, -1)  # Last feature row

# Generate predictions for the next 30 days
predictions = predict_next_days(model, last_data, days=30)

# Rescale the predictions back to original scale
predictions_rescaled = scaler.inverse_transform(predictions)

# Prepare dates for the next 30 days
last_date = pd.to_datetime(df['Date'].iloc[-1])
next_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

# Create a DataFrame for predictions
predictions_df = pd.DataFrame(data=predictions_rescaled, columns=['Predicted Close'], index=next_dates)

# Make sure that the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plot the predictions
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Actual Prices', color='black')
plt.plot(predictions_df.index, predictions_df['Predicted Close'], label='Predicted Prices', color='blue', linestyle='--')
plt.title('Predicted Prices for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
