import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

sns.set()

# Feature engineering functions
def add_moving_average(df, window):
    df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def add_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    df['MACD'] = df['Close'].ewm(span=fast).mean() - df['Close'].ewm(span=slow).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=signal).mean()
    return df

def add_bollinger_bands(df, window=20, num_std=2):
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Upper'] = df['BB_Middle'] + num_std * df['Close'].rolling(window=window).std()
    df['BB_Lower'] = df['BB_Middle'] - num_std * df['Close'].rolling(window=window).std()
    return df

# Load the dataset with error handling
try:
    df = pd.read_csv('sp500_daily_data_last_4_years.csv')
except FileNotFoundError:
    print("Error: The specified file was not found. Loading default dataset.")
    df = pd.read_csv('default_dataset.csv')  # Replace with your default dataset path
except pd.errors.EmptyDataError:
    print("Error: The file is empty. Please check the file.")
    exit()  # Exit if the data is empty
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()  # Exit for any other unforeseen errors

df = add_moving_average(df, 50)
df = add_moving_average(df, 200)
df = add_rsi(df)
df = add_macd(df)
df = add_bollinger_bands(df)

# Drop rows with NaN values
df.dropna(inplace=True)

# Select features and target
features = df[['Open', 'High', 'Low', 'Volume', 'MA_50', 'MA_200', 'RSI', 'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower']]
target = df['Close'].values.reshape(-1, 1)

# Scale the features and target
scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)
scaler_target = MinMaxScaler()
target_scaled = scaler_target.fit_transform(target)  # Save the scaler for inverse transformation

# Ridge regression for feature selection
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_importance = pd.Series(abs(ridge.coef_[0]), index=features.columns).sort_values(ascending=False)
print("\nTop 10 features selected by Ridge:")
print(ridge_importance.head(10))

# PCA for further dimensionality reduction
pca = PCA(n_components=4)  # Adjusted to retain 4 components
pca.fit(X_train)
features_pca = pca.transform(features_scaled)
print(f"\nNumber of components selected by PCA: {pca.n_components_}")

# Define the Encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim=4):  # Change default to 4
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim)  # Ensure output is 4
        )

    def forward(self, x):
        return self.encoder(x)

# Initialize the model, loss function, and optimizer
input_dim = features_pca.shape[1]  # Number of PCA features
hidden_dim = 128
encoded_dim = 4  # Output dimension of the encoder

model = Encoder(input_dim, hidden_dim, encoded_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert features_pca to tensor
input_tensor = torch.tensor(features_pca, dtype=torch.float32)

# Train the model
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    encoded = model(input_tensor)
    loss = criterion(encoded, input_tensor)  # Ensure the target matches the output dimension
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
print("Fitting ensemble models...")
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

# Evaluate predictions for each model
for model_name, predictions in zip(['AdaBoost', 'Bagging', 'Extra Trees', 'Gradient Boosting', 'Random Forest'],
                                    [predictions_ada, predictions_bagging, predictions_et, predictions_gb, predictions_rf]):
    mae, mse, r2 = evaluate_predictions(target_scaled_input[:n], np.array(predictions[:n]))
    print(f"{model_name} - MAE: {mae}, MSE: {mse}, R²: {r2}")

def predict_next_days(model, last_data, actual_values, days=30):
    input_data = last_data.copy()
    predictions = []

    for day in range(days):
        predicted_value = model.predict(input_data[-1].reshape(1, -1))[0]
        
        # Add noise to predictions
        predicted_value += np.random.normal(0, 0.01)  # Adjust the standard deviation as necessary

        # Append the predicted value to the list
        predictions.append(predicted_value)

        # Create a new input for the next prediction by combining recent actual values
        new_input = np.concatenate((input_data[-1][:-1], [predicted_value]))  # Include the predicted value
        if day < len(actual_values):  # Ensure there's an actual value available
            new_input = np.concatenate((new_input[:-1], [actual_values[day]]))  # Replace the last feature with the actual

        input_data = np.vstack((input_data, new_input))  # Append the new input

    return np.array(predictions).reshape(days, 1)

# Use the last available features for predictions
last_features = thought_vector_scaled[-1].reshape(1, -1)

# Make predictions for the next 30 days using the last features from the thought vector
predicted_values = predict_next_days(ada, last_features, target_scaled_input[-30:], days=30)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predicted_values, columns=['Predicted Close'])
predictions_df.index = pd.date_range(start=df['Date'].iloc[-30], periods=30, freq='D')

# Inverse scale the predictions
predictions_ada_inverse = scaler_target.inverse_transform(predictions_ada.reshape(-1, 1)).flatten()
predictions_bagging_inverse = scaler_target.inverse_transform(predictions_bagging.reshape(-1, 1)).flatten()
predictions_et_inverse = scaler_target.inverse_transform(predictions_et.reshape(-1, 1)).flatten()
predictions_gb_inverse = scaler_target.inverse_transform(predictions_gb.reshape(-1, 1)).flatten()
predictions_rf_inverse = scaler_target.inverse_transform(predictions_rf.reshape(-1, 1)).flatten()

# Inverse scale the predicted values for future days
predicted_values_inverse = scaler_target.inverse_transform(predicted_values)

# Create a DataFrame for the inverse scaled predictions
predictions_df = pd.DataFrame(predicted_values_inverse, columns=['Predicted Close'])
predictions_df.index = pd.date_range(start=df['Date'].iloc[-30], periods=30, freq='D')

# Plotting predictions against actual values
plt.figure(figsize=(14, 7))
plt.plot(scaler_target.inverse_transform(target_scaled_input[:n].reshape(-1, 1)), label='Actual Prices', color='black')
plt.plot(predictions_ada_inverse, label='AdaBoost Predictions', color='blue', alpha=0.6)
plt.plot(predictions_rf_inverse, label='Random Forest Predictions', color='red', alpha=0.6)
plt.legend()
plt.title('Model Predictions vs Actual Prices')
plt.show()

print(predictions_df)

# Assuming you have the previous code already executed up to the predictions_df creation.

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predicted_values_inverse, columns=['Predicted Close'])
predictions_df.index = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plotting predictions against actual values
plt.figure(figsize=(14, 7))
# Plot actual prices for the last available dates
plt.plot(df['Date'].iloc[-30:], scaler_target.inverse_transform(target_scaled_input[-30:].reshape(-1, 1)), label='Actual Prices', color='black')
# Plot predicted values for the next 30 days
plt.plot(predictions_df.index, predictions_df['Predicted Close'], label='Predicted Prices', color='orange', linestyle='--')
plt.legend()
plt.title('Predicted Prices for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Generate predictions for each model using the predict_next_days function
predictions_ada = predict_next_days(ada, last_features, target_scaled_input[-30:], days=30)
predictions_bagging = predict_next_days(bagging, last_features, target_scaled_input[-30:], days=30)
predictions_et = predict_next_days(et, last_features, target_scaled_input[-30:], days=30)
predictions_gb = predict_next_days(gb, last_features, target_scaled_input[-30:], days=30)
predictions_rf = predict_next_days(rf, last_features, target_scaled_input[-30:], days=30)

# Inverse scale all predictions
predictions_ada_inverse = scaler_target.inverse_transform(predictions_ada)
predictions_bagging_inverse = scaler_target.inverse_transform(predictions_bagging)
predictions_et_inverse = scaler_target.inverse_transform(predictions_et)
predictions_gb_inverse = scaler_target.inverse_transform(predictions_gb)
predictions_rf_inverse = scaler_target.inverse_transform(predictions_rf)

# Assuming the previous code has been executed and you have predictions from all models.

# Create a DataFrame to hold the predictions
adjusted_predictions_df = pd.DataFrame({
    'Date': pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D'),
    'AdaBoost': predictions_ada_inverse.flatten(),
    'Bagging': predictions_bagging_inverse.flatten(),
    'Extra Trees': predictions_et_inverse.flatten(),
    'Gradient Boosting': predictions_gb_inverse.flatten(),
    'Random Forest': predictions_rf_inverse.flatten()
})

# Plotting predictions for each model
plt.figure(figsize=(14, 7))

# Plot actual prices for the last available dates
plt.plot(df['Date'].iloc[-30:], scaler_target.inverse_transform(target_scaled_input[-30:].reshape(-1, 1)),
         label='Actual Prices', color='black')

# Plot adjusted predicted values for the next 30 days for each model
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['AdaBoost'], label='AdaBoost', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Bagging'], label='Bagging', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Extra Trees'], label='Extra Trees', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Gradient Boosting'], label='Gradient Boosting', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Random Forest'], label='Random Forest', linestyle='--')

plt.legend()
plt.title('Predicted Prices for the Next 30 Days by Different Models')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def apply_sentiment_modifier(predictions, sentiment_rating, max_adjustment=0.05):
    """
    Apply a sentiment modifier to the predictions.

    :param predictions: List or array of predicted prices
    :param sentiment_rating: Rating from 0 (bearish) to 10 (bullish)
    :param max_adjustment: Maximum percentage adjustment (default 5%)
    :return: Modified predictions
    """
    sentiment_scale = (sentiment_rating - 5) / 5  # Scale the rating to [-1, 1]
    adjustment_factor = sentiment_scale * max_adjustment  # Calculate adjustment
    modified_predictions = predictions * (1 + adjustment_factor)  # Adjust predictions
    return modified_predictions

# Add this line to set the sentiment rating
sentiment_rating = 7  # Example rating (slightly bullish)

# Modify predictions using the sentiment modifier
adjusted_predictions_ada = apply_sentiment_modifier(predictions_ada_inverse, sentiment_rating)
adjusted_predictions_bagging = apply_sentiment_modifier(predictions_bagging_inverse, sentiment_rating)
adjusted_predictions_et = apply_sentiment_modifier(predictions_et_inverse, sentiment_rating)
adjusted_predictions_gb = apply_sentiment_modifier(predictions_gb_inverse, sentiment_rating)
adjusted_predictions_rf = apply_sentiment_modifier(predictions_rf_inverse, sentiment_rating)

# Create a DataFrame for the adjusted predictions
adjusted_predictions_df = pd.DataFrame({
    'Date': pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D'),
    'AdaBoost': adjusted_predictions_ada.flatten(),
    'Bagging': adjusted_predictions_bagging.flatten(),
    'Extra Trees': adjusted_predictions_et.flatten(),
    'Gradient Boosting': adjusted_predictions_gb.flatten(),
    'Random Forest': adjusted_predictions_rf.flatten()
})

# Plotting predictions for each model with sentiment adjustment
plt.figure(figsize=(14, 7))

# Plot actual prices for the last available dates
plt.plot(df['Date'].iloc[-30:], scaler_target.inverse_transform(target_scaled_input[-30:].reshape(-1, 1)),
         label='Actual Prices', color='black')

# Plot adjusted predicted values for the next 30 days for each model
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['AdaBoost'], label='AdaBoost', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Bagging'], label='Bagging', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Extra Trees'], label='Extra Trees', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Gradient Boosting'], label='Gradient Boosting', linestyle='--')
plt.plot(adjusted_predictions_df['Date'], adjusted_predictions_df['Random Forest'], label='Random Forest', linestyle='--')

plt.legend()
plt.title('Sentiment-Adjusted Predicted Prices for the Next 30 Days by Different Models')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

