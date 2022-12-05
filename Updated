# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Set up the Coinbase API client
client = Client(api_key="your_api_key", api_secret="your_api_secret")

# Get the market data for Chainlink from Coinbase
data = client.get_historic_prices("LINK-USD")

# Convert the data to a Pandas DataFrame
data = pd.DataFrame(data)

# Clean and preprocess the data
data = data.dropna()
data["time"] = pd.to_datetime(data["time"])

# Split the data into training and testing sets
X_train = data[["time", "volume", "market_cap"]][:int(len(data) * 0.8)]
y_train = data[["price"]][:int(len(data) * 0.8)]
X_test = data[["time", "volume", "market_cap"]][int(len(data) * 0.8):]
y_test = data[["price"]][int(len(data) * 0.8):]

# Create a linear regression model
lin_reg = LinearRegression()

# Train the model on the training data
lin_reg.fit(X_train, y_train)

# Evaluate the model on the testing data
lin_reg_score = lin_reg.score(X_test, y_test)

# Create a random forest regression model
rf_reg = RandomForestRegressor()

# Use grid search to find the best hyperparameters for the model
param_grid = {"n_estimators": [10, 100, 1000], "max_depth": [2, 5, 10]}
grid_search = GridSearchCV(rf_reg, param_grid, cv=5)

# Train the model on the training data
grid_search.fit(X_train, y_train)

# Evaluate the model on the testing data
rf_reg_score = grid_search.score(X_test, y_test)

# Choose the model with the best performance on the testing data
if lin_reg_score > rf_reg_score:
    model = lin_reg
else:
    model = rf_reg

# Use the chosen model to make predictions about future prices
future_prices = model.predict(X_test)

# Output the predicted prices
print(future_prices)

# Use the predicted prices to recommend a risk allocation strategy
if future_prices[-1] > future_prices[0]:
    # If the predicted price is expected to increase, recommend a higher allocation to Chainlink
    allocation = 0.8
else:
    # If the predicted price is expected to decrease, recommend a lower allocation to Chainlink
    allocation = 0.2

# Output the recommended allocation
print(f"The recommended allocation to Chainlink is {allocation * 100}%")

# Monitor the portfolio and adjust the allocation as needed
while True:
    # Get the latest market
    data for Chainlink
    from Coinbase
new_data = client.get_historic_prices("LINK-USD")

# Convert the data to a Pandas DataFrame
new_data = pd.DataFrame(new_data)

# Clean and preprocess the data
new_data = new_data.dropna()
new_data["time"] = pd.to_datetime(new_data["time"])

# Use the chosen model to make predictions about future prices
new_future_prices = model.predict(new_data)

# Compare the predicted prices with the previous predictions
if new_future_prices[-1] > future_prices[-1]:
    # If the predicted price is expected to increase, recommend a higher allocation to Chainlink
    allocation = 0.8
else:
    # If the predicted price is expected to decrease, recommend a lower allocation to Chainlink
    allocation = 0.2

# Output the updated allocation
print(f"The updated allocation to Chainlink is {allocation * 100}%")

# Set the new future prices as the previous prices for the next iteration
future_prices = new_future_prices

# Wait for a specified time interval before checking the market again
time.sleep(60)"


