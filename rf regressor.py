# !pip install scikit-learn pandas matplotlib graphviz yfinance

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from graphviz import Source
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Bitcoin data (replace with your actual data source)
btc = yf.download("BTC-USD", start="2022-01-01", end="2023-01-01")

btc["Tomorrow"] = btc["Close"].shift(-1)
btc = btc.dropna()

X = btc.drop(columns=["Adj Close", "Tomorrow"])
y = btc["Tomorrow"]

features = X.columns

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=0)

# Create a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust parameters as needed

# Train the model
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Visualize individual trees
for idx, tree in enumerate(model.estimators_):  # Visualize the first 5 trees
  dot_data = export_graphviz(tree,
                            feature_names=features,
                            filled=True,
                            rounded=True,
                            special_characters=True)
  graph = Source(dot_data)
  graph.render(f"tree_{idx}", format="png")  # Save each tree as a PNG image

# Display the first tree
from IPython.display import Image
Image(filename='tree_0.png')