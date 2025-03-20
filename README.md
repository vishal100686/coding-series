from sklearn.linear_model import LinearRegression
import numpy as np

# Example Data: [Pizza Size], [Price]
X = np.array([8, 12, 16, 20]).reshape(-1, 1)
y = np.array([8, 12, 16, 20])

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict for a 14-inch pizza
print(model.predict([[14]]))  # Output: 14.0

