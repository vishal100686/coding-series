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

from sklearn.linear_model import LogisticRegression

# Example Data
X = [[0], [1], [2], [3]]  # Number of spam words
y = [0, 0, 1, 1]  # Not spam (0), Spam (1)

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Predict for an email with 2 spam words
print(model.predict([[2]]))  # Output: 1 (Spam)
