import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load and process data
df = pd.read_csv("data/data_logistic_regression.csv")

# Independent, dependent features
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Creating the Training Set and the Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building and training the model
model = LogisticRegression(random_state = 42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(sc, "model/scaler.pkl")
