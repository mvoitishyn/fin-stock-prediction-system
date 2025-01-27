import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/processed/stockdata/AAPL.csv")
data["Return"] = data["Close"].pct_change()
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()

data.dropna(inplace=True)

data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

X = data[["Return", "MA5", "MA10"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Prediction Accuracy: {accuracy:.2f}')
