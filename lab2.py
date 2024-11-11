import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


data = pd.read_csv('C:\yu\outube.csv')
    

numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())




X = data.drop(['SalePrice'], axis=1)  
y = data['SalePrice']


X = pd.get_dummies(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  
])




model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)


y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R^2 score: {r2:.4f}")


if r2 >= 0.75:
    print(f"Модель досягла необхідної точності R^2: {r2:.4f}")
else:
    print(f"Модель не досягла необхідної точності R^2: {r2:.4f}")

