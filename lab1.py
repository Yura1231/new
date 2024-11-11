import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf



data = pd.read_csv('C:\yu\youtube.csv')


X = data['CONTENT']  
y = data['CLASS']    
yuasdasd= 123

sad=1
data.head()

vectorizer = TfidfVectorizer(max_features=5000)  
X_tfidf = vectorizer.fit_transform(X).toarray()


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cross_val_scores = []


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.5), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = create_model()  
    
    
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
    
   
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    cross_val_scores.append(val_acc)


mean_cross_val_score = np.mean(cross_val_scores)
print(f"Average Cross-Validation Accuracy: {mean_cross_val_score:.4f}")


model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

test_accuracy = accuracy_score(y_test, y_pred)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Accuracy using predict: {test_accuracy:.4f}")



