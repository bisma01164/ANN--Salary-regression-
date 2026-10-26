import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_csv("Churn_Modelling.csv")

# Preprocessing same as scaler
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

ohe = OneHotEncoder(sparse_output=False)
geo_encoded = ohe.fit_transform(df[['Geography']])
geo_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

X = pd.concat([
    df[['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember']],
    geo_df
], axis=1)

y = df['EstimatedSalary']  # Salary predict karna hai

print(f"X shape: {X.shape}") # (10000, 11) aayega

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ANN Model jo 11 input lega
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse'
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Model ko save karo - dono naam se taake app.py ko mil jaye
model.save("model.h5")
model.save("regression_model.h5")
model.save("ann_model.h5")

print("Model ban gaya with 11 inputs!")