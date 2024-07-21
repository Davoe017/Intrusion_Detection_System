#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("scikit-learn version:", sklearn.__version__)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)


# In[19]:


df = pd.read_csv('C:\\Users\\David\\Desktop\\University Studies\\400 level omega semester\\CIS421\\IDS_Data1.csv')

# Inspect the DataFrame
print(df.head())  # Display the first few rows
print(df.shape)   # Display the dimensions of the DataFrame
print(df.info())


# In[20]:


num_col = ['Sport', 'TotPkts', 'TotBytes', 'SrcPkts', 'DstPkts', 'SrcBytes',
                   'count', 'srv_count', 'dst_host_count', 'dst_host_same_srv_rate',
                   'dst_host_diff_srv_rate', 'dst_bytes']

# Initialize StandardScaler, LabelEncoder
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Normalize numeric columns
df[num_col] = scaler.fit_transform(df[num_col])

categ_col = ['service', 'flag', 'protocol_type', 'class']

# Encode each categorical columns
for col in categ_col:
    print(f"Mapping for '{col}':")
    df[col] = label_encoder.fit_transform(df[col])
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {idx}: {label}")
print(df.head())
df.shape


# In[21]:


print(df.head(20))


# In[22]:


from sklearn.ensemble import RandomForestClassifier

# Split your data into features (X) and target (y)
X = df.drop(columns=['class'])
y = df['class']

# Initialize the model
clf = RandomForestClassifier(random_state=42)

# Fit the model
clf.fit(X, y)

# Get feature importances
feature_importances = clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(feature_importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature '{X.columns[indices[f]]}' - Importance: {feature_importances[indices[f]]}")


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

features = ['dst_bytes', 'flag', 'dst_host_same_srv_rate', 'count', 'dst_host_diff_srv_rate',
            'service', 'dst_host_count', 'protocol_type', 'srv_count']
target = 'class'

X = df[features]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(len(features),)),
    Dropout(0.2),  # Dropout layer to prevent overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),  # Dropout layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  # Binary classification
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)#To prevent overfitting

history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')


# In[27]:


model.save('C:\\Users\\David\\Desktop\\University Studies\\400 level omega semester\\CIS421\\IDSmodel.h5')


# In[28]:


predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Example: Evaluate the model
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




