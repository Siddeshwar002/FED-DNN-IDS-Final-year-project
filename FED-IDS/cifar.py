import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Load the dataset
# df = pd.read_csv("02-16-2018.csv")

df = pd.read_csv('cic-collection.csv')
df = df.drop(columns='Label')

df.ClassLabel.value_counts()
# Preprocess the dataset
# df = df.dropna()
# df = df.drop_duplicates()
# df = df.drop(['Dst Port', 'Protocol', 'Timestamp'], axis=1)
# features = ['Flow Duration', 'Tot Fwd Pkts',
#             'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
#             'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
#             'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
#             'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
#             'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
#             'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
#             'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
#             'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
#             'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
#             'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
#             'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
#             'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
#             'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
#             'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
#             'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
#             'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
#             'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
#             'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
#             'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
#             'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

# df[features] = df[features].apply(pd.to_numeric, errors='coerce', axis=1)
# df = df.fillna(0)
# df["Label"] = df["Label"].apply(
#     lambda x: "BENIGN" if x == "Benign" else "MALICIOUS")
# le = LabelEncoder()
# df["Label"] = le.fit_transform(df["Label"])


def binarizer(df):
    df.loc[df['ClassLabel'] != 'Benign', 'ClassLabel'] = 1
    df.loc[df['ClassLabel'] == 'Benign', 'ClassLabel'] = 0
    # print(df['Label'].value_counts())
    df['ClassLabel'] = df['ClassLabel'].astype(dtype=np.int32)
    return df


df = binarizer(df)
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
# df = df.sample(frac=1).reset_index(drop=True)


# Split the dataset into training and testing sets except label column
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[:-1]], df["ClassLabel"], test_size=0.2, random_state=42)

# # Normalize the data to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Reshape the data to fit the LSTM input shape
timesteps = 1
num_features = X_train.shape[1]
X_train = X_train.reshape(-1, timesteps, num_features)
X_test = X_test.reshape(-1, timesteps, num_features)

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, num_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# Compile the model using binary cross-entropy loss and Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model on the training set and validate it on the testing set
num_epochs = 10
batch_size = 32
model.fit(X_train, y_train, epochs=num_epochs,
          batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
