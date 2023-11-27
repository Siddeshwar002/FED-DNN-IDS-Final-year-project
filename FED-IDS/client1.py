import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint
from tensorflow.keras.regularizers import L1, L2
import sys
import flwr as fl


df = pd.read_csv('cic-collection.csv')
df = df.drop(columns='Label')

df.ClassLabel.value_counts()


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
model.add(LSTM(59, input_shape=(timesteps, num_features),
          return_sequences=True, kernel_regularizer=L2(0.001)))
model.add(Dropout(0.6))
model.add(LSTM(59, kernel_regularizer=L2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Compile the model using binary cross-entropy loss and Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model on the training set and validate it on the testing set
num_epochs = 10
batch_size = 32
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

# Define the ModelCheckpoint callback function


checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(X_train, y_train, epochs=1,
                      validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping,   checkpoint])
        hist = r.history
        print("Fit history : ", hist)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(X_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:"+str(sys.argv[1]),
    client=FlowerClient(),
    grpc_max_message_length=1024*1024*1024
)
