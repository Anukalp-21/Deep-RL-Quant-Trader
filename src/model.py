import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Dropout,BatchNormalization,Activation,LSTM,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
def LSTM_Model(input_dim, n_action):
    i = Input(shape=(input_dim))
    x = LSTM(32)(i)
    x = Dropout(0.7)(x)
    x = Dense(16)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(n_action)(x)
    model = Model(i, x)
    optimizer = Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(loss='huber_loss', optimizer=optimizer)
    return model