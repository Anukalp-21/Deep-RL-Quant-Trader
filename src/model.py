import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Dropout,GaussianNoise,Activation,LSTM,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def LSTM_Model(input_dim,n_action):
    i=Input(shape=(input_dim))
    x=GaussianNoise(0.1)(i)
    x=LSTM(64)(x)
    x=Dropout(0.3)(x)
    x=Dense(64,kernel_regularizer=l2(0.01))(x)
    x=Activation('relu')(x)
    x=Dropout(0.3)(x)
    x=Dense(n_action)(x)
    model=Model(i,x)
    optimizer=Adam(learning_rate=7e-5,clipnorm=1.0)
    model.compile(loss='huber_loss',optimizer=optimizer)
    return model