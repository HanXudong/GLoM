from keras.layers import Input, Dense
from keras.models import Model
import keras
from keras import backend as K
import numpy as np

def cos_H(y_true, y_pred):
    return 1. - K.cos((y_pred - y_true)*2*np.pi)

def HSV():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = keras.layers.concatenate([input_Rmu, input_WordVector])
	x_fc1 = Dense(32, activation='sigmoid')(x_fc1)
	x_fc1 = Dense(16, activation='sigmoid')(x_fc1)

	x_m1 = Dense(3, activation='sigmoid')(x_fc1)
	x_m1_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m1])

	x_m2 = Dense(3, activation='sigmoid')(x_fc1)
	x_m2_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m2])

	x_m3 = Dense(3, activation='sigmoid')(x_fc1)
	x_m3_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m3])


	x_m_dot_R = keras.layers.concatenate([x_m2_dot_R, x_m3_dot_R])
	H =  Dense(1, activation='sigmoid')(x_m1_dot_R)

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=[H, x_m_dot_R])

	model.compile(optimizer='adam', loss=[cos_H, 'mse'])

	return model

