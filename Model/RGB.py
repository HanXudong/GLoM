from keras.layers import Input, Dense
from keras.models import Model
import keras
import keras.backend as K

def cos_distance(y_true, y_pred):
    return 1. - K.batch_dot(K.l2_normalize(y_true, axis=-1), K.l2_normalize(y_pred, axis=-1), axes=1)

def euclidean_distance_loss(y_true, y_pred):
    
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def RGB():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = keras.layers.concatenate([input_Rmu, input_WordVector])
	x_fc1 = Dense(30, activation='sigmoid')(x_fc1)
	x_fc1 = Dense(10, activation='sigmoid')(x_fc1)

	x_m1 = Dense(3, activation='sigmoid', name='m_1')(x_fc1)
	x_m1_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m1])

	x_m2 = Dense(3, activation='sigmoid', name='m_2')(x_fc1)
	x_m2_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m2])

	x_m3 = Dense(3, activation='sigmoid', name='m_3')(x_fc1)
	x_m3_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m3])

	beta = Dense(3, activation='sigmoid', name='beta')(x_fc1)

	x_m_dot_R = keras.layers.concatenate([x_m1_dot_R, x_m2_dot_R, x_m3_dot_R])

	out_put_T = keras.layers.Add()([beta, x_m_dot_R])

	out_put_m = keras.layers.Subtract()([out_put_T, input_Rmu])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=[out_put_T, out_put_m])

	model.compile(optimizer='adam', loss=["mse", cos_distance])

	return model