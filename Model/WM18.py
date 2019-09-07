from keras.layers import Input, Dense
from keras.models import Model
import keras
import keras.backend as K

def cos_distance(y_true, y_pred):
    return 1. - K.batch_dot(K.l2_normalize(y_true, axis=-1), K.l2_normalize(y_pred, axis=-1), axes=1)

def euclidean_distance_loss(y_true, y_pred):
    
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def WM18():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = keras.layers.concatenate([input_Rmu, input_WordVector])
	x_fc1 = Dense(30, activation='sigmoid')(x_fc1)

	x_fc1 = keras.layers.concatenate([x_fc1, input_Rmu])

	out_put_T = Dense(3, activation='sigmoid')(x_fc1)

	out_put_m = keras.layers.Subtract()([out_put_T, input_Rmu])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=[out_put_T, out_put_m])

	model.compile(optimizer = keras.optimizers.Adadelta(), loss=[euclidean_distance_loss, cos_distance])

	return model