from keras.layers import Input, Dense, Lambda
from keras.models import Model
import keras

def my_product(tensors):
    out1 = tensors[0] * tensors[1]
    return out1

def my_product_output_shape(input_shapes):
    shape1 = list(input_shapes[0])
    return tuple(shape1)

def DC_RGB():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = Dense(32, activation='sigmoid', name='fc1')(input_WordVector)
	x_fc1 = Dense(16, activation='sigmoid', name='fc2')(x_fc1)

	m = Dense(3, activation='sigmoid', name='m')(x_fc1)
	alpha = Dense(1, activation='sigmoid', name='alpha')(x_fc1)

	m = Lambda(my_product, my_product_output_shape)([m, alpha])

	alpha = Lambda(lambda x: 1-x)(alpha)
	r = Lambda(my_product, my_product_output_shape)([input_Rmu, alpha])

	out_put_T = keras.layers.Add()([m, r])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=out_put_T)

	model.compile(optimizer='adam', loss='mse')

	return model

def DC_RGB_v2():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = Dense(32, activation='sigmoid')(input_WordVector)
	x_fc1 = Dense(16, activation='sigmoid')(x_fc1)

	m = Dense(3, activation='sigmoid', name='m')(x_fc1)

	x_fc2 = keras.layers.concatenate([input_Rmu, m])
	# x_fc2 = Dense(32, activation='sigmoid')(x_fc2)
	# x_fc2 = Dense(10, activation='sigmoid')(x_fc2)
	alpha = Dense(1, activation='sigmoid', name='alpha')(x_fc2)

	m = Lambda(my_product, my_product_output_shape)([m, alpha])

	alpha = Lambda(lambda x: 1-x)(alpha)
	r = Lambda(my_product, my_product_output_shape)([input_Rmu, alpha])

	out_put_T = keras.layers.Add()([m, r])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=out_put_T)

	model.compile(optimizer='adam', loss='mse')

	return model

def general_RGB():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = Dense(32, activation='sigmoid')(input_WordVector)
	x_fc1 = Dense(16, activation='sigmoid')(x_fc1)

	x_m1 = Dense(3, activation='sigmoid')(x_fc1)
	x_m1_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m1])

	x_m2 = Dense(3, activation='sigmoid')(x_fc1)
	x_m2_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m2])

	x_m3 = Dense(3, activation='sigmoid')(x_fc1)
	x_m3_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m3])

	beta = Dense(32, activation='sigmoid')(input_WordVector)
	beta = Dense(3, activation='sigmoid')(x_fc1)

	x_m_dot_R = keras.layers.concatenate([x_m1_dot_R, x_m2_dot_R, x_m3_dot_R])

	out_put_T = keras.layers.Add()([beta, x_m_dot_R])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=out_put_T)

	model.compile(optimizer='adam', loss='mse')
	
	return model
