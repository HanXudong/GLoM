from keras.layers import Input, Dense, Lambda
from keras.models import Model
import keras
import keras.backend as K

def cos_distance(y_true, y_pred):
    return 1. - K.batch_dot(K.l2_normalize(y_true, axis=-1), K.l2_normalize(y_pred, axis=-1), axes=1)

def my_product(tensors):
    out1 = tensors[0] * tensors[1]
    return out1

def my_product_output_shape(input_shapes):
    shape1 = list(input_shapes[0])
    return tuple(shape1)

def DC_RGB():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = Dense(32, activation='relu')(input_WordVector)
	x_fc1 = Dense(16, activation='relu')(x_fc1)

	m = Dense(3, activation='relu', name="m_layer")(x_fc1)
	alpha = Dense(1, activation='relu', name="alpha_layer")(x_fc1)

	m = Lambda(my_product, my_product_output_shape)([m, alpha])

	alpha = Lambda(lambda x: 1-x)(alpha)
	r = Lambda(my_product, my_product_output_shape)([input_Rmu, alpha])

	out_put_T = keras.layers.Add()([m, r])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=out_put_T)

	model.compile(optimizer='adam', loss='mse')

	return model

def ConvP_RGB():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_ConvP = Input(shape=(3,), dtype='float32', name='input_ConvP')
    
	x_fc1 = Dense(32, activation='relu')(input_ConvP)
	x_fc1 = Dense(10, activation='relu')(x_fc1)
    
	x_m1 = Dense(3, activation='relu')(x_fc1)
	x_m1_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m1])

	x_m2 = Dense(3, activation='relu')(x_fc1)
	x_m2_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m2])

	x_m3 = Dense(3, activation='relu')(x_fc1)
	x_m3_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m3])
    
	beta = Dense(3, activation='relu')(x_fc1)

	x_m_dot_R = keras.layers.concatenate([x_m1_dot_R, x_m2_dot_R, x_m3_dot_R])

	out_put_T = keras.layers.Add()([beta, x_m_dot_R])

	out_put_m = keras.layers.Subtract()([out_put_T, input_Rmu])

	model = Model(inputs=[input_Rmu, input_ConvP], outputs=[out_put_T, out_put_m])

	model.compile(optimizer='adam', loss=["mse", cos_distance])
	
	return model
    
def ConvP_RGBv2():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')
    
	x_fc1 = Dense(30, activation='relu')(input_WordVector)
	x_fc1 = Dense(10, activation='relu')(x_fc1)
       
	x_m1 = Dense(3, activation='relu')(x_fc1)
	x_m1_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m1])

	x_m2 = Dense(3, activation='relu')(x_fc1)
	x_m2_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m2])

	x_m3 = Dense(3, activation='relu')(x_fc1)
	x_m3_dot_R = keras.layers.Dot(axes=1)([input_Rmu, x_m3])
    
	beta = Dense(3, activation='relu')(x_fc1)

	x_m_dot_R = keras.layers.concatenate([x_m1_dot_R, x_m2_dot_R, x_m3_dot_R])

	out_put_T = keras.layers.Add()([beta, x_m_dot_R])

	out_put_m = keras.layers.Subtract()([out_put_T, input_Rmu])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=[out_put_T, out_put_m])

	model.compile(optimizer='adam', loss=["mse", cos_distance])
	
	return model

def DC_RGB_v2():
	input_Rmu = Input(shape=(3,), dtype='float32', name='input_Rmu')
	input_WordVector = Input(shape=(300,), dtype='float32', name='input_WordVector')

	x_fc1 = Dense(32, activation='sigmoid')(input_WordVector)
	x_fc1 = Dense(16, activation='sigmoid')(x_fc1)

	m = Dense(3, activation='sigmoid')(x_fc1)

	x_fc2 = keras.layers.concatenate([input_Rmu, input_WordVector])
	x_fc2 = Dense(32, activation='sigmoid')(x_fc2)
	x_fc2 = Dense(10, activation='sigmoid')(x_fc2)
	alpha = Dense(1, activation='sigmoid')(x_fc2)

	m = Lambda(my_product, my_product_output_shape, name="m")([m, alpha])

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

	#beta = Dense(32, activation='sigmoid')(input_WordVector)
	beta = Dense(3, activation='sigmoid')(x_fc1)

	x_m_dot_R = keras.layers.concatenate([x_m1_dot_R, x_m2_dot_R, x_m3_dot_R])

	out_put_T = keras.layers.Add()([beta, x_m_dot_R])

	model = Model(inputs=[input_Rmu, input_WordVector], outputs=out_put_T)

	model.compile(optimizer='adam', loss='mse')
	
	return model
