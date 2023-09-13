import tensorflow as tf

print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
	Dense,
	Input,
	Conv2D,
	Conv3D,
	Flatten,
	Lambda,
	Multiply,
	Reshape,
	MaxPooling3D,
	MaxPooling2D,
	MaxPooling1D,
	AveragePooling1D,
	AveragePooling2D,
	AveragePooling3D,
	Concatenate,
	Permute,
)
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as K

from utils import custom_loss_func, kernel_pwm_init, custom_regularizer_kl

### Load trained model
def load_trained_model(dict_params):
    file_checkpoint = dict_params["file_trained_model"]
    loss_func = dict_params["loss_func"]
    train_base_model = dict_params["train_base_model"]
    train_pwm_layer = dict_params["train_pwm_layer"]
    train_tf_layer = dict_params["train_tf_layer"]
    dict_layer_trainable = {
        "pwms_train_lyr": train_pwm_layer,
        "combine_pwms": train_tf_layer,
        "pwms_init_lyr":train_pwm_layer
    }
    if dict_params["conv_init"]:
        my_loss_func = custom_loss_func(loss_func)
    else:
        pwm_filters_train = dict_params["pwm_filters_train"]
        model = model_classifier(dict_params)
        my_loss_func = custom_loss_func(
            loss_func, model.layers[3].trainable_weights[0], pwm_filters_train, 1,
        )
    model_trained = keras.models.load_model(
        file_checkpoint, custom_objects={"loss": my_loss_func}
    )
    if train_base_model:
        model_trained.trainable = train_base_model
        for layer in model_trained.layers:
            if layer.name in ["pwms_train_lyr", "combine_pwms", "pwms_init_lyr"]:
                layer.trainable = dict_layer_trainable[layer.name]
    elif train_pwm_layer or train_tf_layer:
        model_trained.trainable = True
        for layer in model_trained.layers:
            if layer.name not in ["pwms_train_lyr", "combine_pwms", "pwms_init_lyr"]:
                layer.trainable = False
            else:
                layer.trainable = dict_layer_trainable[layer.name]
    else:
        model_trained.trainable = train_base_model
    return model_trained

def seq_annotator(dict_params):
	n_pwms = dict_params['n_pwms']
	l_pwms = dict_params['l_pwms']
	n_seqs = dict_params['n_seqs']
	n_cells = dict_params['n_cells']
	l_seqs = dict_params['l_seqs']
	init_seed_value = dict_params['init_seed_value']

	if dict_params['conv_init'] == 'true':
		act_fn_pwm = 'relu'
	else:
		act_fn_pwm = "relu"
		pwm_filters_fix = dict_params['pwm_filters_fix']
		print(pwm_filters_fix.shape)

	sequence_inputs = Input(shape=(2, 4 * n_seqs, l_seqs), name='sequence_inputs')

	# x = Reshape((2, 4 * n_seqs, l_seqs, 1), input_shape=(2, 4 * n_seqs, l_seqs))(sequence_inputs)
	x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(sequence_inputs)
	if dict_params['conv_init'] == 'true':
		x = Conv3D(
		    n_pwms,
		    (1, 4, l_pwms),
		    name="pwms_init_lyr",
		    kernel_initializer=keras.initializers.RandomNormal(seed=init_seed_value),
		    strides=(1, 4, 1),
		    use_bias=False,
		    activation=act_fn_pwm,
		    trainable=False,
		)(x)
	else:
		x = Conv3D(
		    pwm_filters_fix.shape[4],
		    (1, 4, l_pwms),
		    name="pwms_fix_layer",
		    kernel_initializer=kernel_pwm_init(pwm_filters_fix),
		    strides=(1, 4, 1),
		    use_bias=False,
		    activation=act_fn_pwm,
		    trainable=False,
		)(x)
	x_p, x_n = Lambda(lambda x: tf.split(x, 2, 1), name='rc_split')(x)
	x_n_r = Lambda(lambda x: tf.reverse(x, axis=[3]), name='rc_neg')(x_n)
	x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1), name='rc_concat')([x_p, x_n_r])
	x = MaxPooling3D(pool_size=(2, 1, 100), name='maxpool_pwms')(x)
	# x = Reshape((n_seqs, 9, n_pwms, 1), input_shape=(1, n_seqs, 9, n_pwms))(x)
	x = Permute((2, 3, 4, 1), input_shape=(1, n_seqs, 9, n_pwms))(x)
	x = AveragePooling3D(pool_size=(1, 9, 1), name='avg_pwms')(x)
	x = Lambda(lambda x: tf.squeeze(x, [2,4]))(x)
	# x = Permute((2,1), input_shape=(n_seqs, n_pwms), name='transpose_2')(x)
	# x = Reshape((n_seqs, n_pwms), input_shape=(n_seqs, 1, n_pwms, 1))(x)
	predictions = x 

	seq_annotations = Model(inputs=sequence_inputs, outputs=predictions)
	return seq_annotations


def model_classifier(dict_params):
	n_pwms = dict_params['n_pwms']
	l_pwms = dict_params['l_pwms']
	n_seqs = dict_params['n_seqs']
	n_cells = dict_params['n_cells']
	l_seqs = dict_params['l_seqs']
	wt_l1 = dict_params['wt_l1']
	wt_l2 = dict_params['wt_l2']
	shape_lyrs = dict_params['shape_lyrs']
	act_funcs = dict_params['act_funcs']
	init_seed_value = dict_params['init_seed_value']
	loss_func = dict_params['loss_func']
	wt_kl = dict_params['wt_kl']

	sequence_inputs = Input(shape=(2, 4 * n_seqs, l_seqs), name='sequence_inputs')
	cell_inputs = Input(shape=(n_seqs, n_cells), name='cell_inputs')

	# x = Reshape((2, 4 * n_seqs, l_seqs, 1), input_shape=(2, 4 * n_seqs, l_seqs))(sequence_inputs)
	x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(sequence_inputs)

	if dict_params['conv_init'] == 'true':
		x = Conv3D(
	    n_pwms,
	    (1, 4, l_pwms),
	    name="pwms_init_lyr",
	    kernel_initializer=keras.initializers.RandomNormal(seed=init_seed_value),
	    # kernel_regularizer=keras.regularizers.l1_l2(l1=wt_l1, l2=wt_l2),
	    strides=(1, 4, 1),
	    use_bias=False,
	    activation="relu",
	    trainable=True,
		)(x)
	elif dict_params['conv_init'] == 'all_true':
		pwm_filters_train = dict_params['pwm_filters_train']
		print(pwm_filters_train.shape)
		x = Conv3D(
		    pwm_filters_train.shape[4],
		    (1, 4, l_pwms),
		    name="pwms_train_lyr",
		    kernel_initializer=keras.initializers.RandomNormal(seed=init_seed_value),
		    # kernel_regularizer=keras.regularizers.l1_l2(l1=wt_l1, l2=wt_l2),
		    kernel_regularizer=custom_regularizer_kl(pwm_filters_train, wt_kl),
		    strides=(1, 4, 1),
		    use_bias=False,
		    activation="relu",
		    trainable=True,
		)(x)
	elif dict_params['conv_init'] == 'all':
		pwm_filters_train = dict_params['pwm_filters_train']
		print(pwm_filters_train.shape)
		x = Conv3D(
		    pwm_filters_train.shape[4],
		    (1, 4, l_pwms),
		    name="pwms_train_lyr",
		    kernel_initializer=kernel_pwm_init(pwm_filters_train),
		    # kernel_regularizer=keras.regularizers.l1_l2(l1=wt_l1, l2=wt_l2),
		    kernel_regularizer=custom_regularizer_kl(pwm_filters_train, wt_kl),
		    strides=(1, 4, 1),
		    use_bias=False,
		    activation="relu",
		    trainable=True,
		)(x)
	elif dict_params['conv_init'] == 'none':
		pwm_filters_fix = dict_params['pwm_filters_fix']
		print(pwm_filters_fix.shape)
		x = Conv3D(
		    pwm_filters_fix.shape[4],
		    (1, 4, l_pwms),
		    name="pwms_fix_layer",
		    kernel_initializer=kernel_pwm_init(pwm_filters_fix),
		    kernel_regularizer=keras.regularizers.l1_l2(l1=wt_l1, l2=wt_l2),
		    strides=(1, 4, 1),
		    use_bias=False,
		    activation="relu",
		    trainable=False,
		)(x)
	else:
		pwm_filters_train = dict_params['pwm_filters_train']	
		pwm_filters_fix = dict_params['pwm_filters_fix']
		pwms_order = dict_params['pwms_order']
		x_train = Conv3D(
		    pwm_filters_train.shape[4],
		    (1, 4, l_pwms),
		    name="pwms_train_lyr",
		    kernel_initializer=kernel_pwm_init(pwm_filters_train),
		    kernel_regularizer=keras.regularizers.l1_l2(l1=wt_l1, l2=wt_l2),
		    strides=(1, 4, 1),
		    use_bias=False,
		    activation="relu",
		    trainable=True,
		)(x)
		x_fix = Conv3D(
		    pwm_filters_fix.shape[4],
		    (1, 4, l_pwms),
		    name="pwms_fix_layer",
		    kernel_initializer=kernel_pwm_init(pwm_filters_fix),
		    kernel_regularizer=keras.regularizers.l1_l2(l1=wt_l1, l2=wt_l2),
		    strides=(1, 4, 1),
		    use_bias=False,
		    activation="relu",
		    trainable=False,
		)(x)
		x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1), name='concat_pwms')([x_fix, x_train])
		x = Lambda(lambda x: tf.gather(x, K.cast(pwms_order, dtype='int32'), axis=-1), name='reorder_pwms')(x)
	x_p, x_n = Lambda(lambda x: tf.split(x, 2, 1), name='rc_split')(x)
	x_n_r = Lambda(lambda x: tf.reverse(x, axis=[3]), name='rc_neg')(x_n)
	x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1), name='rc_concat')([x_p, x_n_r])

	x = MaxPooling3D(pool_size=(2, 1, 100), name='maxpool_pwms')(x)
	# x = Reshape((n_seqs, 9, n_pwms, 1), input_shape=(1, n_seqs, 9, n_pwms))(x)
	x = Permute((2, 3, 4, 1), input_shape=(1, n_seqs, 9, n_pwms))(x)
	x = AveragePooling3D(pool_size=(1, 9, 1), name='avg_pwms')(x)

	# x = MaxPooling3D(pool_size=(2, 1, 1), name='maxpool_pos')(x)
	# x = Reshape((n_seqs, n_pwms, l_seqs-l_pwms+1), input_shape=(1, n_seqs, l_seqs-l_pwms+1, n_pwms))(x)
	# x = Lambda(lambda x: tf.nn.top_k(x, k=10, sorted=False, name='maxpool_10_pwms').values)(x)
	# x = Reshape((n_seqs, n_pwms, 10, 1), input_shape=(n_seqs, n_pwms, 10))(x)
	# x = AveragePooling3D(pool_size=(1, 1, 10), name='avg_pwms')(x)

	# x = Reshape((n_pwms, n_seqs), input_shape=(n_seqs, n_pwms))(x)
	# x = Reshape((n_pwms, n_seqs, 1), input_shape=(n_pwms, n_seqs))(x)

	x = Lambda(lambda x: tf.squeeze(x, [2,4]))(x)
	x = Permute((2,1), input_shape=(n_seqs, n_pwms), name='transpose_2')(x)
	x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)

	x = Conv2D(
		n_cells,
		(n_pwms, 1),
		strides=(1, 1),
		activation=act_funcs[0],
		name='combine_pwms',
		use_bias=True,
		kernel_initializer=keras.initializers.RandomNormal(seed=init_seed_value),
		kernel_regularizer=keras.regularizers.l1_l2(l1=wt_l1, l2=wt_l2),
	)(x)
	x = Lambda(lambda x: tf.squeeze(x, [1]))(x)
	# x = Reshape(
	# 	(n_seqs, n_cells), input_shape=(1, n_seqs, n_cells)
	# )(x)
	x = Multiply(name='mult_acc_pwms')([x, cell_inputs])
	x = AveragePooling1D(pool_size=(n_seqs))(x)
	x = Flatten()(x)

	x_split = Lambda(lambda x: tf.split(x, n_cells, 1), name='cell_split')(x)
	x1 = Concatenate()([x_split[0], x_split[1]])
	x1 = Dense(1,use_bias=True,
		name='prediction_1',
		activation="tanh",
		kernel_initializer=keras.initializers.RandomNormal(seed=init_seed_value)
	)(x1)
	x1 = Reshape((1,1))(x1)

	if n_cells > 2:
		x2 = Concatenate()([x_split[0], x_split[2]])
		x2 = Dense(1,use_bias=True,
			name='prediction_2',
			activation="tanh",
			kernel_initializer=keras.initializers.RandomNormal(seed=init_seed_value)
		)(x2)
		x2 = Reshape((1,1))(x2)
		x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1), name='concat_cell_preds_1')([x1, x2])

		for i in range(3, n_cells):
			x_temp = Concatenate()([x_split[0], x_split[i]])
			x_temp = Dense(1,use_bias=True,
				name='prediction_' + str(i),
				activation="tanh",
				kernel_initializer=keras.initializers.RandomNormal(seed=init_seed_value)
			)(x_temp)
			x_temp = Reshape((1,1))(x_temp)
			x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1), name='concat_cell_preds_' + str(i-1))([x, x_temp])
		x = Flatten()(x)
	else:
		x = Flatten(x1)

	predictions = x 

	classifier = Model(inputs=[sequence_inputs, cell_inputs], outputs=predictions)
	return classifier 