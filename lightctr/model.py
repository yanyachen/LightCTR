import tensorflow as tf
from tensorflow.python.ops import variable_scope
from .head import binary_classification_estimator_spec
from .block import *
from .utils import *


def LR(features, labels, mode, params, config):
    '''
    feature_columns, sparse_combiner
    optimizer
    '''
    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['sparse_combiner']
        )
        logits = linear_block(
            features,
            feature_columns=params['feature_columns']
        )

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={'Linear': params['optimizer']}
    )


def DNN(features, labels, mode, params, config):
    '''
    feature_columns
    hidden_units, activation_fn
    dropout, batch_norm
    initializer, regularizer
    optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Main
    with variable_scope.variable_scope('DNN'):
        # Input
        input_units = tf.compat.v1.feature_column.input_layer(
            features=features,
            feature_columns=params['feature_columns']
        )
        # DNN
        dnn_block = DNN(
            hidden_units=params['hidden_units'],
            output_units=1,
            activation=params['activation_fn'],
            dropout=params['dropout'],
            batch_norm=params['batch_norm'],
            kernel_initializer=params['initializer'],
            kernel_regularizer=params['regularizer']
        )
        logits = dnn_block(
            inputs=input_units,
            training=is_training
        )

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={'DNN': params['optimizer']}
    )


def WideDeep(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    dnn_feature_columns
    dnn_hidden_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    dnn_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features,
            feature_columns=params['linear_feature_columns']
        )

    # DNN
    with variable_scope.variable_scope('DNN'):
        dnn_input_units = tf.compat.v1.feature_column.input_layer(
            features=features,
            feature_columns=params['dnn_feature_columns']
        )
        dnn_block = DNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=1,
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            batch_norm=params['dnn_batch_norm'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        dnn_logits = dnn_block(
            inputs=dnn_input_units,
            training=is_training
        )

    # Combine
    logits = linear_logits + dnn_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'DNN': params['dnn_optimizer']
        }
    )


def MLR(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner
    cluster_feature_columns, cluster_sparse_combiner
    num_cluster
    optimizer
    '''
    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=params['num_cluster'],
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # Cluster
    with variable_scope.variable_scope('Cluster'):
        cluster_block = Linear(
            output_units=params['num_cluster'],
            sparse_combiner=params['cluster_sparse_combiner']
        )
        cluster_logits = cluster_block(
            features=features,
            feature_columns=params['cluster_feature_columns'],
        )
        cluster_softmax = tf.nn.softmax(
            cluster_logits,
            axis=1,
            name='Softmax'
        )

    # Combine
    logits = tf.reduce_sum(
        input_tensor=tf.multiply(linear_logits, cluster_softmax),
        axis=-1
    )

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            None: params['optimizer']
        }
    )


def FM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    fm_feature_columns, embedding_size, fm_optimizer
    '''
    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # FM
    with variable_scope.variable_scope('FM'):
        embedding_columns = categorical_to_embedding(
            params['fm_feature_columns'],
            params['embedding_size']
        )
        fm_field_size = len(embedding_columns)
        fm_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        fm_block = FM()
        fm_output_units = fm_block(
            fm_input_units,
            field_size=fm_field_size,
            embedding_size=params['embedding_size']
        )
        fm_logits = tf.reduce_sum(
            input_tensor=fm_output_units,
            axis=1,
            keepdims=True
        )

    # Combine
    logits = linear_logits + fm_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'FM': params['fm_optimizer']
        }
    )


def FwFM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    fwfm_feature_columns, embedding_size, fwfm_optimizer
    '''
    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # FM
    with variable_scope.variable_scope('FM'):
        embedding_columns = categorical_to_embedding(
            params['fwfm_feature_columns'],
            params['embedding_size']
        )
        fm_field_size = len(embedding_columns)
        fm_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        fwfm_block = FwFM()
        fm_output_units = fwfm_block(
            tf.reshape(
                fm_input_units,
                shape=(-1, fm_field_size, params['embedding_size'])
            ),
            field_size=fm_field_size,
            embedding_size=params['embedding_size']
        )
        fm_logits = tf.reduce_sum(
            input_tensor=fm_output_units,
            axis=1,
            keepdims=True
        )

    # Combine
    logits = linear_logits + fm_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'FM': params['fwfm_optimizer']
        }
    )


def FFM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    ffm_feature_columns, embedding_size
    ffm_optimizer
    '''
    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # FFM
    with variable_scope.variable_scope('FFM'):
        ffm_field_size = len(params['ffm_feature_columns'])
        ffm_logits = []
        for i in range(0, ffm_field_size, +1):
            for j in range(i+1, ffm_field_size, +1):
                field_i_feature = params['ffm_feature_columns'][i]
                field_j_feature = params['ffm_feature_columns'][j]
                embedding_columns = categorical_to_embedding(
                    categorical_columns=list(
                        set(field_i_feature + field_j_feature)
                    ),
                    embedding_size=params['embedding_size']
                )
                fm_field_size = len(embedding_columns)
                fm_input_units = tf.compat.v1.feature_column.input_layer(
                    features, embedding_columns
                )
                fm_block = FM()
                fm_output_units = fm_block(
                    fm_input_units,
                    field_size=fm_field_size,
                    embedding_size=params['embedding_size']
                )
                fm_logits = tf.reduce_sum(
                    input_tensor=fm_output_units,
                    axis=1,
                    keepdims=True
                )
                ffm_logits.append(fm_logits)

    logits = tf.add_n([linear_logits] + ffm_logits)

    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'FFM': params['ffm_optimizer']
        }
    )


def NFM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    nfm_feature_columns, embedding_size
    dnn_hidden_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    nfm_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # NFM
    with variable_scope.variable_scope('NFM'):
        embedding_columns = categorical_to_embedding(
            params['nfm_feature_columns'],
            params['embedding_size']
        )
        fm_field_size = len(embedding_columns)
        fm_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        fm_block = FM()
        fm_output_units = fm_block(
            fm_input_units,
            field_size=fm_field_size,
            embedding_size=params['embedding_size']
        )
        dnn_block = DNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=1,
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            batch_norm=params['dnn_batch_norm'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        nfm_logits = dnn_block(
            inputs=fm_output_units,
            training=is_training
        )

    # Combine
    logits = linear_logits + nfm_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'NFM': params['nfm_optimizer']
        }
    )


def AFM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    afm_feature_columns, embedding_size
    dnn_hidden_units, dnn_activation_fn
    attention_dropout
    dnn_initializer, dnn_regularizer
    afm_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    with variable_scope.variable_scope('AFM'):
        # Bi-Interaction
        embedding_columns = categorical_to_embedding(
            params['afm_feature_columns'],
            params['embedding_size']
        )
        fm_field_size = len(embedding_columns)
        fm_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        feature_i, feature_j = pairwise_feature(
            fm_input_units, fm_field_size, params['embedding_size']
        )
        fm_product_units = tf.multiply(feature_i, feature_j)
        num_interactions = int(fm_field_size * (fm_field_size - 1) / 2)
        # Attention
        attention_block = AdditiveAttention(
            hidden_units=params['dnn_hidden_units'],
            activation=params['dnn_activation_fn'],
            dropout=params['attention_dropout'],
            pooling='sum',
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        fm_product_attention = attention_block(
            fm_product_units,
            field_size=num_interactions,
            embedding_size=params['embedding_size'],
            training=is_training
        )
        afm_linear_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        afm_logits = afm_linear_layer(fm_product_attention)

    # Combine
    logits = linear_logits + afm_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'AFM': params['afm_optimizer']
        }
    )


def DeepFM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    deepfm_feature_columns, embedding_size
    dnn_hidden_units, dnn_output_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    deepfm_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # DeepFM
    with variable_scope.variable_scope('DeepFM'):
        # FM
        embedding_columns = categorical_to_embedding(
            params['deepfm_feature_columns'],
            params['embedding_size']
        )
        fm_field_size = len(embedding_columns)
        fm_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        fm_block = FM()
        fm_output_units = fm_block(
            fm_input_units,
            field_size=fm_field_size,
            embedding_size=params['embedding_size']
        )
        # DNN
        dnn_block = DNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=params['dnn_output_units'],
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            batch_norm=params['dnn_batch_norm'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        dnn_output_units = dnn_block(
            inputs=fm_input_units,
            training=is_training
        )
        # Combine
        deepfm_units = tf.concat(
            [linear_logits, fm_output_units, dnn_output_units],
            axis=-1
        )
        deepfm_linear_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        logits = deepfm_linear_layer(deepfm_units)

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'DeepFM': params['deepfm_optimizer']
        }
    )


def MVM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    mvm_feature_columns, embedding_size, mvm_optimizer
    '''
    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # MVM
    with variable_scope.variable_scope('MVM'):
        embedding_columns = categorical_to_embedding(
            params['mvm_feature_columns'], params['embedding_size']
        )
        mvm_field_size = len(embedding_columns)
        mvm_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        mvm_block = MVM()
        mvm_output_units = mvm_block(
            tf.reshape(
                mvm_input_units,
                (-1, mvm_field_size, params['embedding_size'])
            ),
            field_size=mvm_field_size,
            embedding_size=params['embedding_size']
        )
        mvm_logits = tf.reduce_sum(
            input_tensor=mvm_output_units,
            axis=1,
            keepdims=True
        )

    # Combine
    logits = linear_logits + mvm_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'MVM': params['mvm_optimizer']
        }
    )


def DeepMVM(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    deepmvm_feature_columns, embedding_size
    dnn_hidden_units, dnn_output_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    deepmvm_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # DeepMVM
    with variable_scope.variable_scope('DeepMVM'):
        # MVM
        embedding_columns = categorical_to_embedding(
            params['deepmvm_feature_columns'], params['embedding_size']
        )
        mvm_field_size = len(embedding_columns)
        mvm_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        mvm_block = MVM()
        mvm_output_units = mvm_block(
            tf.reshape(
                mvm_input_units,
                (-1, mvm_field_size, params['embedding_size'])
            ),
            field_size=mvm_field_size,
            embedding_size=params['embedding_size']
        )
        # DNN
        dnn_block = DNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=params['dnn_output_units'],
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            batch_norm=params['dnn_batch_norm'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        dnn_output_units = dnn_block(
            inputs=mvm_input_units,
            training=is_training
        )
        # Combine
        deepmvm_units = tf.concat(
            [linear_logits, mvm_output_units, dnn_output_units],
            axis=-1
        )
        deepmvm_linear_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        logits = deepmvm_linear_layer(deepmvm_units)

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'DeepMVM': params['deepmvm_optimizer']
        }
    )


def DeepCrossing(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner
    linear_output_units, linear_optimizer
    dnn_feature_columns
    dnn_hidden_units, dnn_activation_fn
    dnn_dropout
    dnn_initializer, dnn_regularizer
    dnn_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=params['linear_output_units'],
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features=features,
            feature_columns=params['linear_feature_columns'],
        )

    # Residual DNN
    with variable_scope.variable_scope('DNN'):
        dnn_input_units = tf.compat.v1.feature_column.input_layer(
            features=features,
            feature_columns=params['dnn_feature_columns']
        )
        dnn_block = ResidualDNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=1,
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        logits = dnn_block(
            inputs=tf.concat(
                [linear_logits, dnn_input_units],
                axis=-1
            ),
            training=is_training
        )

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'DNN': params['dnn_optimizer']
        }
    )


def DCN(features, labels, mode, params, config):
    '''
    feature_columns
    dcn_num_layer
    dnn_hidden_units, dnn_output_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Input
    input_units = tf.compat.v1.feature_column.input_layer(
        features=features,
        feature_columns=params['feature_columns']
    )

    # DCN
    with variable_scope.variable_scope('DCN'):
        cross_block = CrossNet(params['dcn_num_layer'])
        cross_output_units = cross_block(input_units)

    # DNN
    with variable_scope.variable_scope('DNN'):
        dnn_block = DNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=params['dnn_output_units'],
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            batch_norm=params['dnn_batch_norm'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        dnn_output_units = dnn_block(
            inputs=input_units,
            training=is_training
        )

    # Combine
    with variable_scope.variable_scope('Linear'):
        dcn_output_units = tf.concat(
            [cross_output_units, dnn_output_units],
            axis=-1
        )
        dcn_linear_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        logits = dcn_linear_layer(dcn_output_units)

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={None: params['optimizer']}
    )


def PNN(features, labels, mode, params, config):
    '''
    linear_feature_columns, linear_sparse_combiner, linear_optimizer
    pnn_feature_columns, embedding_size
    dnn_hidden_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    pnn_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features,
            feature_columns=params['linear_feature_columns']
        )

    # PNN
    with variable_scope.variable_scope('PNN'):
        embedding_columns = categorical_to_embedding(
            params['pnn_feature_columns'], params['embedding_size']
        )
        pnn_field_size = len(embedding_columns)
        pnn_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )
        inner_product_block = InnerProduct()
        kernel_product_block = KernelProduct(
            kernel_type='mat', trainable=False
        )
        pnn_inner_product = inner_product_block(
            tf.reshape(
                pnn_input_units,
                (-1, pnn_field_size, params['embedding_size'])
            ),
            field_size=pnn_field_size,
            embedding_size=params['embedding_size']
        )
        pnn_outer_product = kernel_product_block(
            tf.reshape(
                pnn_input_units,
                (-1, pnn_field_size, params['embedding_size'])
            ),
            field_size=pnn_field_size,
            embedding_size=params['embedding_size']
        )
        dnn_input_units = tf.concat(
            values=[
                    pnn_input_units,
                    pnn_inner_product,
                    pnn_outer_product
                ],
            axis=-1
        )
        dnn_block = DNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=1,
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            batch_norm=params['dnn_batch_norm'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        dnn_logits = dnn_block(
            inputs=dnn_input_units,
            training=is_training
        )

    # Combine
    logits = linear_logits + dnn_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Linear': params['linear_optimizer'],
            'PNN': params['pnn_optimizer']
        }
    )


def xDeepFM(features, labels, mode, params, config):
    '''
    feature_columns, embedding_size
    linear_sparse_combiner
    cin_hidden_units, cin_activation_fn, cin_skip
    dnn_hidden_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features,
            feature_columns=params['feature_columns']
        )

    # Embedding
    with variable_scope.variable_scope('CIN'):
        embedding_columns = categorical_to_embedding(
                params['feature_columns'], params['embedding_size']
            )
        cin_field_size = len(embedding_columns)
        dnn_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )

    # CIN
    with variable_scope.variable_scope('CIN'):
        cin_block = CIN(
            hidden_units=params['cin_hidden_units'],
            activation=params['cin_activation_fn'],
            skip=params['cin_skip']
        )
        cin_output_units = cin_block(
            tf.reshape(
                dnn_input_units,
                (-1, cin_field_size, params['embedding_size'])
            ),
            field_size=cin_field_size,
            embedding_size=params['embedding_size']
        )
        cin_linear_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        cin_logits = cin_linear_layer(cin_output_units)

    # DNN
    with variable_scope.variable_scope('DNN'):
        dnn_block = DNN(
            hidden_units=params['dnn_hidden_units'],
            output_units=1,
            activation=params['dnn_activation_fn'],
            dropout=params['dnn_dropout'],
            batch_norm=params['dnn_batch_norm'],
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        dnn_logits = dnn_block(
            inputs=dnn_input_units,
            training=is_training
        )

    # Combine
    logits = linear_logits + cin_logits + dnn_logits

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            None: params['optimizer']
        }
    )


def AutoInt(features, labels, mode, params, config):
    '''
    feature_columns, embedding_size
    linear_sparse_combiner
    int_hidden_units, int_num_heads
    int_dropout, int_activation_fn, int_residual
    dnn_hidden_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Linear
    with variable_scope.variable_scope('Linear'):
        linear_block = Linear(
            output_units=1,
            sparse_combiner=params['linear_sparse_combiner']
        )
        linear_logits = linear_block(
            features,
            feature_columns=params['feature_columns']
        )

    # Embedding
    with variable_scope.variable_scope('Interacting'):
        embedding_columns = categorical_to_embedding(
                params['feature_columns'], params['embedding_size']
            )
        int_field_size = len(embedding_columns)
        dnn_input_units = tf.compat.v1.feature_column.input_layer(
            features, embedding_columns
        )

    # Interacting
    with variable_scope.variable_scope('Interacting'):
        interacting_block = Interacting(
            hidden_units=params['int_hidden_units'],
            num_heads=params['int_num_heads'],
            dropout=params['int_dropout'],
            activation=params['int_activation_fn'],
            residual=params['int_residual'],
        )
        int_output_units = interacting_block(
            dnn_input_units,
            field_size=int_field_size,
            embedding_size=params['embedding_size'],
            training=is_training
        )
        int_linear_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            kernel_initializer=params['dnn_initializer'],
            kernel_regularizer=params['dnn_regularizer']
        )
        int_logits = int_linear_layer(int_output_units)

        # DNN
        with variable_scope.variable_scope('DNN'):
            dnn_block = DNN(
                hidden_units=params['dnn_hidden_units'],
                output_units=1,
                activation=params['dnn_activation_fn'],
                dropout=params['dnn_dropout'],
                batch_norm=params['dnn_batch_norm'],
                kernel_initializer=params['dnn_initializer'],
                kernel_regularizer=params['dnn_regularizer']
            )
            dnn_logits = dnn_block(
                inputs=dnn_input_units,
                training=is_training
            )
        # Combine
        logits = linear_logits + int_logits + dnn_logits

        # Return
        return binary_classification_estimator_spec(
            mode=mode,
            labels=labels,
            logits=logits,
            optimizers={
                None: params['optimizer']
            }
        )


def FiBiNET(features, labels, mode, params, config):
    '''
    feature_columns, embedding_size
    reduction_ratio
    interaction_type
    dnn_hidden_units, dnn_activation_fn
    dnn_dropout, dnn_batch_norm
    dnn_initializer, dnn_regularizer
    optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Input
    num_feature = len(params['feature_columns'])
    embedding_columns = categorical_to_embedding(
        categorical_columns=params['feature_columns'],
        embedding_size=params['embedding_size']
    )
    input_units = tf.compat.v1.feature_column.input_layer(
        features=features,
        feature_columns=embedding_columns
    )

    # SENET
    feature_input_units = tf.keras.backend.reshape(
        input_units,
        shape=(-1, num_feature, params['embedding_size'])
    )

    feature_senet_block = SENET(
        reduction_ratio=params['reduction_ratio'],
        activation=params['dnn_activation_fn'],
        kernel_initializer=params['dnn_initializer'],
        kernel_regularizer=params['dnn_regularizer']
    )
    feature_scaled_units = feature_senet_block(
        inputs=feature_input_units,
        field_size=num_feature,
        embedding_size=params['embedding_size'],
        training=is_training
    )

    # BilinearInteraction
    feature_bilinear = BilinearInteraction(
        interaction_type=params['interaction_type']
    )(
        inputs=feature_input_units,
        field_size=num_feature,
        embedding_size=params['embedding_size']
    )

    feature_scaled_bilinear = BilinearInteraction(
        interaction_type=params['interaction_type']
    )(
        inputs=feature_scaled_units,
        field_size=num_feature,
        embedding_size=params['embedding_size']
    )

    # DNN
    dnn_input_units = tf.reshape(
        tf.concat([feature_bilinear, feature_scaled_bilinear], axis=1),
        shape=(-1, 2 * num_feature * num_feature * params['embedding_size'])
    )
    dnn_block = DNN(
        hidden_units=params['dnn_hidden_units'],
        output_units=1,
        activation=params['dnn_activation_fn'],
        dropout=params['dnn_dropout'],
        batch_norm=params['dnn_batch_norm'],
        kernel_initializer=params['dnn_initializer'],
        kernel_regularizer=params['dnn_regularizer']
    )
    logits = dnn_block(
        inputs=dnn_input_units,
        training=is_training
    )

    # Return
    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            None: params['optimizer']
        }
    )


def DeepInt(features, labels, mode, params, config):
    '''
    feature_columns
    num_interaction_layer, num_sub_spaces
    activation_fn
    dropout, residual
    embedding_size
    initializer, regularizer
    embedding_optimizer, pin_optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Input
    with variable_scope.variable_scope('Embedding'):
        num_feature = len(params['feature_columns'])
        embedding_columns = categorical_to_embedding(
            categorical_columns=params['feature_columns'],
            embedding_size=params['embedding_size']
        )
        input_units = tf.compat.v1.feature_column.input_layer(
            features=features,
            feature_columns=embedding_columns
        )
        feature_input_units = tf.keras.backend.reshape(
            input_units,
            shape=(-1, num_feature, params['embedding_size'])
        )

    with variable_scope.variable_scope('PIN'):
        # Interaction
        feature_interaction_block = Polynomial(
            num_interaction_layer=params['num_interaction_layer'],
            num_sub_spaces=params['num_sub_spaces'],
            activation=params['activation_fn'],
            dropout=params['dropout'],
            residual=params['residual'],
            initializer=params['initializer'],
            regularizer=params['regularizer']
        )
        feature_interaction_units = feature_interaction_block(
            inputs=feature_input_units,
            field_size=num_feature,
            embedding_size=params['embedding_size'],
            training=is_training
        )

        # Output
        vector_linear_block = VectorDense(
            units=1,
            activation=tf.keras.activations.linear,
            use_bias=True,
            kernel_initializer=params['initializer'],
            kernel_regularizer=params['regularizer'],
            dropout=None
        )
        vector_logits = vector_linear_block(
            inputs=feature_interaction_units,
            training=is_training,
        )

        logits = tf.reduce_sum(
            input_tensor=tf.keras.backend.squeeze(vector_logits, axis=1),
            axis=1,
            keepdims=True
        )

    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={
            'Embedding': params['embedding_optimizer'],
            'PIN': params['pin_optimizer']
        }
    )


def xDeepInt(features, labels, mode, params, config):
    '''
    feature_columns
    hidden_size, num_heads
    num_interaction_layer, num_sub_spaces
    activation_fn
    dropout, residual
    embedding_size
    initializer, regularizer
    optimizer
    '''
    # Prep
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Input
    num_feature = len(params['feature_columns'])
    embedding_columns = categorical_to_embedding(
        categorical_columns=params['feature_columns'],
        embedding_size=params['embedding_size']
    )
    input_units = tf.compat.v1.feature_column.input_layer(
        features=features,
        feature_columns=embedding_columns
    )
    feature_input_units = tf.keras.backend.reshape(
        input_units,
        shape=(-1, num_feature, params['embedding_size'])
    )

    # Interaction
    feature_interaction_block = MultiHeadPolynomial(
        hidden_size=params['hidden_size'],
        num_heads=params['num_heads'],
        num_interaction_layer=params['num_interaction_layer'],
        num_sub_spaces=params['num_sub_spaces'],
        activation=params['activation_fn'],
        dropout=params['dropout'],
        residual=params['residual'],
        initializer=params['initializer'],
        regularizer=params['regularizer']
    )
    feature_interaction_units = feature_interaction_block(
        inputs=feature_input_units,
        field_size=num_feature,
        embedding_size=params['embedding_size'],
        training=is_training
    )

    # Output
    vector_linear_block = VectorDense(
        units=1,
        activation=tf.keras.activations.linear,
        use_bias=True,
        kernel_initializer=params['initializer'],
        kernel_regularizer=params['regularizer'],
        dropout=None
    )
    vector_logits = vector_linear_block(
        inputs=feature_interaction_units,
        training=is_training,
    )

    logits = tf.reduce_sum(
        input_tensor=tf.keras.backend.squeeze(vector_logits, axis=1),
        axis=1,
        keepdims=True
    )

    return binary_classification_estimator_spec(
        mode=mode,
        labels=labels,
        logits=logits,
        optimizers={None: params['optimizer']}
    )
