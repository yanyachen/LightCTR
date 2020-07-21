import tensorflow as tf
from tensorflow.python.ops import variable_scope
from .layer import *
from .utils import pairwise_feature


class Linear(object):

    def __init__(
        self,
        output_units=1,
        sparse_combiner='sqrtn'
    ):
        self.output_units = output_units
        self.sparse_combiner = sparse_combiner

    def __call__(
        self,
        features,
        feature_columns
    ):
        outputs = tf.compat.v1.feature_column.linear_model(
            features=features,
            feature_columns=feature_columns,
            units=self.output_units,
            sparse_combiner=self.sparse_combiner,
            weight_collections=None,
            trainable=True,
            cols_to_vars=None
        )
        return outputs


class DNN(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        output_units,
        activation=None,
        dropout=None,
        batch_norm=None,
        kernel_initializer=None,
        kernel_regularizer=None
    ):
        # Init
        super(DNN, self).__init__()
        self.dnn = tf.keras.Sequential()

        # Input_Layer
        if dropout is not None:
            self.dnn.add(tf.keras.layers.Dropout(rate=dropout[0]))

        # Hidden_Layer
        for layer_id, num_hidden_units in enumerate(hidden_units):

            self.dnn.add(
                tf.keras.layers.Dense(
                    units=num_hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )

            if batch_norm is not None and batch_norm[layer_id]:
                self.dnn.add(tf.keras.layers.BatchNormalization())

            if dropout is not None and dropout[layer_id+1] > 0:
                self.dnn.add(
                    tf.keras.layers.Dropout(rate=dropout[layer_id+1])
                )

        # Output_Layer
        self.dnn.add(
            tf.keras.layers.Dense(
                units=output_units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        )

    def call(
        self,
        inputs,
        training
    ):
        outputs = self.dnn(
            inputs=inputs,
            training=training
        )
        return outputs


class ResidualDNN(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        output_units,
        activation=None,
        dropout=None,
        kernel_initializer=None,
        kernel_regularizer=None
    ):
        super(ResidualDNN, self).__init__()
        self.residual_dnn = tf.keras.Sequential()

        # Hidden_Layer
        for layer_id, num_hidden_units in enumerate(hidden_units):

            self.residual_dnn.add(
                ResidualDense(
                    units=num_hidden_units,
                    activation=activation,
                    dropout=dropout[layer_id] if dropout is not None else None,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )

        # Output_Layer
        self.residual_dnn.add(
            tf.keras.layers.Dense(
                units=output_units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        )

    def call(
        self,
        inputs,
        training
    ):
        outputs = self.residual_dnn(
            inputs=inputs,
            training=training
        )
        return outputs


class FM(tf.keras.Model):

    def __init__(self):
        super(FM, self).__init__()

    def call(
        self,
        inputs,
        field_size,
        embedding_size
    ):
        # Input
        inputs = tf.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )
        # FM
        sum_square = tf.square(tf.reduce_sum(input_tensor=inputs, axis=1))
        square_sum = tf.reduce_sum(input_tensor=tf.square(inputs), axis=1)
        fm_units = 0.5 * tf.subtract(sum_square, square_sum)

        return fm_units


class FwFM(tf.keras.Model):

    def __init__(self):
        super(FwFM, self).__init__()

    def build(
        self,
        input_shape
    ):
        field_size = input_shape[1].value
        num_interactions = int(field_size * (field_size - 1) // 2)

        self.interaction_weight = tf.Variable(
            tf.ones(shape=(num_interactions, 1), dtype=tf.float32),
            trainable=True, name='interaction_weight'
        )

    def call(
        self,
        inputs,
        field_size,
        embedding_size
    ):
        # Input
        feature_i, feature_j = pairwise_feature(
            inputs,
            field_size,
            embedding_size
        )
        # FwFM
        outputs = tf.reduce_sum(
            input_tensor=self.interaction_weight * feature_i * feature_j,
            axis=1
        )

        return outputs


class MVM(tf.keras.Model):

    def __init__(self):
        super(MVM, self).__init__()

    def build(
        self,
        input_shape
    ):
        field_size = input_shape[1].value
        embedding_size = input_shape[2].value
        self.bias = tf.Variable(
            tf.zeros(shape=(field_size, embedding_size)),
            trainable=True,
            dtype=tf.float32
        )

    def call(
        self,
        inputs,
        field_size,
        embedding_size
    ):
        # Input
        inputs = tf.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )
        inputs = tf.add(self.bias, inputs)
        # MVM
        mvm_units = tf.ones_like(inputs[:, 0, :])
        for i in range(0, field_size):
            mvm_units = tf.multiply(mvm_units, inputs[:, i, :])
        mvm_units = tf.reshape(mvm_units, shape=(-1, embedding_size))

        return mvm_units


class CrossNet(tf.keras.Model):

    def __init__(
        self,
        num_layer
    ):
        super(CrossNet, self).__init__()
        self.num_layer = num_layer

    def build(
        self,
        input_shape
    ):
        last_dim = input_shape[-1].value
        self.bias = tf.Variable(
            tf.zeros(shape=(self.num_layer, last_dim)),
            trainable=True,
            dtype=tf.float32
        )
        self.weight = tf.Variable(
            tf.zeros(shape=(self.num_layer, last_dim)),
            trainable=True,
            dtype=tf.float32
        )

    def call(
        self,
        inputs
    ):
        outputs = inputs
        for i in range(0, self.num_layer, +1):
            outputs += tf.add(
                self.bias[i, :],
                tf.multiply(
                    inputs,
                    tf.tensordot(
                        tf.expand_dims(outputs, axis=1),
                        self.weight[i, :],
                        axes=1
                    )
                )
            )

        return outputs


class InnerProduct(tf.keras.Model):

    def __init__(self):
        super(InnerProduct, self).__init__()

    def call(
        self,
        inputs,
        field_size,
        embedding_size
    ):
        feature_i, feature_j = pairwise_feature(
            inputs, field_size, embedding_size
        )
        inner_product = tf.reduce_sum(
            input_tensor=tf.multiply(feature_i, feature_j),
            axis=-1
        )

        return inner_product


class KernelProduct(tf.keras.Model):

    def __init__(
        self,
        kernel_type,
        trainable=False
    ):
        super(KernelProduct, self).__init__()
        self.kernel_type = kernel_type
        self.trainable = trainable

    def build(
        self,
        input_shape
    ):
        # Prep
        field_size = input_shape[1].value
        embedding_size = input_shape[2].value
        num_interactions = int(field_size * (field_size - 1) / 2)
        # Kernel
        if self.kernel_type == 'mat':
            self.kernel = tf.Variable(
                tf.tile(
                    tf.expand_dims(
                        tf.eye(embedding_size, dtype=tf.float32), axis=1
                    ),
                    multiples=(1, num_interactions, 1)
                ),
                trainable=self.trainable,
                dtype=tf.float32
            )
        elif self.kernel_type == 'vec':
            self.kernel = tf.Variable(
                tf.ones(
                    shape=(num_interactions, embedding_size),
                    dtype=tf.float32
                ),
                trainable=self.trainable,
                dtype=tf.float32
            )
        elif self.kernel_type == 'num':
            self.kernel = tf.Variable(
                tf.ones(shape=(num_interactions, 1), dtype=tf.float32),
                trainable=self.trainable,
                dtype=tf.float32
            )

    def call(
        self,
        inputs,
        field_size,
        embedding_size
    ):
        # Input
        inputs = tf.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )
        feature_i, feature_j = pairwise_feature(
            inputs, field_size, embedding_size
        )
        # Product
        if self.kernel_type == 'mat':
            feature_ik = tf.reduce_sum(
                input_tensor=tf.multiply(
                    tf.expand_dims(feature_i, axis=1),
                    self.kernel
                ),
                axis=1
            )
        else:
            feature_ik = tf.multiply(feature_i, self.kernel)
        # Kernel Product
        kernel_product = tf.reduce_sum(
            input_tensor=tf.multiply(feature_ik, feature_j),
            axis=-1
        )

        return kernel_product


class CIN(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        activation=tf.keras.activations.linear,
        skip=False
    ):
        super(CIN, self).__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.skip = skip

    def build(
        self,
        input_shape
    ):
        # Prep
        field_size = input_shape[1].value
        embedding_size = input_shape[2].value
        self.layer_field_num = []
        self.filter = []
        self.bias = []
        # Hidden_Layer
        self.layer_field_num.append(field_size)
        for layer_id, num_hidden_units in enumerate(self.hidden_units):
            # Filter
            self.filter.append(
                tf.Variable(
                    tf.compat.v1.glorot_uniform_initializer()((
                        1,
                        self.layer_field_num[0] *
                        self.layer_field_num[layer_id],
                        num_hidden_units
                    )),
                    trainable=True,
                    dtype=tf.float32
                )
            )
            # Bias
            self.bias.append(
                tf.Variable(
                    tf.zeros((num_hidden_units)),
                    trainable=True,
                    dtype=tf.float32
                )
            )
            # Skip
            if not self.skip:
                self.layer_field_num.append(num_hidden_units)
            else:
                self.layer_field_num.append(num_hidden_units // 2)

    def call(
        self,
        inputs,
        field_size,
        embedding_size
    ):
        # Prep
        cin_state_list = []
        output_list = []
        # Input
        inputs = tf.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )
        # CIN
        state_split_0 = tf.split(
            inputs, [1] * embedding_size, axis=2
        )
        cin_state_list.append(inputs)
        for layer_id, num_hidden_units in enumerate(self.hidden_units):
            # Z
            state_split_k = tf.split(
                cin_state_list[layer_id], [1] * embedding_size, axis=2
            )
            state_z_k = tf.transpose(
                a=tf.reshape(
                    tf.matmul(state_split_0, state_split_k, transpose_b=True),
                    shape=(
                        embedding_size,
                        -1,
                        self.layer_field_num[0] *
                        self.layer_field_num[layer_id]
                    )
                ),
                perm=[1, 0, 2]
            )
            # Feature Map
            state_z_k_conv = tf.nn.conv1d(
                input=state_z_k,
                filters=self.filter[layer_id],
                stride=1,
                padding='VALID'
            )
            state_z_k_conv_bias = tf.nn.bias_add(
                state_z_k_conv, self.bias[layer_id]
            )
            feature_map_k = tf.transpose(
                a=self.activation(state_z_k_conv_bias),
                perm=[0, 2, 1]
            )
            # Skip
            if not self.skip:
                state_unit = feature_map_k
                output_unit = feature_map_k
            else:
                if layer_id != len(self.hidden_units) - 1:
                    state_unit, output_unit = tf.split(
                        feature_map_k,
                        [num_hidden_units // 2] * 2,
                        axis=1
                    )
                else:
                    state_unit = 0
                    output_unit = feature_map_k
            # Add
            cin_state_list.append(state_unit)
            output_list.append(output_unit)

        # Return
        outputs = tf.reduce_sum(
            input_tensor=tf.concat(output_list, axis=1),
            axis=-1,
            keepdims=False
        )
        return outputs


class AdditiveAttention(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        activation,
        dropout,
        pooling,
        kernel_initializer,
        kernel_regularizer,
    ):
        super(AdditiveAttention, self).__init__()
        self.attention = DNN(
            hidden_units=hidden_units,
            output_units=1,
            activation=activation,
            dropout=None,
            batch_norm=None,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.dropout = dropout
        if self.dropout is not None and self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(
                rate=float(dropout)
            )
        self.pooling = pooling

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training,
        context=None
    ):
        # Input
        inputs = tf.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )
        attention_inputs = tf.reshape(
            inputs,
            shape=(-1, embedding_size)
        )

        if context is None:
            attention_inputs_context = attention_inputs
        else:
            attention_context = tf.reshape(
                tf.tile(context, multiples=(1, field_size)),
                shape=(-1, int(context.shape[-1]))
            )
            attention_inputs_context = tf.concat(
                [attention_inputs, attention_context],
                axis=-1
            )

        # Attention
        attention_outputs = self.attention(
            attention_inputs_context,
            training=training
        )
        attention_softmax = tf.keras.layers.Softmax(axis=1)(
            tf.reshape(
                attention_outputs,
                shape=(-1, field_size, 1)
            )
        )
        attention_embedding = tf.keras.layers.multiply(
            [attention_softmax, inputs]
        )

        # Pooling
        if self.pooling == 'average':
            attention_embedding = tf.reduce_mean(
                input_tensor=attention_embedding,
                axis=1
            )
        elif self.pooling == 'sum':
            attention_embedding = tf.reduce_sum(
                input_tensor=attention_embedding,
                axis=1
            )
        elif self.pooling == 'max':
            attention_embedding = tf.reduce_max(
                input_tensor=attention_embedding,
                axis=1
            )

        # Dropout
        if self.dropout is not None and self.dropout > 0:
            attention_embedding = self.dropout_layer(
                attention_embedding,
                training=training
            )

        return attention_embedding


class ScaledDotProductAttention(tf.keras.Model):

    def __init__(
        self,
        dropout
    ):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_layer = tf.keras.layers.Dropout(rate=float(dropout))

    def call(
        self,
        query,
        key,
        value,
        field_size,
        embedding_size,
        training
    ):
        # Input
        query = tf.reshape(
            query, shape=(-1, field_size, embedding_size)
        )
        key = tf.reshape(
            key, shape=(-1, field_size, embedding_size)
        )
        value = tf.reshape(
            value, shape=(-1, field_size, embedding_size)
        )

        # Attnetion
        attention_weights = tf.keras.backend.batch_dot(
            query, key,
            axes=[2, 2]
        )

        attention_softmax = tf.nn.softmax(
            attention_weights / tf.sqrt(float(embedding_size)),
            axis=-1
        )
        attention_softmax = self.dropout_layer(
            attention_softmax, training=training
        )

        attention_units = tf.keras.backend.batch_dot(
            attention_softmax, value,
            axes=[2, 1]
        )
        return attention_units


class GeneralAttention(tf.keras.Model):

    def __init__(
        self,
        dropout
    ):
        super(GeneralAttention, self).__init__()
        self.dropout_layer = tf.keras.layers.Dropout(rate=float(dropout))

    def build(self, input_shape):
        embedding_size = input_shape[-1].value
        self.kernel = tf.Variable(
            initial_value=tf.linalg.eye(embedding_size),
            trainable=True,
            dtype=tf.float32
        )

    def call(
        self,
        query,
        key,
        value,
        field_size,
        embedding_size,
        training
    ):
        # Input
        query = tf.reshape(
            query, shape=(-1, field_size, embedding_size)
        )
        key = tf.reshape(
            key, shape=(-1, field_size, embedding_size)
        )
        value = tf.reshape(
            value, shape=(-1, field_size, embedding_size)
        )

        # Attnetion
        attention_weights = tf.keras.backend.batch_dot(
            tf.keras.backend.dot(query, self.kernel),
            key,
            axes=[2, 2]
        )

        attention_softmax = tf.nn.softmax(
            attention_weights / tf.sqrt(float(embedding_size)),
            axis=-1
        )
        attention_softmax = self.dropout_layer(
            attention_softmax, training=training
        )

        attention_units = tf.keras.backend.batch_dot(
            attention_softmax, value,
            axes=[2, 1]
        )
        return attention_units


class MultiHeadAttention(tf.keras.Model):

    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout,
        activation=None
    ):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=activation,
            use_bias=False if activation is None else True
        )
        self.key_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=activation,
            use_bias=False if activation is None else True
        )
        self.value_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=activation,
            use_bias=False if activation is None else True
        )
        self.output_projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=None,
            use_bias=False
        )
        self.num_heads = num_heads
        self.dropout_layer = tf.keras.layers.Dropout(rate=float(dropout))

    def call(
        self,
        query,
        key,
        value,
        field_size,
        embedding_size,
        training
    ):
        # Input
        query = tf.reshape(
            query, shape=(-1, field_size, embedding_size)
        )
        key = tf.reshape(
            key, shape=(-1, field_size, embedding_size)
        )
        value = tf.reshape(
            value, shape=(-1, field_size, embedding_size)
        )

        # QKV Linear Projections
        query = self.query_projection_layer(query)
        key = self.key_projection_layer(key)
        value = self.value_projection_layer(value)

        # Multi-Head Split
        query = tf.concat(tf.split(query, self.num_heads, axis=2), axis=0)
        key = tf.concat(tf.split(key, self.num_heads, axis=2), axis=0)
        value = tf.concat(tf.split(value, self.num_heads, axis=2), axis=0)

        # Attnetion
        attention_weights = tf.keras.backend.batch_dot(
            query, key,
            axes=[2, 2]
        )

        depth = (self.hidden_size // self.num_heads)
        attention_softmax = tf.nn.softmax(
            attention_weights / tf.sqrt(float(depth)),
            axis=-1
        )
        attention_softmax = self.dropout_layer(
            attention_softmax, training=training
        )

        attention_units = tf.keras.backend.batch_dot(
            attention_softmax, value,
            axes=[2, 1]
        )

        # Multi-Head Combine
        attention_units = tf.concat(
            tf.split(attention_units, self.num_heads, axis=0), axis=2
        )

        # Output Linear Projection
        attention_units = self.output_projection_layer(attention_units)

        return attention_units


class Transformer(tf.keras.Model):

    def __init__(
        self,
        num_layer,
        attention_method,
        attention_hidden_size,
        attention_num_heads,
        attention_activation,
        attention_dropout,
        dnn_units,
        dnn_activation,
        dnn_dropout,
        initializer,
        regularizer
    ):
        # Init
        super(Transformer, self).__init__()

        self.projection_layer = tf.keras.layers.Dense(
            units=attention_hidden_size,
            activation=tf.keras.activations.linear,
            use_bias=False
        )

        if attention_method == 'Dot':
            self.attention_layers = [
                ScaledDotProductAttention(dropout=attention_dropout)
                for _ in range(num_layer)
            ]
        elif attention_method == 'General':
            self.attention_layers = [
                GeneralAttention(dropout=attention_dropout)
                for _ in range(num_layer)
            ]
        elif attention_method == 'MultiHead':
            self.attention_layers = [
                MultiHeadAttention(
                    hidden_size=attention_hidden_size,
                    num_heads=attention_num_heads,
                    activation=attention_activation,
                    dropout=attention_dropout
                )
                for _ in range(num_layer)
            ]
        else:
            raise ValueError('Attention Method Not Supported')

        self.dnn_layers = [
            ResidualDense(
                units=dnn_units,
                activation=dnn_activation,
                dropout=dnn_dropout,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                output_activation=tf.keras.activations.linear
            )
            for _ in range(num_layer)
        ]

        self.num_layer = num_layer
        self.attention_hidden_size = attention_hidden_size

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )

        # Interaction
        inputs = self.projection_layer(inputs)
        outputs = inputs

        for layer_id in range(0, self.num_layer, +1):

            outputs = LayerNormalization()(outputs)
            attention_units = self.attention_layers[layer_id](
                outputs,
                outputs,
                outputs,
                field_size=field_size,
                embedding_size=self.attention_hidden_size,
                training=training
            )
            outputs = tf.keras.layers.add(
                [outputs, attention_units]
            )

            outputs = LayerNormalization()(outputs)
            feedforward_units = self.dnn_layers[layer_id](
                outputs, training=training
            )
            outputs = tf.keras.layers.add(
                [outputs, feedforward_units]
            )

        # Output
        outputs = LayerNormalization()(outputs)

        return outputs


class Interacting(tf.keras.Model):

    def __init__(
        self,
        hidden_units,
        num_heads,
        dropout,
        activation,
        residual
    ):
        super(Interacting, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.residual = residual

        self.num_layer = len(self.hidden_units)
        self.attention_layers = []
        self.dnn_layers = []
        for layer_id, num_hidden_units in enumerate(self.hidden_units):
            self.attention_layers.append(
                MultiHeadAttention(
                    hidden_size=num_hidden_units,
                    num_heads=num_heads,
                    dropout=dropout[layer_id],
                    activation=activation
                )
            )
            if residual:
                self.dnn_layers.append(
                    tf.keras.layers.Dense(
                        units=num_hidden_units,
                        activation=activation,
                        use_bias=True
                    )
                )

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )

        # Interaction
        outputs = inputs
        for layer_id in range(0, self.num_layer, +1):

            attention_outputs = self.attention_layers[layer_id](
                outputs,
                outputs,
                outputs,
                field_size=field_size,
                embedding_size=(
                    embedding_size
                    if layer_id == 0
                    else self.hidden_units[layer_id-1]
                ),
                training=training
            )

            if self.residual:
                dnn_outputs = self.dnn_layers[layer_id](outputs)
                outputs = attention_outputs + dnn_outputs
            else:
                outputs = attention_outputs

            outputs = self.activation(outputs)
            outputs = LayerNormalization()(outputs)

        # Output
        outputs = tf.reshape(
            outputs,
            shape=(-1, field_size * self.hidden_units[layer_id])
        )
        return outputs


class SENET(tf.keras.Model):
    def __init__(
        self,
        reduction_ratio,
        activation,
        kernel_initializer,
        kernel_regularizer
    ):
        super(SENET, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        field_size = input_shape[1].value
        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D(
            data_format='channels_first'
        )
        self.dnn = tf.keras.models.Sequential()
        self.dnn.add(
            tf.keras.layers.Dense(
                units=field_size // self.reduction_ratio,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )
        )
        self.dnn.add(
            tf.keras.layers.Dense(
                units=field_size,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )
        )
        self.dnn.add(
            tf.keras.layers.Reshape((field_size, 1))
        )

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )
        # SENET
        squeeze = self.pooling_layer(inputs)
        excitation = self.dnn(squeeze)
        scale = tf.math.multiply(inputs, excitation)
        return scale


class BilinearInteraction(tf.keras.Model):
    def __init__(
        self,
        interaction_type
    ):
        super(BilinearInteraction, self).__init__()
        self.interaction_type = interaction_type

    def build(self, input_shape):
        field_size = input_shape[1].value
        embedding_size = input_shape[2].value

        if self.interaction_type == 'All':
            self.bilinear_kernel = self.add_weight(
                shape=(embedding_size, embedding_size),
                initializer=tf.compat.v1.glorot_uniform_initializer(),
                name='bilinear_kernel'
            )
        elif self.interaction_type == 'Each':
            self.bilinear_kernel = [
                self.add_weight(
                    shape=(embedding_size, embedding_size),
                    initializer=tf.compat.v1.glorot_uniform_initializer(),
                    name='bilinear_kernel' + '_' + str(i)
                )
                for i in range(field_size)
            ]
        elif self.interaction_type == 'Interaction':
            self.bilinear_kernel = [
                self.add_weight(
                    shape=(embedding_size, embedding_size),
                    initializer=tf.compat.v1.glorot_uniform_initializer(),
                    name='bilinear_kernel' + '_' + str(i) + str(j)
                )
                for i in range(field_size) for j in range(field_size)
            ]

    def call(
        self,
        inputs,
        field_size,
        embedding_size
    ):
        # Input
        inputs = tf.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )
        # Product
        if self.interaction_type == 'All':
            index_i, index_j = zip(*[
                [i, j]
                for i in range(field_size)
                for j in range(field_size)
            ])
            outputs = tf.multiply(
                tf.tensordot(
                    tf.gather(inputs, indices=index_i, axis=1),
                    self.bilinear_kernel,
                    axes=(-1, 0)
                ),
                tf.gather(inputs, indices=index_j, axis=1)
            )
        elif self.interaction_type == 'Each':
            output_list = []
            for i in range(field_size):
                index_i = [i for j in range(field_size)]
                index_j = [j for j in range(field_size)]
                output = tf.multiply(
                    tf.tensordot(
                        tf.gather(inputs, indices=index_i, axis=1),
                        self.bilinear_kernel[i],
                        axes=(-1, 0)
                    ),
                    tf.gather(inputs, indices=index_j, axis=1)
                )
                output_list.append(output)
            outputs = tf.concat(output_list, axis=1)
        elif self.interaction_type == 'Interaction':
            output_list = [
                tf.multiply(
                    tf.tensordot(
                        inputs[:, i, :],
                        self.bilinear_kernel[j + i * field_size],
                        axes=(-1, 0)
                    ),
                    inputs[:, j, :]
                )
                for i in range(field_size) for j in range(field_size)
            ]
            outputs = tf.concat(
                [tf.expand_dims(each, axis=1) for each in output_list],
                axis=1
            )
        # Concat
        return outputs


class Polynomial(tf.keras.Model):

    def __init__(
        self,
        num_interaction_layer,
        num_sub_spaces,
        activation,
        dropout,
        residual,
        initializer,
        regularizer
    ):
        super(Polynomial, self).__init__()
        self.num_interaction_layer = num_interaction_layer
        self.num_sub_spaces = num_sub_spaces
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        field_size = input_shape[1].value
        self.vector_dense_layers = [
            VectorDense(
                units=int(field_size * self.num_sub_spaces),
                activation=self.activation,
                use_bias=False,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                dropout=self.dropout[i] if self.dropout is not None else None
            )
            for i in range(self.num_interaction_layer)
        ]

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )

        # Split
        inputs = tf.concat(
            tf.split(inputs, self.num_sub_spaces, axis=2),
            axis=1
        )

        # Interaction
        interaction = inputs
        if not self.residual:
            interaction_list = []
            interaction_list.append(interaction)

        for layer_id in range(0, self.num_interaction_layer, +1):

            weighted_inputs = self.vector_dense_layers[layer_id](
                inputs,
                training=training
            )

            if self.residual:
                interaction = tf.keras.layers.multiply(
                    [interaction, (1.0 + weighted_inputs)]
                )
            else:
                interaction = tf.keras.layers.multiply(
                    [interaction, weighted_inputs]
                )
                interaction_list.append(interaction)

        # Output
        if self.residual:
            interaction_outputs = interaction
        else:
            interaction_outputs = tf.keras.backend.concatenate(
                interaction_list, axis=1
            )

        # Combine
        interaction_outputs = tf.concat(
            tf.split(interaction_outputs, self.num_sub_spaces, axis=1),
            axis=2
        )

        return interaction_outputs


class MultiHeadPolynomial(tf.keras.Model):

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_interaction_layer,
        num_sub_spaces,
        activation,
        dropout,
        residual,
        initializer,
        regularizer
    ):
        super(MultiHeadPolynomial, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_interaction_layer = num_interaction_layer
        self.num_sub_spaces = num_sub_spaces
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        self.projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=tf.keras.activations.linear,
            use_bias=False,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer
        )
        self.polynomial_block_list = [
            Polynomial(
                num_interaction_layer=self.num_interaction_layer[i],
                num_sub_spaces=self.num_sub_spaces[i],
                activation=self.activation[i],
                dropout=self.dropout,
                residual=self.residual,
                initializer=self.initializer,
                regularizer=self.regularizer
            )
            for i in range(self.num_heads)
        ]

    def call(
        self,
        inputs,
        field_size,
        embedding_size,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, field_size, embedding_size)
        )

        # Linear Projection
        inputs = self.projection_layer(inputs)

        # Split
        inputs_heads = tf.split(inputs, self.num_heads, axis=2)

        # Polynomial Interaction
        outputs_heads = [
            self.polynomial_block_list[i](
                inputs_heads[i],
                field_size=field_size,
                embedding_size=int(self.hidden_size / self.num_heads),
                training=training
            )
            for i in range(self.num_heads)
        ]

        # Combine
        outputs = tf.concat(outputs_heads, axis=1)

        # Return
        return outputs
