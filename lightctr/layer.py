import tensorflow as tf


class ResidualDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        dropout=None,
        kernel_initializer=None,
        kernel_regularizer=None,
        output_activation=None
    ):
        super(ResidualDense, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        if output_activation is None:
            self.output_activation = self.activation
        else:
            self.output_activation = self.output_activation

    def build(self, input_shape):
        last_dim_units = input_shape[-1].value

        self.layer0 = tf.keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )

        if self.dropout is not None and self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(
                rate=float(self.dropout)
            )

        self.layer1 = tf.keras.layers.Dense(
            units=last_dim_units,
            activation=tf.keras.activations.linear,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )

    def call(self, inputs, training):
        net = self.layer0(inputs)
        if self.dropout is not None and self.dropout > 0:
            net = self.dropout_layer(net, training=training)
        net = self.layer1(net)
        outputs = self.activation(inputs + net)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(LayerNormalization, self).__init__()

    def build(self, input_shape):
        last_dim = input_shape[-1].value

        self.scale = tf.Variable(
            initial_value=tf.ones([last_dim]),
            trainable=True,
            name='layer_norm_scale',
            dtype=tf.float32,
        )
        self.bias = tf.Variable(
            initial_value=tf.zeros([last_dim]),
            trainable=True,
            name='layer_norm_bias',
            dtype=tf.float32,
        )

    def call(self, inputs, epsilon=1e-6):
        mean = tf.reduce_mean(
            input_tensor=inputs, axis=[-1], keepdims=True
        )
        variance = tf.reduce_mean(
            input_tensor=tf.square(inputs - mean), axis=[-1], keepdims=True
        )
        norm_inputs = (inputs - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_inputs * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class AdversarialNoise(tf.keras.layers.Layer):
    def __init__(self, eps):
        super(AdversarialNoise, self).__init__()
        self.eps = eps

    def _scale_l2(self, x):
        ndim = tf.keras.backend.ndim(x)
        feature_dims = [i for i in range(1, ndim)]
        alpha = tf.reduce_max(
            input_tensor=tf.abs(x),
            axis=feature_dims,
            keepdims=True
        ) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(
                input_tensor=tf.pow(x / alpha, 2),
                axis=feature_dims,
                keepdims=True
            ) + 1e-6
        )
        x_unit = x / l2_norm
        return x_unit

    def _truncated_normal_eps(self, x):
        ndim = tf.keras.backend.ndim(x)
        sample_eps = tf.keras.backend.truncated_normal(
                shape=tf.keras.backend.shape(x)[:1],
                mean=tf.cast(self.eps, dtype=tf.float32) / 2.0,
                stddev=tf.square(tf.cast(self.eps, dtype=tf.float32) / 4.0)
        )
        sample_eps = tf.tile(
            input=tf.reshape(
                sample_eps, [-1] + [1 for i in range(ndim-1)]
            ),
            multiples=[1] + list(tf.keras.backend.int_shape(x)[1:])
        )
        return sample_eps

    def call(self, inputs, loss, training):
        if training:
            inputs_grad = tf.gradients(
                ys=loss,
                xs=inputs,
                aggregation_method=(
                    tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
                )
            )
            inputs_grad_dense = tf.squeeze(
                tf.stop_gradient(inputs_grad), axis=0
            )
            noise_unit = self._scale_l2(inputs_grad_dense)
            sample_eps = self._truncated_normal_eps(noise_unit)
            noise = noise_unit * sample_eps
            return inputs + noise
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class TargetedDense(tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        targeted_dropout_type=None,
        target_rate=0.50,
        dropout_rate=0.50,
        **kwargs
    ):
        super(TargetedDense, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.targeted_dropout_type = targeted_dropout_type
        self.target_rate = target_rate
        self.dropout_rate = dropout_rate

    def targeted_weight_dropout(
        self, w, target_rate, dropout_rate, is_training
    ):
        w_shape = w.shape
        w = tf.reshape(w, [-1, w_shape[-1]])
        norm = tf.abs(w)
        idx = tf.cast(
            target_rate * tf.cast(tf.shape(input=w)[0], dtype=tf.float32),
            dtype=tf.int32
        )
        threshold = tf.sort(norm, axis=0)[idx]
        mask = norm < threshold[None, :]

        if not is_training:
            w = (1.0 - tf.cast(mask, dtype=tf.float32)) * w
            w = tf.reshape(w, w_shape)
            return w

        mask = tf.cast(
            tf.logical_and(
                tf.random.uniform(tf.shape(input=w)) < dropout_rate,
                mask
            ), dtype=tf.float32
        )
        w = (1.0 - mask) * w
        w = tf.reshape(w, w_shape)
        return w

    def targeted_unit_dropout(
        self, w, target_rate, dropout_rate, is_training
    ):
        w_shape = w.shape
        w = tf.reshape(w, [-1, w_shape[-1]])
        norm = tf.norm(tensor=w, axis=0)
        idx = int(target_rate * int(w.shape[1]))
        sorted_norms = tf.sort(norm)
        threshold = sorted_norms[idx]
        mask = (norm < threshold)[None, :]
        mask = tf.tile(mask, [w.shape[0], 1])

        mask = tf.compat.v1.where(
            tf.logical_and(
                (1.0 - dropout_rate) < tf.random.uniform(tf.shape(input=w)),
                mask
            ),
            tf.ones_like(w, dtype=tf.float32),
            tf.zeros_like(w, dtype=tf.float32)
        )
        w = (1.0 - mask) * w
        w = tf.reshape(w, w_shape)
        return w

    def call(self, inputs, training):
        inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
        rank = inputs._rank()

        if (self.targeted_dropout_type == 'weight'):
            self.kernel.assign(
                self.targeted_weight_dropout(
                    self.kernel,
                    self.target_rate,
                    self.dropout_rate,
                    training
                )
            )
        elif (self.targeted_dropout_type == 'unit'):
            self.kernel.assign(
                self.targeted_unit_dropout(
                    self.kernel,
                    self.target_rate,
                    self.dropout_rate,
                    training
                )
            )

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = tf.linalg.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class VectorDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        kernel_regularizer=None,
        dropout=None
    ):
        super(VectorDense, self).__init__()
        self.units = units
        self.dropout = dropout

        self.permute_layer = tf.keras.layers.Permute(
            dims=(2, 1)
        )

        if self.dropout is not None and self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(
                rate=float(self.dropout)
            )

        self.dense_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

    def call(self, inputs, training):
        net = self.permute_layer(inputs)
        if self.dropout is not None and self.dropout > 0:
            net = self.dropout_layer(net, training=training)
        net = self.dense_layer(net)
        outputs = self.permute_layer(net)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = tf.TensorShape(input_shape).as_list()
        output_shape[1] = self.units
        return tf.TensorShape(output_shape)
