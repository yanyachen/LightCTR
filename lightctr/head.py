import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_util


def binary_classification_estimator_spec(
    mode,
    labels,
    logits,
    optimizers
):
    # Prediction
    logits = tf.reshape(logits, (-1,))
    probabilities = tf.nn.sigmoid(logits, name='Sigmoid')

    # Mode: PREDICT
    serving_signature = (
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    )
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            serving_signature: tf.estimator.export.PredictOutput(probabilities)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=probabilities,
            export_outputs=export_outputs
        )

    # Loss
    labels = tf.cast(labels, dtype=probabilities.dtype)
    loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )
    loss = tf.compat.v1.losses.compute_weighted_loss(
        losses=loss_vec,
        weights=1.0,
        reduction=tf.compat.v1.losses.Reduction.SUM
    )

    # Mode: EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        auc = tf.compat.v1.metrics.auc(
            labels=labels,
            predictions=probabilities,
            num_thresholds=10000,
            curve='ROC',
            name='auc',
            summation_method='careful_interpolation'
        )
        average_loss = tf.compat.v1.metrics.mean(
            loss_vec,
            weights=array_ops.ones_like(loss_vec),
            name='average_loss'
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={
                'auc': auc,
                'average_loss': average_loss
            }
        )

    # Mode: TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
        # train_op
        train_ops = []
        for scope, optimizer in optimizers.items():
            train_ops.append(
                optimizer.minimize(
                    loss=loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=scope
                    )
                )
            )
        train_op = control_flow_ops.group(*train_ops)
        # update_op
        update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
        if update_ops:
            train_op = control_flow_ops.group(train_op, *update_ops)
        # global_step
        with ops.control_dependencies([train_op]):
            train_op = state_ops.assign_add(
                training_util.get_global_step(), 1
            ).op
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )
