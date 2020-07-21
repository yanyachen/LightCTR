import tensorflow as tf


def categorical_to_embedding(
    categorical_columns,
    embedding_size,
    **kwargs
):
    embedding_columns = [
        tf.feature_column.embedding_column(
            categorical_column=categorical_column,
            dimension=embedding_size,
            **kwargs
        )
        for categorical_column in categorical_columns
    ]
    return embedding_columns


def pairwise_feature(inputs, field_size, embedding_size):
    embeddings = tf.reshape(
        inputs,
        shape=(-1, field_size, embedding_size)
    )

    index_i = []
    index_j = []
    for i in range(0, field_size):
        for j in range(i+1, field_size):
            index_i.append(i)
            index_j.append(j)

    feature_i = tf.gather(embeddings, index_i, axis=1)
    feature_j = tf.gather(embeddings, index_j, axis=1)
    return feature_i, feature_j
