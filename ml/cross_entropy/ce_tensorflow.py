import tensorflow as tf

# logits = tf.random.normal(shape=[2, 3], dtype=tf.float32)
logits = tf.constant(
    [[-1.4710, 0.5296, 0.3145], [1.6896, -0.5447, 0.4537]], dtype=tf.float32
)
sparse_labels = tf.constant([2, 0])
raw_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=sparse_labels, logits=logits
)

labels = tf.one_hot(
    indices=sparse_labels,
    depth=logits.shape[1],
    dtype=tf.int32,  # Output data type (can be float32, int32, etc.)
)
raw_ce_ = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

tf.debugging.assert_near(raw_ce, raw_ce_)
print(raw_ce)
