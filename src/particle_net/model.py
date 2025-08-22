import keras
import tensorflow as tf


# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    reduce_sum = ReduceSumLayer()
    mat_mul = MatMulLayer()
    transpose = TransposeLayer()

    with tf.name_scope("dmat"):
        r_A = reduce_sum(A * A, axis=2, keepdims=True)
        r_B = reduce_sum(B * B, axis=2, keepdims=True)
        m = mat_mul(A, transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + transpose(r_B, perm=(0, 2, 1))
        return D


def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    shape = ShapeLayer()
    tile = TileLayer()
    reshape = ReshapeLayer()
    rangel = RangeLayer()
    expand_dims = ExpandDimsLayer()
    concat = ConcatLayer()
    gather_nd = GatherNDLayer()

    with tf.name_scope("knn"):
        queries_shape = shape(features)
        batch_size = queries_shape[0]
        rangel_result = rangel(batch_size)
        reshape_result = reshape(rangel_result, shape=[-1, 1, 1, 1])
        batch_indices = tile(reshape_result, multiples=[1, num_points, k, 1])
        indices = concat(
            [batch_indices, expand_dims(topk_indices, axis=3)], axis=3
        )  # (N, P, K, 2)
        return gather_nd(features, indices)


def edge_conv(
    points,
    features,
    num_points,
    K,
    channels,
    with_bn=True,
    activation="relu",
    pooling="average",
    name="edgeconv",
):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    tile = TileLayer()
    expand_dims = ExpandDimsLayer()
    concat = ConcatLayer()
    subtract = SubtractLayer()
    reduce_max = ReduceMaxLayer()
    reduce_mean = ReduceMeanLayer()
    squeeze = SqueezeLayer()

    with tf.name_scope("edgeconv"):
        # distance
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
        top_k = TopKLayer()
        _, indices = top_k(-D, k=K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)

        fts = features
        knn_fts = knn(num_points, K, indices, fts)  # (N, P, K, C)
        knn_fts_center = tile(
            expand_dims(fts, axis=2), multiples=[1, 1, K, 1]
        )  # (N, P, K, C)
        knn_fts = concat(
            [knn_fts_center, subtract(knn_fts, knn_fts_center)], axis=-1
        )  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(
                channel,
                kernel_size=(1, 1),
                strides=1,
                data_format="channels_last",
                use_bias=False if with_bn else True,
                kernel_initializer="glorot_normal",
                name="%s_conv%d" % (name, idx),
            )(x)
            if with_bn:
                x = keras.layers.BatchNormalization(
                    name="%s_bn%d" % (name, idx), momentum=0.7
                )(x)
            if activation:
                x = keras.layers.Activation(activation, name="%s_act%d" % (name, idx))(
                    x
                )

        if pooling == "max":
            fts = reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut
        sc = keras.layers.Conv2D(
            channels[-1],
            kernel_size=(1, 1),
            strides=1,
            data_format="channels_last",
            use_bias=False if with_bn else True,
            kernel_initializer="glorot_normal",
            name="%s_sc_conv" % name,
        )(expand_dims(features, axis=2))
        if with_bn:
            sc = keras.layers.BatchNormalization(name="%s_sc_bn" % name, momentum=0.7)(
                sc
            )
        sc = squeeze(sc, axis=2)

        if activation:
            return keras.layers.Activation(activation, name="%s_sc_act" % name)(
                sc + fts
            )  # (N, P, C')
        else:
            return sc + fts


def _particle_net_base(points, features=None, setting=None, name="particle_net"):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    zeros_like = ZerosLikeLayer()
    reduce_max = ReduceMaxLayer()

    with tf.name_scope(name):
        if features is None:
            features = points

        expand_dims = ExpandDimsLayer()
        squeeze = SqueezeLayer()
        add = AddLayer()

        expanded_features = expand_dims(features, axis=2)
        bn_layer = keras.layers.BatchNormalization(name=f"{name}_fts_bn", momentum=0.7)
        normalized_features = bn_layer(expanded_features)
        fts = squeeze(normalized_features, axis=2)

        # NOTE: EdgeConv Blocks
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
            if layer_idx == 0:
                pts = add(zeros_like(points), points)
            else:
                pts = add(zeros_like(fts), fts)
            fts = edge_conv(
                pts,
                fts,
                setting.num_points,
                K,
                channels,
                with_bn=True,
                activation="relu",
                pooling=setting.conv_pooling,
                name="%s_%s%d" % (name, "EdgeConv", layer_idx),
            )

        # NOTE: Global Average Pooling
        pool = reduce_max(fts, axis=1)  # (N, C)

        if setting.fc_params is not None:
            x = pool
            for layer_idx, layer_param in enumerate(setting.fc_params):
                units, drop_rate = layer_param
                x = keras.layers.Dense(units, activation="relu")(x)
                if drop_rate is not None and drop_rate > 0:
                    x = keras.layers.Dropout(drop_rate)(x)
            out = keras.layers.Dense(setting.num_class, activation="softmax")(x)
            return out  # (N, num_classes)
        else:
            return pool


class _DotDict:
    pass


def get_particle_net(num_classes, input_shapes):
    r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (18, (64, 64, 64)),
        (18, (224, 224, 224)),
        (18, (384, 384, 384)),
    ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = "average"
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(125, 0.1)]
    setting.num_points = input_shapes["points"][0]

    points = keras.Input(name="points", shape=input_shapes["points"])
    features = (
        keras.Input(name="features", shape=input_shapes["features"])
        if "features" in input_shapes
        else None
    )
    outputs = _particle_net_base(points, features, setting, name="ParticleNet")

    return keras.Model(inputs=[points, features], outputs=outputs, name="ParticleNet")


class ExpandDimsLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.expand_dims(x, **kwargs)


class SqueezeLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.squeeze(x, **kwargs)


class ZerosLikeLayer(keras.Layer):
    def call(self, x):
        return tf.zeros_like(x, dtype="float32")


class AddLayer(keras.Layer):
    def call(self, x, y):
        return tf.add(x, y)


class ReduceSumLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.reduce_sum(x, **kwargs)


class MatMulLayer(keras.Layer):
    def call(self, a, b):
        return tf.matmul(a, b)


class TransposeLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.transpose(x, **kwargs)


class TopKLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.nn.top_k(x, **kwargs)


class ShapeLayer(keras.Layer):
    def call(self, x):
        return tf.shape(x)


class TileLayer(keras.Layer):
    def call(self, input, **kwargs):
        return tf.tile(input, **kwargs)


class ReshapeLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.reshape(x, **kwargs)


class RangeLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.range(x, **kwargs)


class ConcatLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.concat(x, **kwargs)


class GatherNDLayer(keras.Layer):
    def call(self, x, y):
        return tf.gather_nd(x, y)


class SubtractLayer(keras.Layer):
    def call(self, x, y):
        return tf.subtract(x, y)


class ReduceMeanLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.reduce_mean(x, **kwargs)


class ReduceMaxLayer(keras.Layer):
    def call(self, x, **kwargs):
        return tf.reduce_max(x, **kwargs)
