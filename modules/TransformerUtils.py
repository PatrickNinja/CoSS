import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import dot, Activation

class TransformerUtils:

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, d_model):
        angle_rads = TransformerUtils.get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :], d_model)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
    
        return tf.cast(pos_encoding, dtype=tf.float32)

    @staticmethod
    def create_padding_mask(seqs):
        seqs = tf.cast(tf.math.equal(seqs, 0), tf.float32)
        return seqs[:, tf.newaxis, tf.newaxis, :], seqs  # (batch_size, 1, 1, seq_len), (batch_size, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    @staticmethod
    def loss_function(real, pred):
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_mean(loss_)

    @staticmethod
    def create_masks(sbt_inp, node_inp, tar):
        look_ahead_mask = TransformerUtils.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask, _ = TransformerUtils.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return sbt_padding_mask, node_padding_mask, combined_mask

    @staticmethod
    def self_attention(x, y, mask):
        attn = dot([x, y], axes=[2,2])
        if mask is not None:
            attn += (mask * -1e9)
        attn = Activation('softmax')(attn)
        return dot([attn, x], axes=[2,1])