"""TensorFlow policy/value model and action selection helpers."""

from __future__ import annotations

import os
import importlib.util
from typing import Iterable, Sequence, Tuple

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
if (
    importlib.util.find_spec("tf_agents") is not None
    and importlib.util.find_spec("tf_keras") is not None
):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf


class PolicyValueNet(tf.keras.Model):
    """Small MLP with policy logits and scalar value heads."""

    def __init__(
        self,
        input_dim: int = 97,
        num_actions: int = 96,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.hidden_layers = [
            tf.keras.layers.Dense(size, activation="relu") for size in self.hidden_sizes
        ]
        self.policy_head = tf.keras.layers.Dense(num_actions)
        self.value_head = tf.keras.layers.Dense(1)

    def call(self, inputs, training: bool = False):
        if isinstance(inputs, dict):
            x = inputs["observation"]
        else:
            x = inputs
        x = tf.cast(x, tf.float32)
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        logits = self.policy_head(x, training=training)
        value = self.value_head(x, training=training)
        return logits, value

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "num_actions": self.num_actions,
            "hidden_sizes": self.hidden_sizes,
        }


def mask_logits(logits: tf.Tensor, action_mask: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(action_mask, tf.bool)
    very_negative = tf.constant(-1.0e9, dtype=logits.dtype)
    return tf.where(mask, logits, very_negative)


def action_log_probs(masked_logits: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    log_probs = tf.nn.log_softmax(masked_logits, axis=-1)
    indices = tf.stack(
        [tf.range(tf.shape(actions)[0], dtype=tf.int32), tf.cast(actions, tf.int32)],
        axis=1,
    )
    return tf.gather_nd(log_probs, indices)


def categorical_entropy(masked_logits: tf.Tensor) -> tf.Tensor:
    log_probs = tf.nn.log_softmax(masked_logits, axis=-1)
    probs = tf.exp(log_probs)
    return -tf.reduce_sum(probs * log_probs, axis=-1)


def select_action(
    model: PolicyValueNet,
    obs: np.ndarray,
    action_mask: np.ndarray,
    device: str = "/CPU:0",
    deterministic: bool = False,
) -> Tuple[int, float, float]:
    """Select an action from the masked policy.

    Returns:
        action, log_prob, value
    """
    if not np.any(action_mask):
        raise ValueError("Cannot select an action when no legal actions remain.")

    with tf.device(device):
        obs_tensor = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor(action_mask[None, :], dtype=tf.bool)
        logits, value = model(obs_tensor, training=False)
        masked = mask_logits(logits, mask_tensor)
        if deterministic:
            action_tensor = tf.argmax(masked, axis=-1, output_type=tf.int32)
        else:
            action_tensor = tf.random.categorical(masked, num_samples=1)[:, 0]
        log_prob = action_log_probs(masked, action_tensor)

    return (
        int(action_tensor.numpy()[0]),
        float(log_prob.numpy()[0]),
        float(value.numpy()[0, 0]),
    )
