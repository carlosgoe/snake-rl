import tensorflow as tf
import tensorflow_probability as tfp
from agent import Agent
import numpy as np


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = []
    for rewards in all_rewards:
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        # Determine reward of each step depending on following rewards using discount_rate
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        # Save list of discounted rewards in all_discounted_rewards
        all_discounted_rewards.append(discounted_rewards)
    # Make all_discounted_rewards one dimensional
    flat_rewards = np.concatenate(all_discounted_rewards)
    # Return normalized version of all_discounted_rewards
    return [(discounted_rewards - flat_rewards.mean()) / flat_rewards.std() for discounted_rewards in all_discounted_rewards]


def multinomial(probs, invalid=[]):
    # Remove invalid actions from possible actions
    actions = list(set(range(max(2, probs.shape[1]))) - set(invalid))
    # if output size is 1: select action by generating random float and comparing it to model output
    if probs.shape[1] == 1 and len(actions) == 1:
        return actions[0]
    if probs.shape[1] == 1:
        action = tf.random.uniform([1, 1]) > probs
        return action, tf.constant([[1.]]) - tf.cast(action, tf.float32)
    # Multinomial selection if output size > 0
    probs_c = np.copy(probs)
    if probs.shape[1] > len(actions):
        # Set probability of actions that are not in given list to 0
        probs_c[0, invalid] = 0.
        probs_c = probs_c / np.sum(probs_c)
    one_hot = tf.cast(tfp.distributions.Multinomial(1., probs=probs_c[0]).sample(1), tf.float32)
    return tf.math.argmax(one_hot[0]), one_hot


class PG(Agent):

    def __init__(self, layers, loss_fn, optimizer, discount_factor, file=None):
        # Create model by calling Agent class initializer
        super(PG, self).__init__(layers, loss_fn, optimizer, discount_factor, file)

    def run_policy(self, obs, invalid=[]):
        with tf.GradientTape() as tape:
            # Get model output for observation as input
            pred = self.model(obs[np.newaxis])
            # Select action depending on its output probability
            action, y_target = multinomial(pred, invalid)
            # Calculate loss with one-hot encoded action as target
            loss = tf.reduce_mean(self.loss_fn(y_target, pred))
        # Return action as integer and the gradient that would make it more likely
        return int(np.max(action.numpy().flatten())), tape.gradient(loss, self.model.trainable_variables)

    def apply_grads(self, all_rewards, all_grads, discount=True):
        # Get discounted and normalized rewards
        if discount:
            all_rewards = discount_and_normalize_rewards(all_rewards, self.discount_factor)
        # Calculate weighted mean of gradients by multiplying with rewards
        all_mean_grads = []
        for var_i in range(len(self.model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[ep_i][step][var_i]
                 for ep_i, rewards in enumerate(all_rewards)
                  for step, final_reward in enumerate(rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        # Apply mean gradients to model
        self.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))
