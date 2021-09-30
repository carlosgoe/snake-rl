from collections import deque
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
from agent import Agent


class DQN(Agent):

    def __init__(self, layers, loss_fn, optimizer, discount_factor, buffer_size, file=None):
        # Call agent class initializer
        super(DQN, self).__init__(layers[:-1] + [(layers[-1], 'linear')], loss_fn, optimizer, discount_factor, file)
        if file is None:
            # Create input and hidden layers using the functional API
            input_hidden = [keras.layers.Input(shape=[layers[0]])]
            for l in layers[1:-1]:
                input_hidden.append(keras.layers.Dense(l[0], activation=l[1])(input_hidden[-1]))
            # Append separate layers for state values and action advantages to last hidden layer 
            state_values = keras.layers.Dense(1)(input_hidden[-1])
            raw_advantages = keras.layers.Dense(layers[-1])(input_hidden[-1])
            # Subtract highest advantage from advantages to make it 0
            advantages = raw_advantages - keras.backend.max(raw_advantages, axis=1, keepdims=True)
            # Overwrite model and set its outputs to calculated Q-values (V(s) + A(s, a))
            self.model = keras.Model(inputs=[input_hidden[0]], outputs=[state_values + advantages])
        # Create target model by cloning online model and copying its weights
        self.target = keras.models.clone_model(self.model)
        self.update_target_model()
        # Create a replay buffer to store buffer_size experiences
        self.replay_buffer = deque(maxlen=buffer_size)

    def play_one_step(self, state, epsilon, invalid=[], vector=False):
        # Choose random action with probability epsilon
        if np.random.rand() < epsilon:
            return random.choice(list(set(range(self.n_outputs)) - set(invalid)))
        # Else: choose action with largest predicted Q-value (Q-values of invalid actions are set to -inf)
        q_values = self.model.predict(state[np.newaxis])[0]
        q_values[invalid] = -np.inf
        if vector:
            return q_values
        return np.argmax(q_values)

    # Append current state, action, reward, next state, and done to replay buffer
    def add_experience(self, state, action, reward, next_state, done, invalid=[]):
        self.replay_buffer.append((state, action, reward, next_state, done, invalid))

    # Set target weights to online weights
    def update_target_model(self):
        self.target.set_weights(self.model.get_weights())

    def training_step(self, batch_size):
        # Sample random batch of size batch_size from replay buffer
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones, invalids = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)]
        # Let online model predict best actions for each next state (invalid actions are set to -inf)
        next_q_values = self.model.predict(next_states)
        next_q_values[[i for i, r in enumerate(invalids) for k in range(len(r))], np.concatenate(invalids).astype(np.uint8)] = -np.inf
        best_next_actions = np.argmax(next_q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
        # Let target model determine their Q-values
        next_best_q_values = np.sum(self.target.predict(next_states) * next_mask, axis=1)
        # Calculate target Q-values (current reward + discounted sum of future Q-values)
        target_q_values = (rewards + (1 - dones) * self.discount_factor * next_best_q_values)
        # Create mask to multiply with predicted Q-values (actions that weren't chosen will result in 0)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            # Predict Q-values for each experienced state
            all_Q_values = self.model(states)
            q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            # Calculate loss of predicted and target Q-values
            loss = tf.reduce_mean(self.loss_fn(target_q_values, q_values))
            # Apply corresponding gradients
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
