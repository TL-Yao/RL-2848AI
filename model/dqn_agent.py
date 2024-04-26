import time
from typing import SupportsFloat

from model.cnn_network import CNNNetwork
from model.replay_buffer import ReplayBuffer
from model.hyper_param_config import (
    epsilon_start,
    epsilon_end,
    learning_rate,
    gamma,
    batch_size,
    tau,
    target_update_freq,
    clip_norm,
    regularization_factor,
)
import numpy as np
import tensorflow as tf


class DQNAgent:
    def __init__(
        self,
        _epsilon_start=epsilon_start,
        _epsilon_end=epsilon_end,
        _learning_rate=learning_rate,
        _gamma=gamma,
        _batch_size=batch_size,
        _tau=tau,
        _target_update_freq=target_update_freq,
        logger=None,
    ):
        self.grid_size = 4
        self.action_size = 4

        # init Q network and target network
        self.q_network = CNNNetwork(
            self.grid_size, self.action_size, _learning_rate, regularization_factor
        )
        self.target_q_network = CNNNetwork(
            self.grid_size, self.action_size, _learning_rate, regularization_factor
        )
        self.q_network.build((None, 1, self.grid_size, self.action_size))
        self.target_q_network.build((None, 1, self.grid_size, self.action_size))
        self.target_q_network.set_weights(self.q_network.get_weights())

        self.replay_buffer = ReplayBuffer((1, self.grid_size, self.grid_size))
        self.init_epsilon = _epsilon_start
        self.end_epsilon = _epsilon_end
        self.epsilon = _epsilon_start
        self.gamma = _gamma
        self.batch_size = _batch_size
        self.tau = _tau
        self.target_update_freq = _target_update_freq
        self.target_update_count = 0
        self.logger = logger
        self.log_graph = True

    def act(self, state: np.ndarray, random=False, valid_action=None) -> int:
        # epsilon greedy policy, epsilon% chance random select action without using Q network.
        random |= np.random.choice(
            a=[False, True], size=1, p=[1 - self.epsilon, self.epsilon]
        )[0]

        if random:
            return np.random.choice(
                a=valid_action, size=1, p=[1 / len(valid_action)] * len(valid_action)
            )[0]
        else:
            # predict Q(s, a) for each a
            q_sa = self.q_network.call(state.reshape(1, 1, 4, 4))
            sorted_actions = np.argsort(q_sa)[0][
                ::-1
            ]  # sorting by Q value from high to low
            for action in sorted_actions:
                if action not in valid_action:
                    continue
                else:
                    return action

        print("Couldn't find valid action, but the game is not terminated.")
        print(f"State: {state}")
        return np.random.choice(
            a=self.action_size, size=1, p=[1 / self.action_size] * self.action_size
        )[0]

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: SupportsFloat,
        next_state: np.ndarray,
        terminated: bool,
    ):
        state_tensor = tf.reshape(
            tf.convert_to_tensor(state, dtype=tf.float32), (1, 4, 4)
        )
        next_state_tensor = tf.reshape(
            tf.convert_to_tensor(next_state, dtype=tf.float32), (1, 4, 4)
        )
        action_tensor = tf.convert_to_tensor(action, dtype=tf.int32)
        reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)
        terminated_factor_tensor = tf.convert_to_tensor(
            0.0 if terminated else 1.0, dtype=tf.float32
        )

        self.replay_buffer.push(
            state_tensor,
            action_tensor,
            reward_tensor,
            next_state_tensor,
            terminated_factor_tensor,
        )

    def update(self):
        # soft update
        if self.target_update_count == self.target_update_freq:
            self._soft_update()
            self.target_update_count = 0

        self.target_update_count += 1

        # sample experience
        batch_state, batch_action, batch_reward, batch_next_state, batch_terminated = (
            self.replay_buffer.sample(self.batch_size)
        )

        if self.log_graph:
            tf.summary.trace_on(graph=True, profiler=True)

        gradients, loss = self.train(
            batch_state, batch_action, batch_next_state, batch_reward, batch_terminated
        )

        # clipped_gradients = [
        #     tf.clip_by_norm(g, clip_norm) for g in gradients
        # ]  # gradient clipping
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)

        self.q_network.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        if self.log_graph:
            with self.logger.writer.as_default():
                tf.summary.trace_export(
                    name="Training_Computation_Graph",
                    step=0,
                    profiler_outdir=self.logger.log_path,
                )
            self.log_graph = False
        return loss

    # @tf.function
    def train(
        self,
        batch_state,
        batch_action,
        batch_next_state,
        batch_reward,
        batch_terminated,
    ):
        with tf.GradientTape() as tape:
            # predicting Q(s,a) for each state and action in experience by using Q network
            q_states_predict = self.q_network.call(batch_state, training=True)
            q_states_action = tf.gather_nd(
                q_states_predict,
                tf.stack([tf.range(self.batch_size), batch_action], axis=1),
            )

            # predicting max_a(Q(s')) for each next state in experience by using target network
            target_next_states_predict = self.target_q_network.call(
                batch_next_state, training=True
            )

            max_target_next_s_q = tf.reduce_max(
                target_next_states_predict, axis=1
            )  # get max_q of each Q(s', a)

            # computing predict Q(s, a) of target network by using bellman expectation equation
            q_target_next_s = (
                batch_reward + self.gamma * max_target_next_s_q * batch_terminated
            )

            # computing loss
            loss = self.q_network.compiled_loss(q_states_action, q_target_next_s)

        # train q network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        return gradients, loss

    def _soft_update(self):
        # sof update target-network
        q_weights = self.q_network.get_weights()
        target_weights = self.target_q_network.get_weights()

        updated_weights = [
            (self.tau * q_w + (1 - self.tau) * t_w)
            for q_w, t_w in zip(q_weights, target_weights)
        ]
        tf.keras.backend.batch_set_value(
            zip(self.target_q_network.variables, updated_weights)
        )

    def set_decay_epsilon(self, episode, total_episode):
        decay_epsilon = epsilon_start * (1 - episode / total_episode)
        self.epsilon = max(self.end_epsilon, decay_epsilon)
