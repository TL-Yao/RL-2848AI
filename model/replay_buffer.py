from model.hyper_param_config import replay_buffer_size
from typing import SupportsFloat
import numpy as np
import tensorflow as tf


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.priority_tree = np.zeros(2 * capacity - 1)
        self.exp_data = np.zeros(capacity, dtype=object)  # leaf in tree
        self.exp_data_ptr = 0

    def _propagate(self, idx, change):
        """
        update priority of all parent nodes
        """
        parent = (idx - 1) // 2
        self.priority_tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, priority):
        """
        retrieve experience at leaf by given priority
        """
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.priority_tree):
            return idx
        if priority <= self.priority_tree[left]:
            return self._retrieve(left, priority)
        else:
            return self._retrieve(right, priority - self.priority_tree[left])

    def total(self):
        """
        total priority in tree (root node)
        """
        return self.priority_tree[0]

    def add(self, priority, data):
        """
        add new experience and put in right place by priority
        """
        idx = self.exp_data_ptr + self.capacity - 1
        self.exp_data[self.exp_data_ptr] = data
        self.update(idx, priority)
        self.exp_data_ptr += 1
        if self.exp_data_ptr >= self.capacity:  # reset ptr
            self.exp_data_ptr = 0

    def update(self, idx, priority):
        """
        update priority of given idx experience in tree and update parent node
        """
        change = priority - self.priority_tree[idx]
        self.priority_tree[idx] = priority
        self._propagate(idx, change)

    def get(self, priority):
        """
        retrieve experience and its priority by given priority
        """
        tree_idx = self._retrieve(0, priority)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.priority_tree[tree_idx], self.exp_data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, beta_start, total_episodes):
        self.capacity = capacity
        # alpha is used to define the importance of TD in priority, if alpha = 0, PER degenerate to normal sampling
        self.alpha = alpha
        # beta is used to control bias cause by prioritized sample, prevent model from overfit
        self.beta_start = beta_start
        self.total_episodes = total_episodes
        self.frame = 1
        self.tree = SumTree(capacity)

    def beta_by_episode(self, episode):
        """ """
        return min(
            1.0,
            self.beta_start + episode * (1.0 - self.beta_start) / self.total_episodes,
        )

    def push(self, state, action, reward, next_state, done):

        max_priority = np.max(self.tree.priority_tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        self.beta = self.beta_by_frame(self.frame)
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        self.frame += 1
        return batch, idxs, is_weight

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority**self.alpha)


class ReplayBuffer:
    def __init__(self, state_shape):
        self.capacity = replay_buffer_size
        self.states = np.zeros((replay_buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((replay_buffer_size,), dtype=np.int32)
        self.rewards = np.zeros((replay_buffer_size,), dtype=np.float32)
        self.next_states = np.zeros(
            (replay_buffer_size, *state_shape), dtype=np.float32
        )
        self.terminated_factors = np.zeros((replay_buffer_size,), dtype=np.float32)
        self.position = 0
        self.full = False

    def push(
        self,
        state: np.ndarray,
        action,
        reward: SupportsFloat,
        next_state: np.ndarray,
        terminated_factor,
    ):
        # save experience into buffer
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.terminated_factors[idx] = terminated_factor
        self.position = (idx + 1) % self.capacity
        self.full = self.full or self.position == 0

    def sample(self, batch_size):
        # return random.sample(self.buffer, batch_size)
        indices = np.random.choice(self.capacity, batch_size, replace=False)
        batch_states = tf.convert_to_tensor(self.states[indices])
        batch_actions = tf.convert_to_tensor(self.actions[indices])
        batch_rewards = tf.convert_to_tensor(self.rewards[indices])
        batch_next_states = tf.convert_to_tensor(self.next_states[indices])
        batch_terminated_factors = tf.convert_to_tensor(
            self.terminated_factors[indices]
        )
        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_terminated_factors,
        )
