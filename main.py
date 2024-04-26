from check_performance import get_average
from model.dqn_agent import DQNAgent
from game_env import GameEnv
import time
import numpy as np
import tensorflow as tf
from model.logger import Logger

train_episodes = 10000


def encode_board(board):
    return np.log2(np.where(board == 0, 1, board)) / 16


def train():
    episodes = train_episodes
    env = GameEnv()
    # logs = []
    logger = Logger()
    agent = DQNAgent(logger=logger)

    state, info = env.reset()
    tf.random.set_seed(1234)

    while True:
        # use random select fill up replay buffer first
        action = agent.act(
            encode_board(state), random=True, valid_action=env.get_valid_move()
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.store_experience(
            encode_board(state), action, reward, encode_board(next_state), terminated
        )

        state = next_state.copy()

        if agent.replay_buffer.full:
            break

        if terminated or truncated:
            state, info = env.reset()

    render_step = 50
    for i in range(episodes):
        state, info = env.reset()
        # env.render()
        if i > 0 and i % render_step == 0:
            env.render()

        # start_time = time.time()
        while True:
            board_encoded = encode_board(state)

            action = agent.act(board_encoded, valid_action=env.get_valid_move())

            next_state, reward, terminated, truncated, info = env.step(action)
            next_board_encoded = encode_board(next_state)

            agent.store_experience(
                board_encoded,
                action,
                reward,
                next_board_encoded,
                terminated,
            )
            state = next_state.copy()

            loss = agent.update()

            # env.render()
            if i > 0 and i % render_step == 0:
                env.render()

            if terminated or truncated:
                avg_loss = tf.reduce_mean(loss).numpy() if loss else 0
                logger.log_episode(env.score, env.reward, env.move, env.max_tile, i + 1)
                logger.log_scalar("Loss", avg_loss, i + 1)
                if (i + 1) % 100 == 0:
                    logger.log_weights(agent.q_network, agent.target_q_network, i + 1)
                agent.set_decay_epsilon(i, train_episodes)
                break


train()
