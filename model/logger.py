import os
import tensorflow as tf
from datetime import datetime


class Logger:
    def __init__(self):
        log_dir = "logs/dqn/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, timestamp)
        print(f"version: {self.log_path}")
        self.writer = tf.summary.create_file_writer(self.log_path)

    def log_scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def log_histogram(self, tag, values, step):
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()

    def log_graph(self, func):
        with self.writer.as_default():
            tf.summary.graph(func.get_concrete_function().graph)
            self.writer.flush()

    def log_episode(self, score, reward, steps, max_tile, episode):
        with self.writer.as_default():
            tf.summary.scalar("Episode Score", score, step=episode)
            tf.summary.scalar("Episode Reward", reward, step=episode)
            tf.summary.scalar("Episode Steps", steps, step=episode)
            tf.summary.scalar("Episode Max Tile", max_tile, step=episode)
            self.writer.flush()

    def log_weights(self, q_network, target_network, episode):
        with self.writer.as_default():
            for layer in q_network.layers:
                if "conv" in layer.name or "dense" in layer.name:
                    tf.summary.histogram(
                        f"Q/{layer.name}/weights", layer.weights[0], step=episode
                    )
                    tf.summary.histogram(
                        f"Q/{layer.name}/bias", layer.weights[1], step=episode
                    )
            for layer in target_network.layers:
                if "conv" in layer.name or "dense" in layer.name:
                    tf.summary.histogram(
                        f"Target/{layer.name}/weights", layer.weights[0], step=episode
                    )
                    tf.summary.histogram(
                        f"Target/{layer.name}/bias", layer.weights[1], step=episode
                    )
            self.writer.flush()
