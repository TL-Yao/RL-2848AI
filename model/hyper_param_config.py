replay_buffer_size = 10000
batch_size = 128
gamma = 0.99
learning_rate = 0.000001
epsilon_start = 0.15
epsilon_end = 0.01
tau = 0.1
target_update_freq = 200
clip_norm = 0.025
regularization_factor = 0.1


# reward config
empty_weight = 0
merge_weight = 1
# max_tile_weight = 0.1
monotonic_weight = 0.0
# corner_weight = 0.1

# empty_weight = 0
# merge_weight = 0.3
# # max_tile_weight = 0.1
# monotonic_weight = 0.7
# corner_weight = 0

# PER config
per_alpha = 0.6
beta_start = 0.4
