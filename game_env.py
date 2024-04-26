from __future__ import annotations

import math
from typing import SupportsFloat, Any
import gymnasium as gym
import numpy
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from render_game import Renderer
from time import sleep
from model.hyper_param_config import (
    empty_weight,
    merge_weight,
    # max_tile_weight,
    monotonic_weight,
    # corner_weight,
)


class GameEnv(gym.Env):
    action_space = None
    observation_space = None
    seed = None
    np_random = None
    score = None
    size = None
    max_move = None
    move = None
    render_mode = None
    renderer = None
    terminated = None
    max_tile = None
    new_max = False

    def __init__(
        self,
        size=4,
        max_move=10000,
        render_mode="human",
    ):
        self.size = size

        self.action_space = Discrete(4)  # 4 actions: up(0) down(1) left(2) right(3)
        # observation_space is more like a statement or definition about size and boundary of status
        # a 4x4 board, each cell contain value 0 - 1
        # calculate from log_2(x)/16 (in a 4x4 board, theoretical maximum is 2^16)
        self.observation_space = Box(
            low=0, high=131072, shape=(self.size, self.size), dtype=np.int32
        )

        # initial a random seed，for given seed the experiment is reproducible
        self._seed()

        # set max move
        self.max_move = max_move

        # 4x4 game board
        self.board: numpy.ndarray = np.zeros((self.size, self.size), np.int32)

        # initialize Render object
        self.renderer = Renderer(self.size)

        # initial parameters
        self.score = 0
        self.move = 0
        self.render_mode = render_mode
        self.terminated = False
        self.highest_score = 0
        self.largest_tile = 0
        self.max_tile = 2
        self.reward = 0

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0
        reward_info = {}
        self.new_max = False

        # according to the action, move tiles on board
        merged_score, illegal = self._move(action)

        if not illegal:
            self.move += 1

            self.score += merged_score

            reward, reward_info = self._get_reward(merged_score, action)

            self.reward += reward

            if reward > 1:
                print(self.board)
                print(action)
                print(reward_info)

            # add new random tile
            self._add_random_tile()

        # check if game end
        terminated = self._check_end()
        self.terminated = terminated

        # construct info
        info = {
            "move": self.move,
            "score": self.score,
            "step_reward": reward,
            "is_illegal_move": illegal,
            "reward_info": reward_info,
        }

        return self.board.copy(), reward, self.terminated, False, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.board[:] = 0
        self.move = 0
        self.score = 0
        self.terminated = False
        self.max_tile = 2
        self.reward = 0

        self._add_random_tile()
        self._add_random_tile()

        return self.board.copy(), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.renderer.render(
            self.board,
            self.score,
            self.move,
            self.terminated,
        )
        sleep(0.02)
        return None

    def get_valid_move(self) -> list:
        valid_move = []

        for action in range(4):
            _, illegal = self._move(action, check_mode=True)
            if not illegal:
                valid_move.append(action)

        return valid_move

    def _move(self, action: int, check_mode=False) -> (float, int, bool):
        illegal = True
        merge_score = 0
        is_horizontal_move = action in [2, 3]

        for i in range(self.size):
            # take row if horizon move, column otherwise
            origin_line = self.board[i, :] if is_horizontal_move else self.board[:, i]

            if action in [1, 3]:  # reverse item in the line if swipe down or right
                origin_line = origin_line[::-1]

            combined_line, _merge_score = self._combine(origin_line)

            merge_score += _merge_score

            if not np.array_equal(origin_line, combined_line):
                # consider legal move if any shifted or combined happened in the row/column
                illegal = False

            if not check_mode:  # check mode is used to check if there is legal action

                if action in [1, 3]:  # reverse item in the line back to origin order
                    combined_line = combined_line[::-1]

                if is_horizontal_move:
                    self.board[i, :] = combined_line
                else:
                    self.board[:, i] = combined_line

        return merge_score, illegal

    def _combine(self, tiles: np.ndarray) -> (np.ndarray, int):
        merge_score = 0

        # remove all 0 (shift tiles)
        shifted_tiles = tiles[tiles != 0]

        # combine tiles
        combined = False
        for i in range(1, shifted_tiles.shape[0]):
            if combined:
                combined = False
                continue

            if i != 0 and shifted_tiles[i] == shifted_tiles[i - 1]:
                shifted_tiles[i - 1] += shifted_tiles[i]
                shifted_tiles[i] = 0
                merge_score += shifted_tiles[i - 1]
                combined = True
                if shifted_tiles[i - 1] > self.max_tile:
                    self.max_tile = shifted_tiles[i - 1]
                    self.new_max = True

        # remove combined tiles
        merged_tiles = shifted_tiles[shifted_tiles != 0]

        # pad the array with zeros
        merged_tiles = np.pad(
            merged_tiles,
            (0, self.size - merged_tiles.shape[0]),
            "constant",
            constant_values=0,
        )

        return merged_tiles, merge_score

    def _seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)

    def _add_random_tile(self):
        # 90% select 2, 10% select 4
        selected_value = self.np_random.choice(a=[2, 4], size=1, p=[0.9, 0.1])[0]
        empties = self._get_empties()
        selected_empty = self.np_random.choice(a=empties, size=1)[0]
        self.board[selected_empty[0]][selected_empty[1]] = selected_value

    def _get_empties(self) -> np.ndarray:
        # argwhere return list of coordinates of non-zero item
        # self.board == 0 turn board in into [[ True,  True,  False,  True],...]
        return np.argwhere(self.board == 0)

    def _check_end(self) -> bool:
        #  truncated indicates that the game ends due to reasons other than the board being filled up.
        # truncated = False
        # game terminated because the board being filled up
        # game_terminated = True

        # if self.move >= self.max_move or self.illegal_move > self.max_illegal_move:
        #     truncated = True
        #
        # for direction in range(4):
        #     _, illegal = self._move(direction, check_mode=True)
        #
        #     if not illegal:
        #         game_terminated = False

        return not self.get_valid_move()

    def _get_reward(self, merged_score, action) -> (float, dict):
        def merge_reward():
            if merged_score == 0:
                return 0

            return math.log2(merged_score) / 16

        # def empty_reward():
        #     return self._get_empties().shape[0] / 20
        #
        # # def max_tile_reward():
        # #     if self.new_max and self.max_tile > 128:
        # #         return 1.0
        # #
        # #     return 0.0
        #
        # def early_move_action_reward():
        #     # if self.move < 200:
        #     #     if action in [1, 2]:
        #     #         return 0.5
        #
        #     if action in [1, 2]:
        #         return 0.1
        #     elif action == 3:
        #         return -0.1
        #     elif action == 0:
        #         return -0.2
        #
        #     return 0.0
        #
        # def monotonic_reward():
        #     def check_edge_monotonicity(edge):
        #         # Remove leading zeros and ensure the edge is a flat array
        #         edge = np.trim_zeros(edge.flatten(), "fb")
        #         if edge.shape[0] <= 1:
        #             return 0
        #
        #         is_increasing = np.all(edge[1:] >= np.roll(edge, 1)[1:])
        #         is_decreasing = np.all(edge[1:] <= np.roll(edge, 1)[1:])
        #         if is_increasing or is_decreasing:
        #             max_value = np.max(edge)
        #             return math.log2(max_value) / 16 if max_value > 1 else 0
        #
        #         return int(is_increasing or is_decreasing)
        #
        #     monotonicity_score = 0
        #
        #     # check rows
        #     for row in self.board:
        #         monotonicity_score += check_edge_monotonicity(row)
        #     # check columns
        #     for col in self.board.T:
        #         monotonicity_score += check_edge_monotonicity(col)
        #
        #     return monotonicity_score * 0.25
        #
        # def corner_reward():
        #     corner_values = [
        #         self.board[-1, 0],
        #     ]
        #     max_corner_value = np.max(corner_values)
        #
        #     if max_corner_value == self.max_tile:
        #         return math.log2(max_corner_value) / 20
        #
        #     return 0

        # _empty_reward = empty_reward()
        _merge_reward = merge_reward()
        # # _max_tile_reward = max_tile_reward()
        # _early_move_action_reward = early_move_action_reward()
        # # _monotonic_reward = monotonic_reward()
        # _corner_reward = corner_reward()

        reward = (
            # empty_weight * _empty_reward
            # +
            merge_weight
            * _merge_reward
            # + max_tile_weight * _max_tile_reward
            # + _early_move_action_reward
            # + monotonic_weight * _monotonic_reward
            # + corner_weight * _corner_reward
        )

        info = {
            # "empty_reward": empty_weight * _empty_reward,
            "merge_reward": merge_weight * _merge_reward,
            # "max_tile_reward": _max_tile_reward,
            # "early_move_action_reward": _early_move_action_reward,
            # "monotonic_reward": monotonic_weight * _monotonic_reward,
        }

        if reward > 1:
            print(f"warning: reward too large, {info} ")

        return reward, info
