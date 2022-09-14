from ipp_toolkit.config import PLANNING_RESOLUTION
from ipp_toolkit.planners.planners import GridWorldPlanner
import numpy as np


class MCTSPlanner(GridWorldPlanner):
    def __init__(
        self,
        grid_start,
        grid_end,
        grid_resolution=PLANNING_RESOLUTION,
        epsilon=0.001,
        n_iters=500,
        gamma=0.9,
        K=1,
    ):
        super().__init__(grid_start, grid_end, grid_resolution)

        self.epsilon = epsilon
        self.n_iters = n_iters
        self.gamma = gamma
        self.K = K

        self.planning_steps = np.array(
            [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        )
        # This is the value function on state-action pairs
        # Actions are on an 8-connected grid
        self.Q = np.zeros((self.initial_size[0], self.initial_size[1], 8))

    def _step(self, s, a):
        s_prime = None  # TODO get this
        r_prime = None  # TODO get this
        return s_prime, r_prime

    def _rollout(self, s, depth):
        if np.power(self.gamma, depth) < self.epsilon:
            return 0

        a = None  # Sample a from Omega
        s_prime, r_prime = self._step(s, a)

        return r_prime + self.gamma * self._rollout(s_prime, depth + 1)

    def _simulate(self, s, depth):
        if np.power(self.gamma, depth) < self.epsilon:
            return 0

        if True:  # if fs has untried actions:
            a = None  # Sample a from untried actions
            s_prime, r = self._step(s, a)
            return r + self.gamma * self._rollout(s_prime, depth + 1)

    def _MCTS(self, initial_loc, max_iters=0):
        for i in range(max_iters):
            self._simulate(initial_loc, 0)

        Q_at_state = self.Q[initial_loc[0], initial_loc[1]]
        action_index = np.argmax(Q_at_state)
        action = self.planning_steps[action_index]

        return action  # Argmax_a over Q(s_0,a)

    def plan(
        self, world_model, current_location, n_steps, variance_mean_tradeeoff=1000
    ):
        """
        Arguments:
            world_model: the belief of the world
            current_location: The location (n,)
            n_steps: How many planning steps to take

        Returns:
            A plan specifying the list of locations
        """
        super().plan(world_model, current_location, n_steps)
        self._MCTS(self.index_loc)
