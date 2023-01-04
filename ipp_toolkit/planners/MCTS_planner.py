from ipp_toolkit.config import MEAN_KEY, PLANNING_RESOLUTION, UNCERTAINTY_KEY
from ipp_toolkit.planners.planners import GridWorldPlanner
import numpy as np
import matplotlib.pyplot as plt


class MCTSPlanner(GridWorldPlanner):
    def __init__(
        self,
        grid_start,
        grid_end,
        grid_resolution=PLANNING_RESOLUTION,
        epsilon=0.01,
        n_iters=100,
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

        self.world_model = None
        self.variance_mean_tradeoff = None
        self._init_vars()

    def _reset_vars(self):
        self.Q = np.zeros((self.initial_size[0], self.initial_size[1], 8))
        self.num_sampled = np.zeros((self.initial_size[0], self.initial_size[1], 8))

        # TODO consider eliminating this in favor of just num_samples
        self.untried_actions = np.ones(
            (self.initial_size[0], self.initial_size[1], 8), dtype=bool
        )
        self.world_model = None
        self.variance_mean_tradeoff = None

    def _init_vars(self):
        self._reset_vars()
        # This is the value function on state-action pairs
        # Actions are on an 8-connected grid
        # Precompute which actions are valid
        self.valid_actions = np.ones_like(self.Q, dtype=bool)
        self.valid_actions[:1, :, 0] = False  # Moving down on last row
        self.valid_actions[:, :1, 0] = False  # Moving down on last row

        self.valid_actions[:1, :, 1] = False  # Moving down on last row

        self.valid_actions[:1, :, 2] = False  # Moving down on last row
        self.valid_actions[:, -1:, 2] = False  # Moving down on last row

        self.valid_actions[:, :1, 3] = False  # Moving down on last row

        self.valid_actions[:, -1:, 4] = False  # Moving down on last row

        self.valid_actions[-1:, :, 5] = False  # Moving down on last row
        self.valid_actions[:, :1, 5] = False  # Moving down on last row

        self.valid_actions[-1:, :, 6] = False  # Moving down on last row

        self.valid_actions[-1:, :, 7] = False  # Moving down on last row
        self.valid_actions[:, -1:, 7] = False  # Moving down on last row

    def _step(self, s, a):
        self.untried_actions[s[0], s[1], a] = False

        step = self.planning_steps[a]
        s_prime = s + step  # TODO get this
        loc = self.planning_grid_rectangular[s_prime[0], s_prime[1]]
        values = self.world_model.sample_belief(loc)
        r_prime = (
            values[MEAN_KEY] + self.variance_mean_tradeoff * values[UNCERTAINTY_KEY]
        )

        return s_prime, r_prime

    def _rollout(self, s, depth):
        if np.power(self.gamma, depth) < self.epsilon:
            return 0

        valid_actions = np.where(self.valid_actions[s[0], s[1]])[0]
        a = np.random.choice(valid_actions)
        s_prime, r_prime = self._step(s, a)
        return r_prime + self.gamma * self._rollout(s_prime, depth + 1)

    def _simulate(self, s, depth):
        if np.power(self.gamma, depth) < self.epsilon:
            return 0

        untried_state_actions = self.untried_actions[s[0], s[1]]
        valid_state_actions = self.valid_actions[s[0], s[1]]
        valid_untried_state_actions = np.logical_and(
            valid_state_actions, untried_state_actions
        )
        if np.any(valid_untried_state_actions):
            valid_untried_inds = np.where(valid_untried_state_actions)[0]
            action_ind = np.random.choice(valid_untried_inds)
            s_prime, r = self._step(s, action_ind)
            return r + self.gamma * self._rollout(s_prime, depth + 1)

        num_samples_state = self.num_sampled[s[0], s[1]]
        exploration = self.K * np.sqrt(
            np.log(np.sum(num_samples_state)) / num_samples_state
        )
        expectation = self.Q[s[0], s[1]]

        exploration_plus_expectation = exploration + expectation
        exploration_plus_expectation = exploration_plus_expectation[valid_state_actions]
        valid_inds = np.where(valid_state_actions)[0]
        a = valid_inds[np.argmax(exploration_plus_expectation)]
        s_prime, r_prime = self._step(s, a)
        G = r_prime + self.gamma * self._simulate(s_prime, depth + 1)

        self.num_sampled[s[0], s[1], a] += 1  # Line 31,32

        # Line 33
        q = self.Q[s[0], s[1], a]
        q = q + (G - q) / self.num_sampled[s[0], s[1], a]
        self.Q[s[0], s[1], a] = q
        return G  # line 34

    def _MCTS(self, initial_loc, max_iters=500):
        for i in range(max_iters):
            self._simulate(initial_loc, 0)

        Q_at_state = self.Q[initial_loc[0], initial_loc[1]]
        action_index = np.argmax(Q_at_state)
        action = self.planning_steps[action_index]

        return action  # Argmax_a over Q(s_0,a)

    def _traverse_Q(self, index, n_steps):
        path = []
        for _ in range(n_steps):
            q_values = self.Q[index[0], index[1]]
            action = np.argmax(q_values)
            step = self.planning_steps[action]
            index = index + step
            path.append(self.planning_grid_rectangular[index[0], index[1]])
        return path

    def plan(self, world_model, current_location, n_steps, variance_mean_tradeoff=1000):
        """
        Arguments:
            world_model: the belief of the world
            current_location: The location (n,)
            n_steps: How many planning steps to take

        Returns:
            A plan specifying the list of locations
        """
        super().plan(world_model, current_location, n_steps)
        self.world_model = world_model
        self.variance_mean_tradeoff = variance_mean_tradeoff

        self._MCTS(self.index_loc)
        plan = self._traverse_Q(self.index_loc, n_steps)

        self._reset_vars()
        print(plan)
        return plan
