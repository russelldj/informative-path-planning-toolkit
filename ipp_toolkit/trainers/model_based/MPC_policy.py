import numpy as np
import gym

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        #self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        if len(dyn_models) > 1:
            raise RuntimeError('dyn models greater than 1 currently not supported for model-based')

        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.observation_space.shape[0]

        # action space
        self.ac_space = self.action_space
        self.ac_dim = ac_dim

        if isinstance(self.action_space, gym.spaces.box.Box):
            self.low = self.ac_space.low
            self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def get_random_actions(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample actions and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high]
        if isinstance(self.action_space, gym.spaces.box.Box):
            mult_coeff = (self.high - self.low)
            add_coeff = self.low
            acts = np.random.rand(num_sequences, horizon, self.ac_dim)*mult_coeff + add_coeff
        elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
            acts = np.random.randint(0, self.action_space.n, size=(num_sequences, horizon, self.ac_dim))
        else:
            raise RuntimeError('Illegal action space: ' + str(self.action_space))
        
        return acts

    def sample_action_sequences(self, num_sequences, horizon, env, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            random_action_sequences = self.get_random_actions(num_sequences, horizon) # TODO(Q1) sample random actions
            return random_action_sequences

        elif self.sample_strategy == 'cem':
            raise RuntimeError('not supported')
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf

            #initialize CEM distribution and initial actions
            random_action_sequences = self.get_random_actions(num_sequences, horizon)
            
            mean = np.mean(random_action_sequences, axis=(0))
            var = np.std(random_action_sequences, axis=(0))**2

            for i in range(self.cem_iterations):
                predicted_map_errors = self.evaluate_candidate_sequences(random_action_sequences, obs, env)
                elite_inds = np.argsort(predicted_map_errors)[0:self.cem_num_elites]
                elite_action_sequences = random_action_sequences[elite_inds]

                elite_mean = np.mean(elite_action_sequences, axis=(0))
                elite_var = np.std(elite_action_sequences, axis=(0))**2

                mean = self.cem_alpha*elite_mean + (1-self.cem_alpha)*mean
                var = self.cem_alpha*elite_var + (1-self.cem_alpha)*var

                random_action_sequences = np.random.normal(mean, np.sqrt(var), size=(num_sequences, horizon, self.ac_dim))
                #WARNiNG WARNING WARNING
                #TODO fix this
                random_action_sequences[random_action_sequences < 0] = 0
                random_action_sequences[random_action_sequences > 48] = 48
                random_action_sequences = np.round(random_action_sequences).astype(int)

                #random_action_sequences = np.random.multivariate_normal(mean, np.diag(var), size=(num_sequences, horizon))

                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self.ac_dim)
            #mosr = self.evaluate_candidate_sequences(random_action_sequences, obs)
            #cem_action = random_action_sequences[np.argmax(mosr)]

            mean[mean < 0] = 0
            mean[mean > 48] = 48
            mean = np.round(mean).astype(int)
            cem_action = mean

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs, env):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        
        rewards = np.zeros((len(self.dyn_models), candidate_action_sequences.shape[0]))
        index = 0
        for model in self.dyn_models:
            rewards[index, :] = self.calculate_rewards(obs, candidate_action_sequences, model, env)
            index += 1

        mean_rewards = np.mean(rewards, axis=0)
        
        return mean_rewards

    def get_action(self, obs, env):
        if True or self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1, env=env)[0]

        # sample random actions (N x horizon)
        horizon = np.min([env.max_steps - env.num_steps, self.horizon])
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=horizon, env=env, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs, env)
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence =  np.argmax(predicted_rewards) # TODO (Q2)
            action_to_take =  candidate_action_sequences[best_action_sequence, 0]# TODO (Q2)
            
            return action_to_take[None]  # Unsqueeze the first index

    #TODO make this so can choose multiple horizons
    #TODO rename this
    #TODO if large discretization, don't use all, randomly select
    #TODO first action is always same, maybe add randomness
    def get_best_action(self, obs, env):
        if len(self.dyn_models) > 1:
            raise RuntimeError('more than one dyn_model not supported')

        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            acts = np.arange(self.action_space.n)
            acts = np.expand_dims(acts, 1)

            obs_sequences = np.vstack([obs]*acts.shape[0])
            next_obs =  self.dyn_models[0].get_prediction(obs_sequences, acts, None)
        else:
            acts = self.get_random_actions(self.N, 1)

            obs_sequences = np.vstack([obs]*acts.shape[0])
            next_obs =  self.dyn_models[0].get_prediction(obs_sequences, acts[:, 0, :], None)

        predicted_rewards = env.get_est_reward(next_obs)
        best_action_sequence =  np.argmax(predicted_rewards)
        action_to_take = acts[best_action_sequence, 0]
        
        return action_to_take[None][None]

    def calculate_rewards(self, obs, candidate_action_sequences, model, env):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        obs_sequences = np.vstack([obs]*candidate_action_sequences.shape[0])
        rewards = []

        for i in range(candidate_action_sequences.shape[1]):
            next_obs =  model.get_prediction(obs_sequences, candidate_action_sequences[:, i, :], self.data_statistics)

            #WARNING WARNING WARNING
            #might be affected by env 

            #reward, _ = self.env.get_reward(next_obs, candidate_action_sequences[:, i, :])
            reward = env.get_est_reward(next_obs)
            rewards.append(reward)
            obs_sequences = next_obs

        rewards = np.stack(rewards, axis=1)
        
        #sum_of_rewards = np.sum(rewards, axis=1)  # TODO (Q2)
        #just take the best, not the sum
        #rewards = rewards.max(axis=1).shape
        rewards = np.sum(rewards, axis=1)
        
        return rewards
