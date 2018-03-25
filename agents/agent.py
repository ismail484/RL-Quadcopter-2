import os
from help import *


class DDPG:
    """RL agent learns using DDPG."""

    def __init__(self, task):

        self.task = task

        # Load/save parameters
        self.load_weights = False # try to load weights from previously saved models
        self.save_weights_every = 20  # save weights every n episodes, None to disable
        self.model_name = "ddpg-pytorch-{}".format(self.task.__class__.__name__)
        self.model_ext = ".h5"
        self.model_dir = 'out'
        # Save episode stats
        self.stats_filename = os.path.join(
            './out',
            "stats_{}_{}.csv".format(self.model_name, get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save

        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]

        # Noise process
        self.mu = 0.99
        self.theta = 0.15
        self.sigma = 0.3

        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm's parameters
        self.gamma = 0.1  # discount factor
        self.tau = 0.0005  # for soft update of target parameters

        # Episode's variables
        self.episode = 0
        self.episode_duration = 0
        self.total_reward = None
        self.best_total_reward = -np.inf
        self.score = None
        self.best_score = -np.inf
        self.last_states = None
        self.last_action = None
        
        # constrains to z only
        self.state_start = 2
        self.state_end = 3
        self.state_size = (self.state_end - self.state_start)*self.task.action_repeat
        
        # apply same rotor force to all rotor and see post process
        self.action_size = 1
        self.action_low = self.task.action_low
        self.action_high = self.task.action_high
        self.noise = OUNoise(self.action_size, self.mu, self.theta, self.sigma)

        # Actor (Policy) Model
        self.actor_learning_rate = 0.0003
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_opt = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic (Value) Model
        self.critic_learning_rate = 0.003
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_opt = torch.optim.Adam(self.critic_local.parameters(), lr=self.critic_learning_rate)

        if self.load_weights or self.save_weights_every:
            self.actor_filename = os.path.join(self.model_dir,
                                               "{}_actor{}".format(self.model_name, self.model_ext))
            self.critic_filename = os.path.join(self.model_dir,
                                                "{}_critic{}".format(self.model_name, self.model_ext))
            print("Actor filename :", self.actor_filename)  # [debug]
            print("Critic filename:", self.critic_filename)  # [debug]

        # Load pre-trained model weights, if available
        if self.load_weights and os.path.isfile(self.actor_filename):
            self.load_weights_from_file()

        if self.save_weights_every:
            print("Saving model weights", "every {} episodes".format(
                self.save_weights_every) if self.save_weights_every else "disabled")  # [debug]

    def preprocess_state(self, states):
        """Reduce the state vector to relevant dimensions."""

        repeated_states = np.reshape(states, [self.task.action_repeat,-1])
        return repeated_states[:, self.state_start:self.state_end]  # z positions only

    def postprocess_action(self, action):
        """Return a complete action vector."""
        
        complete_action = action * np.ones((self.task.action_size, 1))  # shape: (4,)
        return complete_action

    def reset_episode(self):
        self.score = self.total_reward / float(self.episode_duration) if self.episode_duration else -np.inf
        if self.best_score < self.score:
            self.best_score = self.score
        if self.total_reward and self.total_reward > self.best_total_reward:
            self.best_total_reward = self.total_reward
        self.total_reward = None
        self.episode_duration = 0
        self.last_states = None
        self.last_action = None
        state = self.task.reset()
        self.episode += 1
        return state

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
                        header=not os.path.isfile(self.stats_filename))  # write header first time only

    def step(self, states, reward, done):
        states = self.preprocess_state(states)
        if self.total_reward:
            self.total_reward += reward
        else:
            self.total_reward = reward

        self.episode_duration += 1
        # Save experience / reward
        if self.last_states is not None and self.last_action is not None:
            self.memory.add(self.last_states, self.last_action, reward, states, done)


        self.last_states = states
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        if done:
            self.write_stats([self.episode, self.total_reward])
            if self.save_weights_every and self.episode % self.save_weights_every == 0:
                self.save_weights()

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = self.preprocess_state(states)
        states = np.reshape(states, [-1, self.state_size])
        actions = self.predict_actions(states)
        actions = actions + self.noise.sample()  # add some noise for exploration
        self.last_action = actions
        actions = self.postprocess_action(actions)
        return actions

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def load_weights_from_file(self):
        try:
            self.actor_local.load_state_dict(torch.load(self.actor_filename))
            self.critic_local.load_state_dict(torch.load(self.critic_filename))
            self.actor_target.load_state_dict(self.actor_local.state_dict())
            self.critic_target.load_state_dict(self.critic_local.state_dict())
            print("Model weights loaded from file!")  # [debug]
        except Exception as e:
            print("Unable to load model weights from file!")
            print("{}: {}".format(e.__class__.__name__, str(e)))

    def save_weights(self):
        torch.save(self.actor_local.state_dict(), self.actor_filename)
        torch.save(self.critic_local.state_dict(), self.critic_filename)
        # print("Model weights saved at episode", self.episode)  # [debug]

    def predict_actions(self, states):

        return to_numpy(
            self.actor_local(to_tensor(np.array([states])))
        ).squeeze(0)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None]).reshape(-1, self.task.action_repeat)
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).reshape(-1, self.task.action_repeat)

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target(to_tensor(next_states, volatile=True))
        Q_targets_next = self.critic_target([
            to_tensor(next_states, volatile=True),
            actions_next,
        ])
        Q_targets_next.volatile = False

        # Compute Q targets for current states and train critic model (local)
        Q_targets = to_tensor(rewards) + to_tensor(np.array([self.gamma])) * Q_targets_next * (1 - to_tensor(dones))

        self.critic_local.zero_grad()
        Q_train = self.critic_local([to_tensor(states), to_tensor(actions)])
        v_loss = torch.nn.MSELoss()(Q_train,Q_targets)
        v_loss.backward()
        self.critic_opt.step()

        # Train actor model (local)
        self.actor_local.zero_grad()
        p_loss = -self.critic_local([
            to_tensor(states),
            self.actor_local(to_tensor(states))
        ])
        p_loss = p_loss.mean()
        p_loss.backward()
        self.actor_opt.step()

        # Soft-update target models
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)


class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, action_low, action_high, h_units_1=64, h_units_2=64, weights_init=3e-3):
        super(Actor, self).__init__()
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.fc1 = torch.nn.Linear(state_size, h_units_1)
        self.fc2 = torch.nn.Linear(h_units_1, h_units_2)
        self.fc3 = torch.nn.Linear(h_units_2, action_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.scale = LambdaLayer(lambda x: (x * to_tensor(np.array([self.action_range]))) + to_tensor(np.array([self.action_low])))
        self.init_weights(weights_init)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = self.scale(out)
        return out


class Critic(torch.nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, h_units_1=64, h_units_2=64, weights_init=3e-3):
        super(Critic, self).__init__()
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.fc1 = torch.nn.Linear(state_size, h_units_1)
        self.fc2 = torch.nn.Linear(h_units_1 + action_size, h_units_2)
        self.fc3 = torch.nn.Linear(h_units_2, 1)
        self.relu = torch.nn.ReLU()
        self.init_weights(weights_init)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
