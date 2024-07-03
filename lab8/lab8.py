
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from itertools import count
from typing import Union, Tuple, Callable
import json
import glob

# ## Task 1: Replay Buffer

class ReplayBuffer(object):
    def __init__(self, size: int = 10000):
        """
        Constructor
        
        Parameters
        ----------
        size
            Maximum number of transitions store in the buffer.
            If the buffer overflows, older states are dropped.
        """
        self.size    = size
        self.length  = 0
        self.idx     = -1
        
        # define buffers
        self.states        = None
        self.states_next   = None
        self.actions       = None
        self.rewards       = None
        self.done          = None
        
    def store(self, 
              s: Union[torch.Tensor, np.ndarray], 
              a: int, 
              r: float, 
              s_next: Union[torch.Tensor, np.ndarray],
              done: bool):
        
        """
        Stores one sample of experience
        
        Parameters
        ----------
        s
            Tensor encoding the current state.
        a
            Current action.
        r
            Current reward.
        s_next
            Tensor encoding the next state.
        done
            Done signal.
        """
        
        # initialize buffers
        if self.states is None:
            print(self.size, s.shape)
            # self.states      = torch.zeros([self.size] + list(s.shape))   # shape is (self.size, 4)
            self.states      = torch.zeros([self.size] + list(s.shape))   # shape is (self.size, 4)
            self.states_next = torch.zeros_like(self.states)              # shape is (self.size, 4)
            self.actions     = torch.zeros((self.size, ))                 # shape is (self.size, )
            self.rewards     = torch.zeros((self.size, ))                 # shape is (self.size, )
            self.done        = torch.zeros((self.size, ))                 # shape is (self.size, ) 
        
        # TODO: store current (s, a, r, s_next, done) behavior sample in the corresponding tensor buffers
        # Note 1: older instances are overwritten if the buffer overflows.
        # Note 2: increment buffer length after each update, until it reaches the maximum allowed value: self.size

        if isinstance(s, torch.Tensor):
            self.states[self.idx] = s.clone().detach()
        elif isinstance(s, np.ndarray):
            self.states[self.idx] = torch.tensor(s)
        
        if isinstance(s_next, torch.Tensor):
            self.states_next[self.idx] = s_next.clone().detach()
        elif isinstance(s_next, np.ndarray):
            self.states_next[self.idx] = torch.tensor(s_next)

        # self.states_next[self.idx] = torch.tensor(s_next).clone().detach()
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.done[self.idx] = done
        self.length = min(self.length + 1, self.size)
        self.idx = (self.idx + 1) % self.size
        
        
    def sample(self, batch_size: int = 128) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a batch of experience
        
        Parameters
        ----------
        batch_size
            Number of experience to sample
            
        Returns
        -------
        Tuple of tensor consisting of a batch of states, actions, rewards, next states, done
        """
        
        assert self.length >= batch_size, "Can not sample from the buffer yet"
        indices = np.random.choice(a=np.arange(self.length), size=batch_size, replace=False)
        
        # Sample (s, a, r, s_next, done) behavior samples   
        # s      = ...         (batch_size, 4)
        # s_next = ...    (batch_size, 4)
        # a = ...              (batch_size, )
        # r = ...               (batch_size, )
        # done = ...    (batch_size, )

        s = self.states[indices]
        s_next = self.states_next[indices]
        a = self.actions[indices]
        r = self.rewards[indices]
        done = self.done[indices]
        
        return s, a, r, s_next, done

# ## Network achitecture

class DQN_RAM(nn.Module):
    def __init__(self, in_features: int, num_actions: int):
        """
        Initialize a deep Q-learning network for testing algorithm
        
        Parameters
        ----------
        in_features
            number of features of input.
        num_actions
            number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN_RAM, self).__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        
        # define architecture
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ## Epsilon scheduler

def eps_generator(max_eps: float=1.0, min_eps: float=0.1, max_iter: int = 10000):
    crt_iter = -1
    
    while True:
        crt_iter += 1
        frac = min(crt_iter/max_iter, 1)
        eps = (1 - frac) * max_eps + frac * min_eps
        yield eps

# ## Epsilon greedy policy

def select_epilson_greedy_action(Q: nn.Module, s: Tensor, eps: float):
    rand = np.random.rand()
    
    # with prob eps select a random action
    if rand < eps:
        return np.random.choice(np.arange(Q.num_actions))
    
    # select best action
    with torch.no_grad():
        output = Q(s).argmax(dim=1).item()
    
    return output

# ## Task 2: DQN target

@torch.no_grad()
def dqn_target(
    Q: nn.Module,
    target_Q: nn.Module,
    r_batch: Tensor,
    s_next_batch: Tensor,
    done_batch: Tensor,
    gamma: float) -> Tensor:
    """
    Computes DQN target
    
    Parameters:
    -----------
    Q
        Behavior Q network.
    target_Q
        Target Q network.
    r_batch
        Batch of rewards.
    s_next_bacth
        Batch of next states.
    done_batch
        Batch of done flag (1 means the episoded finished).
    gamma
        Discount factor.
    
    Returns
    -------
    Batch of DQN targets
    """
    # compute next Q value based on which action gives max Q values
    # Note:  decorator torch.no_grad() ensures that gradients computed based on next Q are not propagated to the target_Q network
    # Note: take note of the done_batch values - if behavior sample i in the batch has a done flag (done_batch[i] = 1), then the next_Q_values[i] 
    #             will only consider the reward[i] (because there is no next_state)
    
    next_Q_values = target_Q(s_next_batch).max(dim=1)[0]
    next_Q_values[done_batch == 1] = 0
    return r_batch + (gamma * next_Q_values)

# ## Task 3: DDQN target

@torch.no_grad()
def ddqn_target(
    Q: nn.Module,
    target_Q: nn.Module,
    r_batch: Tensor,
    s_next_batch: Tensor,
    done_batch: Tensor,
    gamma: float) -> Tensor:
    """
    Computes DQN target
    
    Parameters:
    -----------
    Q
        Behavior Q network.
    target_Q
        Target Q network.
    r_batch
        Batch of rewards.
    s_next_bacth
        Batch of next states.
    done_batch
        Batcho of done flag (1 means the episoded finished).
    gamma
        Discount factor.
    
    Returns
    -------
    Batch of DQN targets
    """
    # cmpute next Q value based on which action gives max Q values
    next_Q_values = target_Q(s_next_batch).gather(1, Q(s_next_batch).argmax(dim=1, keepdim=True)).squeeze()
    next_Q_values[done_batch == 1] = 0
    
    # Compute the target of the current Q values
    return r_batch + (gamma * next_Q_values)

# ##  Learning Alogrithm

def learning(
    env: gym.Env,
    targert_function: Callable,
    batch_size: int = 128,
    gamma: float = 0.99,
    replay_buffer_size=10000,
    num_episodes: int = 100000,
    learning_starts: int = 1000,
    learning_freq: int = 4,
    target_update_freq: int = 100,
    log_every: int = 100,
    lr_SGD: float = 1e-3,
    max_iter_eps: int = 10000):

    """
    DQN Learning
    
    Parameters
    ----------
    env
        gym environment to train on.
    target_function
        Function that computes the Q network target. For DQN - dqn_target, for DDQN - ddqn_target.
    batch_size:
        How many transitions to sample each time experience is replayed.
    gamma
        Discount Factor
    replay_buffer_size
        Replay buffer size.
    num_episodes
        number of episodes to run
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    log_every:
        Logging interval
    """
    # This means we are running on low-dimensional observations (e.g. RAM)
    input_arg = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # define device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize target q function and q function
    Q = DQN_RAM(input_arg, num_actions).to(device)
    target_Q = DQN_RAM(input_arg, num_actions).to(device)
      
    # Construct Q network optimizer function
    optimizer = optim.Adam(Q.parameters(), lr=lr_SGD)
    
    # define criterion
    criterion = nn.MSELoss()

    # Construct the replay buffer
    replay_buffer = ReplayBuffer()
    
    # define epsilon scheduler
    eps_scheduler = iter(eps_generator(max_iter=max_iter_eps))
    
    # define statistics buffer, total number of steps and total number of updates performed
    all_episode_rewards = []
    mean_episode_rewards = []
    mean_episode_losses = []
    total_loss = []
    total_steps = 0
    num_param_updates = 0
    
    for episode in range(1, num_episodes + 1):
        # reset environment
        s, _ = env.reset()
        episode_reward = 0
        
        for _ in count():
            # increse total number of steps
            total_steps += 1
            
            # Choose random action if not yet start learning
            if total_steps > learning_starts:
                eps = next(eps_scheduler)
                s = torch.tensor(s).view(1, -1).float().to(device)
                a = select_epilson_greedy_action(Q, s, eps)
            else:
                a = np.random.choice(np.arange(num_actions))

            # advance one step
            s_next, r, done, _, _ = env.step(a)
            
            # update episode rewards
            episode_reward += r

            # store other info in replay memory
            replay_buffer.store(s, a, r, s_next, done)

            # Resets the environment when reaching an episode boundary.
            if done:
                break

            # update state
            s = s_next

            # perform experience replay and train the network.
            if (total_steps > learning_starts and total_steps % learning_freq == 0):
                for _ in range(learning_freq):
                    # sample experinence from the replay buffer
                    s_batch, a_batch, r_batch, s_next_batch, done_batch = replay_buffer.sample(batch_size)

                    # send everything to device
                    s_batch      = s_batch.float().to(device)
                    a_batch      = a_batch.long().to(device)
                    r_batch      = r_batch.float().to(device)
                    s_next_batch = s_next_batch.float().to(device)
                    done_batch   = done_batch.long().to(device)

                    # comput the q values according to the states and actions
                    Q_values = Q(s_batch).gather(1, a_batch.unsqueeze(1)).view(-1)

                    # Compute the target of the current Q values
                    target_Q_values = targert_function(Q, target_Q, r_batch, s_next_batch, done_batch, gamma)

                    # compute loss
                    loss = criterion(target_Q_values, Q_values)
                    total_loss.append(loss.detach().numpy())

                    # Clear previous gradients before backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # increase number of updates
                    num_param_updates += 1

                    # Periodically update the target network by Q network to target Q network
                    if num_param_updates % target_update_freq == 0:
                        target_Q.load_state_dict(Q.state_dict())

        # append total reward culumated
        all_episode_rewards.append(episode_reward)
        
        # log average reward over the last 100 episodes
        if episode % log_every == 0 and total_steps > learning_starts:
            mean_episode_reward = np.mean(all_episode_rewards[-log_every:])
            mean_episode_rewards.append(mean_episode_reward)

            mean_loss = np.mean(total_loss[-log_every:]).astype(np.float64)
            mean_episode_losses.append(mean_loss)
            print("Episode: %d, Mean reward: %.2f, Mean loss: %.2f, Eps: %.2f" % (episode, mean_episode_reward, mean_loss, eps))
    return mean_episode_rewards, mean_episode_losses

# ## DQN Learning
# 
# #### Task 4a: Modify learning procedure to implement original DQN (model and target networks are the same)
# #### Task 4b: Modify learning procedure to implement target network DQN
# 
# **Note: Experiment with different values of:**
#   - learning_freq 1, 4, 8
#   - target_update_frequency 10, 100, 200
#   - epsilon decay rate      ε=decay(init=0.9, end=0.05, nr_iterations) - explore the nr_iterations

# initialize gym env
            

if __name__ == '__main__':

    generate_data = False
    # activations = ['cos', 'sigmoid', 'tanh']
    # epsilons = [0.0 , 0.1] # 0.2 when decay = TRUE
    # gammas = [0.5, 0.9] # vom varia gamma
    # lrs = [0.0001, 0.0005, 0.001]
    # alphas = [0.01, 0.05, 0.2]

    learning_freqs = [1, 4, 8]
    target_functions = ["dqn_target", "ddqn_target"]
    target_update_freqs = [10, 100, 200]
    max_iter_eps_vect = [5000, 10000, 15000]

    if generate_data:

        for learning_freq in learning_freqs:
            for target_function in target_functions:
                for target_update_freq in target_update_freqs:
                    for max_iter_eps in max_iter_eps_vect:
                       
                        spec = {
                            'learning_freq': learning_freq,
                            'targert_function': target_function,
                            'target_update_freq': target_update_freq,
                            'max_iter_eps' : max_iter_eps
                        }

                        env = gym.make("CartPole-v1")
                        eward, total_loss = None, None
                        # DQN learning
                        if target_function == "dqn_target":
                            reward, total_loss = learning(
                                env=env,                                        # gym environmnet
                                targert_function=dqn_target,      # dqn target construction
                                batch_size=128,                                 # q-network update batch size
                                gamma=0.99,                                     # discount factor
                                replay_buffer_size=10000,                       # size of the replay buffer
                                num_episodes=1000,                              # number of episodes to run
                                learning_starts=1000,                           # number of initial random actions (exploration)
                                learning_freq=spec['learning_freq'],            # frequency of the update
                                target_update_freq=spec['target_update_freq'],  # number of gradient steps after which the target network is updated
                                log_every=100,                                  # logging interval. returns the mean reward per episode.
                                lr_SGD=1e-3,
                                max_iter_eps=spec['max_iter_eps']
                            )
                        else:
                            reward, total_loss = learning(
                                env=env,                                        # gym environmnet
                                targert_function=ddqn_target,      # dqn target construction
                                batch_size=128,                                 # q-network update batch size
                                gamma=0.99,                                     # discount factor
                                replay_buffer_size=10000,                       # size of the replay buffer
                                num_episodes=1000,                              # number of episodes to run
                                learning_starts=1000,                           # number of initial random actions (exploration)
                                learning_freq=spec['learning_freq'],            # frequency of the update
                                target_update_freq=spec['target_update_freq'],  # number of gradient steps after which the target network is updated
                                log_every=100,                                  # logging interval. returns the mean reward per episode.
                                lr_SGD=1e-3,
                                max_iter_eps=spec['max_iter_eps']
                            )

                        # dump the spec dict into a key value string
                        spec_str = '_'.join([f'{k}={v.__name__}' if callable(v) else f'{k}={v}' for k, v in spec.items()])

                        with open(f'dumps/experiment_{spec_str}.json', 'wt') as f:
                            json.dump({
                                'reward': reward,
                                'total_loss': total_loss,
                                'spec': spec
                                }, f)
                            


    else:
        
        files = glob.glob('dumps/*')# experiment*sigmoid*epsilon=0.2*alpha=0.0001*decay=True.json')
        # print(files)

        # fig, ax = plt.subplots(len(files), 2, figsize=(15, 8))
        pattern = r'.*experiment_learning_freq=(\d+)_targert_function=(\w+)_target_update_freq=(\d+)_max_iter_eps=(\d+).json'
        # pattern = r'.*experiment_activation=(\w+)_epsilon=([\d.]+)_gamma=([\d.]+)_lr=([\d.]+)_alpha=([\d.]+)_episodes=(\d+)_decay=(\w+)\.json'

        # plt.figure(figsize=(6, 6))
 

        all_data = dict()
        for k in range(len(files)):
            file = files[k]

            with open(file, 'rt') as f:
                data = json.load(f)

                spec_arr = [v for _, v in data['spec'].items()]
                spec_pair = (spec_arr[0], spec_arr[1], spec_arr[2], spec_arr[3])
                # print(spec_pair)
                all_data[spec_pair] = data

        # for learning_freq in learning_freqs:
        #     for target_function in target_functions:
        #         for target_update_freq in target_update_freqs:
        #             fig = plt.figure(figsize=(20, 6))
        #             ax1 = fig.add_subplot(121)
        #             ax2 = fig.add_subplot(122)
        #             ax1.title.set_text('Rewards')
        #             ax2.title.set_text('Loss')
        #             for max_iter_eps in max_iter_eps_vect:                    
        #                 data = all_data[(learning_freq, target_function, target_update_freq, max_iter_eps)]
        #                 label_name = "learning_freq=" + str(learning_freq) + " target_function=" + target_function + " target_update_freq=" + str(target_update_freq) + " max_iter_eps=" + str(max_iter_eps)
        #                 # plt.plot(data['reward'], )
        #                 print("ok")
        #                 ax1.plot(data['reward'], label=label_name)
        #                 ax2.plot(data['total_loss'], label=label_name)
        #                 # break
        #             label_name = "max_iter_eps/learning_freq=" + str(learning_freq) + " target_function=" + target_function + " target_update_freq=" + str(target_update_freq) + ".png"
        #             # label_name = "gamma/activation=" + activation + " epsilon=" + str(epsilon) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
        #             plt.legend()
        #             plt.savefig(label_name)
        #             plt.close()
        

        # for learning_freq in learning_freqs:
        #     for target_function in target_functions:
        #         for max_iter_eps in max_iter_eps_vect: 
                
        #             fig = plt.figure(figsize=(20, 6))
        #             ax1 = fig.add_subplot(121)
        #             ax2 = fig.add_subplot(122)
        #             ax1.title.set_text('Rewards')
        #             ax2.title.set_text('Loss')
        #             for target_update_freq in target_update_freqs:        
        #                 data = all_data[(learning_freq, target_function, target_update_freq, max_iter_eps)]
        #                 label_name = "learning_freq=" + str(learning_freq) + " target_function=" + target_function + " target_update_freq=" + str(target_update_freq) + " max_iter_eps=" + str(max_iter_eps)
        #                 # plt.plot(data['reward'], )
        #                 print("ok")
        #                 ax1.plot(data['reward'], label=label_name)
        #                 ax2.plot(data['total_loss'], label=label_name)
        #                 # break
        #             label_name = "target_update_freq/learning_freq=" + str(learning_freq) + " target_function=" + target_function + " max_iter_eps=" + str(max_iter_eps) + ".png"
        #             # label_name = "gamma/activation=" + activation + " epsilon=" + str(epsilon) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
        #             plt.legend()
        #             plt.savefig(label_name)
        #             plt.close()
        

        # for learning_freq in learning_freqs:
        #     for target_update_freq in target_update_freqs: 
        #         for max_iter_eps in max_iter_eps_vect: 
                
        #             fig = plt.figure(figsize=(20, 6))
        #             ax1 = fig.add_subplot(121)
        #             ax2 = fig.add_subplot(122)
        #             ax1.title.set_text('Rewards')
        #             ax2.title.set_text('Loss')
        #             for target_function in target_functions:
                           
        #                 data = all_data[(learning_freq, target_function, target_update_freq, max_iter_eps)]
        #                 label_name = "learning_freq=" + str(learning_freq) + " target_function=" + target_function + " target_update_freq=" + str(target_update_freq) + " max_iter_eps=" + str(max_iter_eps)
        #                 # plt.plot(data['reward'], )
        #                 print("ok")
        #                 ax1.plot(data['reward'], label=label_name)
        #                 ax2.plot(data['total_loss'], label=label_name)
        #                 # break
        #             label_name = "target_function/learning_freq=" + str(learning_freq) + " target_update_freq=" + str(target_update_freq) + " max_iter_eps=" + str(max_iter_eps) + ".png"
        #             # label_name = "gamma/activation=" + activation + " epsilon=" + str(epsilon) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
        #             plt.legend()
        #             plt.savefig(label_name)
        #             plt.close()


        for target_function in target_functions:
            for target_update_freq in target_update_freqs: 
                for max_iter_eps in max_iter_eps_vect: 
                
                    fig = plt.figure(figsize=(20, 6))
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    ax1.title.set_text('Rewards')
                    ax2.title.set_text('Loss')
                    for learning_freq in learning_freqs:
                    
                           
                        data = all_data[(learning_freq, target_function, target_update_freq, max_iter_eps)]
                        label_name = "learning_freq=" + str(learning_freq) + " target_function=" + target_function + " target_update_freq=" + str(target_update_freq) + " max_iter_eps=" + str(max_iter_eps)
                        # plt.plot(data['reward'], )
                        print("ok")
                        ax1.plot(data['reward'], label=label_name)
                        ax2.plot(data['total_loss'], label=label_name)
                        # break
                    label_name = "learning_freq/target_function=" + target_function + " target_update_freq=" + str(target_update_freq) + " max_iter_eps=" + str(max_iter_eps) + ".png"
                    # label_name = "gamma/activation=" + activation + " epsilon=" + str(epsilon) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
                    plt.legend()
                    plt.savefig(label_name)
                    plt.close()
