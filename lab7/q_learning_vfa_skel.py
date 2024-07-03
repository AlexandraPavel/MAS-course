import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import json
import tqdm
import random
import sys
import matplotlib.pyplot as plt
import glob
import re

class CosineActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cos(x)

class Estimator(object):
    def __init__(self, state_dim = 4, action_dim = 2, hidden_dim = 100, lr = 0.0001, activation = 'cos'):

        if activation == 'cos':
            activation = CosineActivation()
        elif activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()

        ## TODO 1: Implement the estimator
        self.criterion = torch.nn.MSELoss()
        # self.model = None
        self.model = nn.Sequential(nn.Linear(state_dim, hidden_dim), activation, nn.Linear(hidden_dim, action_dim))
        ## END TODO
        
        self._initialize_weights_and_bias(state_dim, hidden_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    def _initialize_weights_and_bias(self, state_dim = 4, hidden_dim = 100):
        # TODO 2: Initialize the weights and biases of the first layer
        # the first weight is a state_dim x hidden_dim matrix. 
        # Initialize each row with a normal distribution with mean 0 and standard deviation sqrt((i+1) * 0.5), where i is the row index.
        # the bias is uniformly distributed between 0 and 2 pi

        torch.nn.init.uniform_(self.model[0].bias.data, 0, 2*np.pi)
        
        torch.nn.init.normal_(self.model[0].weight.data, mean=0, std=1)
        
        for i in range(state_dim):
            for j in range(hidden_dim):
                self.model[0].weight.data[j, i] *= np.sqrt((i+1) * 0.5)


        pass
        
    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, torch.Tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, state):
            with torch.no_grad():
                if self.model is None:
                    return np.zeros((2,))
                return self.model(torch.Tensor(state))

def select_action(state, epsilon, model):
    ## TODO 3: Implement the epsilon-greedy policy
    action = env.action_space.sample()
    action = np.argmax(model.predict(state)).item() if random.random() > epsilon else action
    ## END TODO
    return action
            


def q_learning(
    env,
    model,
    episodes,
    decay = False,
    gamma = 0.9,
    epsilon = 0.1,
    eps_decay = 0.9,
):

    total_reward = []
    total_loss = []

    for episode in tqdm.tqdm(range(episodes)):
        state, _ = env.reset()

        done = False
        episode_reward = 0

        while not done:
            action = select_action(state, epsilon, model)
            
            step_res = env.step(action)
            next_state, reward, done, _, _ = step_res
            episode_reward += reward

            # TODO 4: Implement the Q-learning update rule, using the model.predict and model.update functions
            # predict the q values for the current state
            q_values = model.predict(state)

            # If the episode is done, the q value for the action taken should be the reward
            if done:
                q_values[action] = torch.tensor(reward)
                
                # compute the loss using the model.update() method which also updates the model
                loss = model.update(state, q_values)
                total_loss.append(loss)
                break
            
            # otherwise, predict the q values for the next state and use them as the TD target for the update
            q_values_next = model.predict(next_state)
            
            # Set the q value for the action taken to the TD target
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            
            # compute the loss using the model.update() which also updates the model
            loss = model.update(state, q_values)
            total_loss.append(loss)

            state = next_state

        # Update epsilon
        if decay:
            epsilon = max(epsilon * eps_decay, 0.01)

        total_reward.append(episode_reward)

    return total_reward, total_loss





if __name__ == '__main__':

    print_flag = False
    activations = ['cos', 'sigmoid', 'tanh']
    epsilons = [0.0 , 0.1] # 0.2 when decay = TRUE
    gammas = [0.5, 0.9] # vom varia gamma
    lrs = [0.0001, 0.0005, 0.001]
    alphas = [0.01, 0.05, 0.2]

    if print_flag:

        for activation in activations:
            for epsilon in epsilons:
                for gamma in gammas:
                    for lr in lrs:
                        for alpha in alphas:
                            spec = {
                                'activation': activation,
                                'epsilon': epsilon,
                                'gamma': gamma,
                                'lr' : lr,
                                'alpha': alpha,
                                'episodes': 2000,
                                'decay': False #True
                            }

                            env = gym.make("CartPole-v1")

                            estimator = Estimator(
                                state_dim = env.observation_space.shape[0],
                                action_dim = env.action_space.n,
                                hidden_dim = 100,
                                lr = spec["lr"] * spec['alpha']
                            )
                            reward, total_loss = q_learning(
                                env,
                                estimator,
                                spec['episodes'],
                                gamma = spec['gamma'],
                                epsilon = spec['epsilon'],
                                decay = spec['decay']
                            )

                            # dump the spec dict into a key value string
                            spec_str = '_'.join([f'{k}={v}' for k, v in spec.items()])

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

        pattern = r'.*experiment_activation=(\w+)_epsilon=([\d.]+)_gamma=([\d.]+)_lr=([\d.]+)_alpha=([\d.]+)_episodes=(\d+)_decay=(\w+)\.json'

        # plt.figure(figsize=(6, 6))
 

        all_data = dict()
        for k in range(len(files)):
            file = files[k]

            with open(file, 'rt') as f:
                data = json.load(f)

                spec_arr = [v for _, v in data['spec'].items()]
                spec_pair = (spec_arr[0], spec_arr[1], spec_arr[2], spec_arr[3], spec_arr[4], spec_arr[6])
                # print(spec_pair)
                all_data[spec_pair] = data

                # label_name = file.split('/')[-1]

                # # Match the pattern in the filename
                # match = re.match(pattern, label_name)
                # activation, epsilon, gamma, lr, alpha, decay = None, None, None, None, None, None

                # # Print the matched groups
                # if match:
                #     print("File:", file)
                #     activation = match.group(1)
                #     epsilon = float(match.group(2))
                #     gamma = float(match.group(3))
                #     lr = float(match.group(4))
                #     alpha = float(match.group(5))
                #     decay = match.group(7)
                #     print(activation, epsilon, gamma, lr, alpha, decay)
                # else:
                #     print("Filename doesn't match the expected pattern:", label_name)
                # plt.plot(data['reward'], label = "activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + decay)
    


        # decay = False
        # k = 0
        # epsilons = [0.0 , 0.1, 0.2]
        # for activation in activations:
        #     for epsilon in epsilons:
        #         if epsilon == 0.2:
        #             decay = True
        #         else:
        #             decay = False
        #         for gamma in gammas:
        #             for lr in lrs:
        #                 fig = plt.figure(figsize=(20, 6))
        #                 ax1 = fig.add_subplot(121)
        #                 ax2 = fig.add_subplot(122)
        #                 ax1.title.set_text('Rewards')
        #                 ax2.title.set_text('Loss')
        #                 for alpha in alphas:
        #                     data = all_data[(activation, epsilon, gamma, lr, alpha, decay)]
        #                     label_name = "activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay)
        #                     # plt.plot(data['reward'], )
        #                     print("ok")
        #                     ax1.plot(data['reward'], label=label_name)
        #                     ax2.plot(data['total_loss'], label=label_name)
        #                     k += 1
        #                     # break
        #                 # label_name = "alpha/activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
        #                 label_name = "alpha/activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " decay=" + str(decay) + ".png"
        #                 plt.legend()
        #                 plt.savefig(label_name)
        #                 plt.close()
        #     #             break
        #     #         break
        #     #     break
        #     # break


        # decay = False
        # k = 0
        # epsilons = [0.0 , 0.1, 0.2]
        # for activation in activations:
        #     for epsilon in epsilons:
        #         if epsilon == 0.2:
        #             decay = True
        #         else:
        #             decay = False
        #         for gamma in gammas:
        #             for alpha in alphas:
                    
        #                 fig = plt.figure(figsize=(20, 6))
        #                 ax1 = fig.add_subplot(121)
        #                 ax2 = fig.add_subplot(122)
        #                 ax1.title.set_text('Rewards')
        #                 ax2.title.set_text('Loss')
        #                 for lr in lrs:
        #                     data = all_data[(activation, epsilon, gamma, lr, alpha, decay)]
        #                     label_name = "activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay)
        #                     # plt.plot(data['reward'], )
        #                     print("ok")
        #                     ax1.plot(data['reward'], label=label_name)
        #                     ax2.plot(data['total_loss'], label=label_name)
        #                     k += 1
        #                     # break
        #                # label_name = "lr/activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
                       
        #                 label_name = "lr/activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
        #                 plt.legend()
        #                 plt.savefig(label_name)
        #                 plt.close()
        #     #             break
        #     #         break
        #     #     break
        #     # break

        decay = False
        k = 0
        epsilons = [0.0 , 0.1, 0.2]
        for activation in activations:
            for epsilon in epsilons:
                if epsilon == 0.2:
                    decay = True
                else:
                    decay = False
                for lr in lrs:
                    for alpha in alphas:
                    
                        fig = plt.figure(figsize=(20, 6))
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122)
                        ax1.title.set_text('Rewards')
                        ax2.title.set_text('Loss')
                        for gamma in gammas:
                            data = all_data[(activation, epsilon, gamma, lr, alpha, decay)]
                            label_name = "activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay)
                            # plt.plot(data['reward'], )
                            print("ok")
                            ax1.plot(data['reward'], label=label_name)
                            ax2.plot(data['total_loss'], label=label_name)
                            k += 1
                            # break

                        label_name = "gamma/activation=" + activation + " epsilon=" + str(epsilon) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
                        plt.legend()
                        plt.savefig(label_name)
                        plt.close()
            #             break
            #         break
            #     break
            # break
                
        # decay = False
        # k = 0
        # epsilons = [0.0 , 0.1, 0.2] # 0.2 when decay = TRUE
        # for gamma in gammas:
        #     for epsilon in epsilons:
        #         if epsilon == 0.2:
        #             decay = True
        #         else:
        #             decay = False
        #         for lr in lrs:
        #             for alpha in alphas:
                    
        #                 fig = plt.figure(figsize=(20, 6))
        #                 ax1 = fig.add_subplot(121)
        #                 ax2 = fig.add_subplot(122)
        #                 ax1.title.set_text('Rewards')
        #                 ax2.title.set_text('Loss')
                        
        #                 for activation in activations:
        #                     data = all_data[(activation, epsilon, gamma, lr, alpha, decay)]
        #                     label_name = "activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay)
        #                     # plt.plot(data['reward'], )
        #                     print("ok")
        #                     ax1.plot(data['reward'], label=label_name)
        #                     ax2.plot(data['total_loss'], label=label_name)
        #                     k += 1
        #                     # break

        #                 label_name = "activation/epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
        #                 plt.legend()
        #                 plt.savefig(label_name)
        #                 plt.close()
        #     #             break
        #     #         break
        #     #     break
        #     # break
                
        # decay = False
        # k = 0
        # epsilons = [0.0 , 0.1, 0.2] # 0.2 when decay = TRUE
        # for gamma in gammas:
        #     for activation in activations:
        #         for lr in lrs:
        #             for alpha in alphas:
                    
        #                 fig = plt.figure(figsize=(20, 6))
        #                 ax1 = fig.add_subplot(121)
        #                 ax2 = fig.add_subplot(122)
        #                 ax1.title.set_text('Rewards')
        #                 ax2.title.set_text('Loss')
                        
        #                 for epsilon in epsilons:
        #                     if epsilon == 0.2:
        #                         decay = True
        #                     else:
        #                         decay = False
        #                     data = all_data[(activation, epsilon, gamma, lr, alpha, decay)]
        #                     label_name = "activation=" + activation + " epsilon=" + str(epsilon) + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay)
        #                     # plt.plot(data['reward'], )
        #                     print("ok")
        #                     ax1.plot(data['reward'], label=label_name)
        #                     ax2.plot(data['total_loss'], label=label_name)
        #                     k += 1
        #                     # break

        #                 label_name = "epsilon/activation=" + activation + " gamma=" + str(gamma) + " lr=" + str(lr) + " alpha=" + str(alpha) + " decay=" + str(decay) + ".png"
        #                 plt.legend()
        #                 plt.savefig(label_name)
        #                 plt.close()
        #     #             break
        #     #         break
        #     #     break
        #     # break


