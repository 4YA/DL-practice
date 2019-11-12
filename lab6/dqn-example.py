"""DLP DQN Lab"""
__author__ = 'chengscott'
__copyright__ = 'Copyright 2019, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy
import matplotlib.pyplot as plt


class ReplayMemory:
  def __init__(self, capacity):
    self._buffer = deque(maxlen=capacity)

  def __len__(self):
    return len(self._buffer)

  def append(self, *transition):
    # (state, action, reward, next_state, done)
 
    self._buffer.append(tuple(map(tuple, transition)))
    #self._buffer.append(transition)

  def sample(self, batch_size=1):
    return random.sample(self._buffer, batch_size)


class DQN(nn.Module):
  def __init__(self, state_dim=4, action_dim=2, hidden_dim=24):
    super(DQN, self).__init__()
    self.fn1 = nn.Linear(state_dim,hidden_dim)
    self.fn2 = nn.Linear(hidden_dim,hidden_dim)
    self.fn3 = nn.Linear(hidden_dim,action_dim)
  
  def forward(self, x):
    x = F.leaky_relu(self.fn1(x))
    x = F.leaky_relu(self.fn2(x))
    x = F.leaky_relu(self.fn3(x))
    return x

def detailOfTensor(Tensor):
    print(Tensor,Tensor.dtype,Tensor.size(),Tensor.device)

def select_action(epsilon, state, model,action_dim=2):
  """epsilon-greedy based on behavior network"""
  
  if random.random() < epsilon:
    return torch.argmax(torch.FloatTensor(2))
  else:
    with torch.no_grad():
      return torch.argmax(model(state))

def update_behavior_network(behavior,target,optimizer,criterion):
  def transitions_to_tensors(transitions, device=args.device):
    """convert a batch of transitions to tensors"""
    
    return (torch.Tensor(x).to(device) for x in zip(*transitions))
 
  # sample a minibatch of transitions
  transitions = memory.sample(args.batch_size)
  
  state, action, reward, next_state, done = transitions_to_tensors(transitions)
  action = action.type(torch.LongTensor).to(args.device)
  action = action.squeeze(1)
 
  
 
  q_value = behavior(state)
  #detailOfTensor(q_value)
  q_value = torch.cat([torch.index_select(q_value[x], 0, action[x]) for x in range(args.batch_size)])
  #detailOfTensor(q_value)
  #detailOfTensor(target(next_state))
  #detailOfTensor(torch.max(target(next_state)))
  
  with torch.no_grad():
    behavior_next_temp =  behavior(next_state)
    target_next_temp = target(next_state)
  
    q_next = torch.cat([reward[x]-reward[x] if done[x] else reward[x] + 
            args.gamma*target_next_temp[x][torch.argmax(behavior_next_temp[x]).data.cpu().numpy()] 
            for x in range(args.batch_size)] )
  
  loss = criterion(q_value, q_next)
  
  # optimize
  optimizer.zero_grad()
  loss.backward()
  nn.utils.clip_grad_norm_(behavior.parameters(), 5)
  optimizer.step()


def train(env,behavior,target,optimizer,criterion):
  print('Start Training')
  total_steps, epsilon = 0, 1.
  r_total = []
  e_total =  [(i+1)  for i in range(args.episode)] 
  for episode in range(args.episode):
    total_reward = 0
    state = env.reset()
    for t in itertools.count(start=1):
      # select action
      if total_steps < args.warmup:
        action = env.action_space.sample()
        action = torch.LongTensor([action]).to(args.device)
        action = action.squeeze(0)
      else:
        state_tensor = torch.Tensor(state).to(args.device)
        action = select_action(epsilon, state_tensor, behavior).to(args.device)
        
        epsilon = max(epsilon * args.eps_decay, args.eps_min)
      # execute action
      next_state, reward, done, _ = env.step(action.data.cpu().numpy())
     
      # store transition
      memory.append(state, [action], [reward / 10], next_state, [int(done)])
      if total_steps >= args.warmup and total_steps % args.freq == 0:
        # update the behavior network
        update_behavior_network(behavior,target,optimizer,criterion)
      if total_steps % args.target_freq == 0:
        target.load_state_dict(behavior.state_dict())
        

      state = next_state
      total_reward += reward
      total_steps += 1
      if done:
        print('Step: {}\tEpisode: {}\tTotal reward: {}\tEpsilon: {}'.format(
            total_steps, episode, total_reward, epsilon))
        r_total.append(total_reward)
        break
  
  plt.title('DQN training reward')
  plt.ylabel('Reward',fontsize=16)
  plt.xlabel('Episode',fontsize=16)
  plt.plot(e_total,r_total)
  plt.savefig('dqn_result.png',dpi = 200)
  plt.clf()
  env.close()



def test(env, render,target):
  print('Start Testing')
  epsilon = args.test_epsilon
  seeds = (20190813 + i for i in range(10))
  average_reward = 0
  for seed in seeds:
    total_reward = 0
    env.seed(seed)
    state = env.reset()
    total_steps = 0
    for t in itertools.count(start=1):
      state_tensor = torch.Tensor(state).to(args.device)
      action = select_action(0, state_tensor, target)
      # execute action
      next_state, reward, done, _ = env.step(action.data.cpu().numpy())
      state = next_state
      total_reward += reward
      total_steps += 1
      if done:
        print('Total_Step: {}\t Total_reward: {}'.format(
            total_steps, total_reward))
        average_reward += total_reward
        break
  print('Average reward: {}\t '.format(average_reward / 10))   
  env.close()


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-d', '--device', default='cuda')
  # network
  parser.add_argument('-m', '--model', default='cartpole_model')
  parser.add_argument('--restore', action='store_true')
  # train
  parser.add_argument('-e', '--episode', default=1000, type=int)
  parser.add_argument('-c', '--capacity', default=15000, type=int)
  parser.add_argument('-bs', '--batch_size', default=512, type=int)
  parser.add_argument('--warmup', default=10000, type=int)
  parser.add_argument('--lr', default=.0005, type=float)
  parser.add_argument('--eps_decay', default=.995, type=float)
  parser.add_argument('--eps_min', default=.01, type=float)
  parser.add_argument('--gamma', default=.99, type=float)
  parser.add_argument('--freq', default=4, type=int)
  parser.add_argument('--target_freq', default=100, type=int)
  # test
  parser.add_argument('--test_epsilon', default=.001, type=float)
  parser.add_argument('--render', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  # environment
  env = gym.make('CartPole-v1')
  # behavior network
  behavior_net = DQN().to(args.device)
  if not args.restore:
    # target network
    target_net = DQN().to(args.device)
    # initialize target network
    target_net.load_state_dict(target_net.state_dict())
    # optimizer
    optimizer = optim.Adam(behavior_net.parameters(), lr=args.lr)
    # criterion
    criterion = nn.MSELoss()
    # memory
    memory = ReplayMemory(capacity=args.capacity)
    # train
    train(env,behavior_net,target_net,optimizer,criterion)
    # save model
    torch.save(behavior_net, args.model)
  # load model
  behavior_net = torch.load(args.model)
  # test
  test(env, args.render ,behavior_net)
