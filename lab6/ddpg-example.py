"""DLP DDPG Lab"""
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
import matplotlib.pyplot as plt

class OrnsteinUhlenbeckProcess:
  """1-dimension Ornstein-Uhlenbeck process"""
  def sample(self, mu=0, std=.2, theta=.15, dt=1e-2, sqrt_dt=1e-1):
    self.x += theta * (mu - self.x) * dt + std * sqrt_dt * random.gauss(0, 1)
    return self.x

  def reset(self, x0=0):
    self.x = x0


class ReplayMemory:
  def __init__(self, capacity):
    self._buffer = deque(maxlen=capacity)

  def __len__(self):
    return len(self._buffer)

  def append(self, *transition):
    # (state, action, reward, next_state, done)
    self._buffer.append(tuple(map(tuple, transition)))

  def sample(self, batch_size=1):
    return random.sample(self._buffer, batch_size)


class ActorNet(nn.Module):
  def __init__(self, state_dim=3, action_dim=1, hidden_dim=(400, 300)):
    super().__init__()
    self.fn1 = nn.Linear(state_dim,hidden_dim[0])
    self.fn2 = nn.Linear(hidden_dim[0],hidden_dim[1])
    self.fn3 = nn.Linear(hidden_dim[1],action_dim)
    

  def forward(self, x):
    #x = torch.t(x)
    x = F.relu(self.fn1(x))
    x = F.relu(self.fn2(x))
    x = torch.tanh(self.fn3(x))
    return x


class CriticNet(nn.Module):
  def __init__(self, state_dim=3, action_dim=1, hidden_dim=(400, 300)):
    super().__init__()
    h1, h2 = hidden_dim
    self.critic_head = nn.Sequential(
        nn.Linear(state_dim, h1),
        nn.ReLU(),
    )
    self.critic = nn.Sequential(
        nn.Linear(h1 + action_dim, h2),
        nn.ReLU(),
        nn.Linear(h2, action_dim),
    )

  def forward(self, x, action):
    x = self.critic_head(x)
    return self.critic(torch.cat([x, action], dim=1))

def select_action(state,model,is_test = True,low=-2, high=2):
  """based on the behavior (actor) network and exploration noise"""
  if is_test == False:
    with torch.no_grad():
      action = model(state).data.item() + random_process.sample()
  else:
    with torch.no_grad():
      action = model(state).data.item() 

    
  return max(min(action, high), low)
 

def update_behavior_network(actor_net,critic_net,target_actor_net,target_critic_net,critic_opt,criterion):
  def transitions_to_tensors(transitions, device=args.device):
    """convert a batch of transitions to tensors"""

    return (torch.Tensor(x).to(device) for x in zip(*transitions))
  
  # sample a minibatch of transitions
  transitions = memory.sample(args.batch_size)
  state, action, reward, state_next, done = transitions_to_tensors(transitions)

  ## update critic ##
  
  q_value = critic_net(state,action)
  with torch.no_grad():
    a_next = target_actor_net(state_next)
    q_next = args.gamma * target_critic_net(state_next,a_next) + reward
  critic_loss = criterion(q_value, q_next)

  # optimize critic
  actor_net.zero_grad()
  critic_net.zero_grad()
  critic_loss.backward()
  critic_opt.step()

  ## update actor ##
  action = actor_net(state)
  actor_loss = -torch.mean(critic_net(state, action))
 
  
  # optimize actor
 
  actor_net.zero_grad()
  critic_net.zero_grad()
  actor_loss.backward()
  actor_opt.step()


def update_target_network(target_net, net):
  tau = args.tau
  for target, behavior in zip(target_net.parameters(), net.parameters()):
    target.data.copy_(tau * behavior + (1-tau) * target)

def detailOfTensor(Tensor):
    print(Tensor,Tensor.dtype,Tensor.size(),Tensor.device)

def train(env,actor_net,critic_net,target_actor_net,target_critic_net,actor_opt,critic_opt,criterion):
  print('Start Training')
  total_steps = 0
  r_total = []
  e_total =  [(i+1)  for i in range(args.episode)] 
  for episode in range(args.episode):
    total_reward = 0
    random_process.reset()
    state = env.reset()
    for t in itertools.count(start=1):
      # select action
      if total_steps < args.warmup:
        action = float(env.action_space.sample())
      else:
        state_tensor = torch.Tensor(state).to(args.device)
        state_tensor = state_tensor.squeeze(-1)
        action = select_action(state_tensor,actor_net,is_test=1)
       
      # execute action
    
      next_state, reward, done, _ = env.step([action])
     
      # store transition
      memory.append(state, [action], [reward], next_state, [int(done)])
      if total_steps >= args.warmup:
        # update the behavior networks
        update_behavior_network(actor_net,critic_net,target_actor_net,target_critic_net,critic_opt,criterion)
        # update the target networks
        update_target_network(target_actor_net, actor_net)
        update_target_network(target_critic_net, critic_net)

      state = next_state
      total_reward += reward
      total_steps += 1
      if done:
        print('Step: {}\tEpisode: {}\tLength: {}\tTotal reward: {}'.format(
            total_steps, episode, t, total_reward))
        r_total.append(total_reward)
        break
  plt.title('DDPG training reward')
  plt.ylabel('Reward',fontsize=16)
  plt.xlabel('Episode',fontsize=16)
  plt.plot(e_total,r_total)
  plt.savefig('ddpg_result.png',dpi = 200)
  plt.clf()
  env.close()


def test(env, render,actor_net):
  print('Start Testing')
  seeds = (20190813 + i for i in range(10))
  average_reward = 0
  total_steps = 0
  for seed in seeds:
    total_reward = 0
    env.seed(seed)
    state = env.reset()
    for t in itertools.count(start=1):
      # select action
      state_tensor = torch.Tensor(state).to(args.device)
      state_tensor = state_tensor.squeeze(-1)
      action = select_action(state_tensor,actor_net)
   
      # execute action
      next_state, reward, done, _ = env.step([action])
      # store transition
    
      state = next_state
      total_reward += reward
      total_steps += 1
      if done:
        print('Total_Step: {}     Total_reward: {}'.format(
            total_steps, total_reward))
        average_reward += total_reward
        break
  print('Average reward: {}\t '.format(average_reward / 10))   
  env.close()


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-d', '--device', default='cuda')
  # network
  parser.add_argument('-m', '--model', default='pendulum_model')
  parser.add_argument('--restore', action='store_true')
  # train
  parser.add_argument('-e', '--episode', default=1500, type=int)
  parser.add_argument('-c', '--capacity', default=20000, type=int)
  parser.add_argument('-bs', '--batch_size', default=64, type=int)
  parser.add_argument('--warmup', default=10000, type=int)
  parser.add_argument('--lra', default=1e-4, type=float)
  parser.add_argument('--lrc', default=1e-3, type=float)
  parser.add_argument('--gamma', default=.99, type=float)
  parser.add_argument('--tau', default=.001, type=float)
  # test
  parser.add_argument('--render', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  # environment
  env = gym.make('Pendulum-v0')
  # behavior network
  actor_net = ActorNet().to(args.device)
  critic_net = CriticNet().to(args.device)
  if not args.restore:
    # target network
    target_actor_net = ActorNet().to(args.device)
    target_critic_net = CriticNet().to(args.device)
    # initialize target network
    target_actor_net.load_state_dict(actor_net.state_dict())
    target_critic_net.load_state_dict(critic_net.state_dict())
    
    actor_opt =  optim.Adam(actor_net.parameters(), lr=args.lra)
    critic_opt =  optim.Adam(critic_net.parameters(), lr=args.lrc)
    criterion = nn.MSELoss()
   
    # random process
    random_process = OrnsteinUhlenbeckProcess()
    # memory
    memory = ReplayMemory(capacity=args.capacity)
    # train
    train(env,actor_net,critic_net,target_actor_net,target_critic_net,actor_opt,critic_opt,criterion)
    # save model
    torch.save(
        {
            'actor': actor_net.state_dict(),
            'critic': critic_net.state_dict(),
        }, args.model)
  # load model
  model = torch.load(args.model)
  actor_net.load_state_dict(model['actor'])
  critic_net.load_state_dict(model['critic'])
  # test
  test(env, args.render,actor_net)
