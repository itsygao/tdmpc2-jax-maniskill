import gymnasium as gym
import numpy as np
import random
from typing import *
import jax
from collections import deque

class ListSet():
    def __init__(self):
        self.pos_dict = {}
        self.items = []

    def add(self, item):
        if item in self.pos_dict:
            return
        self.items.append(item)
        self.pos_dict[item] = len(self.items)-1

    def remove(self, item):
        pos = self.pos_dict.pop(item)
        last_item = self.items.pop()
        if pos != len(self.items):
            self.items[pos] = last_item
            self.pos_dict[last_item] = pos

    def choice(self, n):
        return random.choices(self.items, k=n)


class NewSequentialReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               sequence_length: int = 1,
               seed: Optional[int] = None,
               ):
    """
    Sequential replay buffer with support for parallel environments. 

    To simplify the implementation and speed up sampling, episode boundaries are NOT respected. 
    i.e., the sampled subsequences may span multiple episodes. 
    Any code using this buffer should handle this with termination/truncation signals

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store PER PARALLEL ENVIRONMENT
    dummy_input : Dict
        Example input from the environment. Used to determine the shape and dtype of the data to store, (num_envs, x_dim)
    num_envs : int, optional
        Number of parallel environments used for data collection, by default 1
    seed : Optional[int], optional
        Seed for sampling, by default None
    """
    self.capacity = capacity
    self.num_envs = num_envs
    self.data = jax.tree.map(lambda x: np.zeros(
        (capacity,) + np.asarray(x).shape, np.asarray(x).dtype), dummy_input)

    self.size = 0
    self.current_ind = 0
    self.sequence_length = sequence_length # horizon
    self.breaks = [deque([0, 0]) for _ in range(num_envs)] # consecutive breaks i, j means [i, j) forms a trajectory
    self.traj_lens = [deque([0]) for _ in range(num_envs)] # cached trajectory lengths to avoid duplicate computations
    self.good_starts = [ListSet() for _ in range(num_envs)]

    self.np_random = np.random.RandomState(seed=seed)

  def __len__(self):
    return self.size

  def insert(self, data: Dict, dones) -> None:
    # Insert the data
    # dict(
    # observation: ...,
    # action: ...,
    # reward: ...,
    # next_observation: ...,
    # terminated: ...,
    # truncated: ...,
    jax.tree.map(lambda x, y: x.__setitem__(self.current_ind, y),
                 self.data, data)
    full = self.size == self.capacity

    self.current_ind = (self.current_ind + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

    for idx, done in enumerate(dones):
      self.breaks[idx][-1] = (self.breaks[idx][-1] + 1) % self.capacity # move back the break end ptr. If below checks for new available starts
      if self.breaks[idx][-1] - self.breaks[idx][-2] >= self.sequence_length or \
            (self.breaks[idx][-1] < self.breaks[idx][-2] and self.breaks[idx][-1] + self.capacity - self.breaks[idx][-2] >= self.sequence_length):
        self.good_starts[idx].add((self.breaks[idx][-1] - self.sequence_length + self.capacity) % self.capacity)
      self.traj_lens[idx][-1] += 1

      if full: # full before insertion. Remember [ )
        self.breaks[idx][0] = self.breaks[idx][-1] # update the movement of the [-1] pointer to [0] when full. Now [ )
        if self.breaks[idx][-1] == 0 and self.breaks[idx][0] != 0:
          self.breaks[idx].appendleft(0) # prepend a 0 to signal the new first start of a traj
          self.traj_lens[idx].appendleft(self.breaks[idx][1] - self.breaks[idx][0])
        
        self.traj_lens[idx][0] -= 1
        if self.breaks[idx][0] == self.breaks[idx][1]: # after moving a starting idx, there's duplicates
          self.breaks[idx].popleft() # leftmost element of deque always means a starting idx of the first stored completed traj
          assert self.traj_lens[idx][0] == 0
          self.traj_lens[idx].popleft()
        else: # if not entering this 'else', it means that the if check in 'else' doesn't apply since the first traj has been fully overwritten
          # self.breaks finished updates. Now use the diff btw breaks[1] and breaks[0] to see if good_start also need update
          # print(f"capa {self.capacity}, breaks {self.breaks[idx]}, lens {self.traj_lens[idx]}, good {self.good_starts[idx]}")
          if (self.breaks[idx][1] - self.breaks[idx][0]) % self.capacity >= self.sequence_length - 1: # need removal from good_starts
            self.good_starts[idx].remove((self.breaks[idx][0] - 1) % self.capacity) # FIFO
            # assert popped_start == ((self.breaks[idx][0] - 1) % self.capacity), \
            #   f"capa {self.capacity}, popped {popped_start}, rhs {(self.breaks[idx][0] - 1) % self.capacity}, break {self.breaks[idx]}, {self.good_starts[idx]}"
        
      if done: # add new breakpoints
        self.breaks[idx].append(self.breaks[idx][-1])
        self.traj_lens[idx].append(0)
      
      # if self.np_random.rand() < 0.02:
      #   print(f"capa {self.capacity}, breaks {self.breaks[0]}, lens {self.traj_lens[0]}, good {self.good_starts[0]}")
  
  def get_state(self) -> Dict:
    # Yuan: need extra work to restore
    return {
      'current_ind': self.current_ind,
      'size': self.size,
      'data': self.data,
    }
  
  def restore(self, state: Dict) -> None:
    # Yuan: need extra work to restore
    self.current_ind = state['current_ind']
    self.size = state['size']
    self.data = state['data']

  def sample(self, batch_size: int, sequence_length: int) -> Dict:
    """
    Sample a batch uniformly across environments and time steps

    Parameters
    ----------
    batch_size : int
    sequence_length : int

    Returns
    -------
    Dict
    """
    env_inds, sequence_starts = [], []
    base_sample_size = batch_size // self.num_envs
    remainder = batch_size % self.num_envs
    # print(self.good_starts[-1])
    for idx, curr_listset in enumerate(self.good_starts):
      num_elements = base_sample_size + (remainder > idx)
      if num_elements == 0:
        break
      sampled_elements = np.array(curr_listset.choice(num_elements))[:, np.newaxis]
      env_ind = np.full((num_elements, 1), idx)
      env_inds += [env_ind]
      sequence_starts += [sampled_elements]

    env_inds = np.concatenate(env_inds, 0)
    sequence_inds = (np.concatenate(sequence_starts, 0) + np.arange(sequence_length)) % self.capacity

    # env_inds = self.np_random.randint(0, self.num_envs, size=(batch_size, 1))

    # if self.size < self.capacity:
    #   # This requires special handling to avoid sampling from the empty part of the buffer. 
    #   # Once the buffer is full, we can sample to our heart's content
    #   buffer_starts = self.np_random.randint(
    #       0, self.size - sequence_length - 1, size=(batch_size, 1))
    #   sequence_inds = buffer_starts + np.arange(sequence_length)
    # else:
    #   buffer_starts = self.np_random.randint(
    #       0, self.size, size=(batch_size, 1))
    #   sequence_inds = buffer_starts + np.arange(sequence_length)
    #   sequence_inds = sequence_inds % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    batch = jax.tree.map(lambda x: np.swapaxes(
        x[sequence_inds, env_inds], 0, 1), self.data)

    return batch

class OrigSequentialReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               seed: Optional[int] = None,
               ):
    """
    Sequential replay buffer with support for parallel environments. 

    To simplify the implementation and speed up sampling, episode boundaries are NOT respected. i.e., the sampled subsequences may span multiple episodes. Any code using this buffer should handle this with termination/truncation signals

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store PER PARALLEL ENVIRONMENT
    dummy_input : Dict
        Example input from the environment. Used to determine the shape and dtype of the data to store
    num_envs : int, optional
        Number of parallel environments used for data collection, by default 1
    seed : Optional[int], optional
        Seed for sampling, by default None
    """
    self.capacity = capacity
    self.num_envs = num_envs
    self.data = jax.tree.map(lambda x: np.zeros(
        (capacity,) + np.asarray(x).shape, np.asarray(x).dtype), dummy_input)

    self.size = 0
    self.current_ind = 0

    self.np_random = np.random.RandomState(seed=seed)

  def __len__(self):
    return self.size

  def insert(self, data: Dict) -> None:
    # Insert the data
    jax.tree.map(lambda x, y: x.__setitem__(self.current_ind, y),
                 self.data, data)

    self.current_ind = (self.current_ind + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)
  
  def get_state(self) -> Dict:
    return {
      'current_ind': self.current_ind,
      'size': self.size,
      'data': self.data,
    }
  
  def restore(self, state: Dict) -> None:
    self.current_ind = state['current_ind']
    self.size = state['size']
    self.data = state['data']

  def sample(self, batch_size: int, sequence_length: int) -> Dict:
    """
    Sample a batch uniformly across environments and time steps

    Parameters
    ----------
    batch_size : int
    sequence_length : int

    Returns
    -------
    Dict
    """
    env_inds = self.np_random.randint(0, self.num_envs, size=(batch_size, 1))

    if self.size < self.capacity:
      # This requires special handling to avoid sampling from the empty part of the buffer. Once the buffer is full, we can sample to our heart's content
      buffer_starts = self.np_random.randint(
          0, self.size - sequence_length - 1, size=(batch_size, 1))
      sequence_inds = buffer_starts + np.arange(sequence_length)
    else:
      buffer_starts = self.np_random.randint(
          0, self.size, size=(batch_size, 1))
      sequence_inds = buffer_starts + np.arange(sequence_length)
      sequence_inds = sequence_inds % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    batch = jax.tree.map(lambda x: np.swapaxes(
        x[sequence_inds, env_inds], 0, 1), self.data)

    return batch

if __name__ == '__main__':
  # def make_env():
  #   def thunk():
  #     return gym.make('CartPole-v1')
  env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * 2)
  dummy_input = {'obs': env.observation_space.sample()}
  rb = OrigSequentialReplayBuffer(100, dummy_input, num_envs=2)

  obs, _ = env.reset()
  ep_count = np.zeros(env.num_envs, dtype=int)
  for i in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, _ = env.step(action)
    rb.insert({'obs': obs})
    if i > 3:
      print(rb.sample(2, 3)['obs'].shape)
