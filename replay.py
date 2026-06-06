"""
Prioritized Experience Replay buffer using a Sum Tree.
"""


import numpy as np




class SumTree:
   """
   Binary tree where each leaf stores a priority and
   internal nodes store the sum of their children.
   Allows O(log n) sampling proportional to priority.
   """


   def __init__(self, capacity):
       """
       Args:
           capacity: int, max number of transitions
       """
       self.capacity = capacity
       # tree[0] is root; leaves occupy indices [capacity-1, 2*capacity-2]
       self.tree = np.zeros(2 * capacity - 1)
       self.data = np.zeros(capacity, dtype=object)
       self.write_index = 0
       self.size = 0


   def total(self):
       """Return the total sum of all priorities."""
       return self.tree[0]


   def add(self, priority, data):
       """
       Store a transition with the given priority.
       Args:
           priority: float
           data: tuple (state, action, reward, next_state, done)
       """
       leaf_index = self.write_index + self.capacity - 1
       self.data[self.write_index] = data
       self.update(leaf_index, priority)


       self.write_index = (self.write_index + 1) % self.capacity
       self.size = min(self.size + 1, self.capacity)


   def update(self, tree_index, priority):
       """
       Update the priority of a leaf node and propagate.
       Args:
           tree_index: int, index in the tree array
           priority: float, new priority
       """
       delta = priority - self.tree[tree_index]
       self.tree[tree_index] = priority
       # Propagate delta up to root
       while tree_index != 0:
           tree_index = (tree_index - 1) // 2
           self.tree[tree_index] += delta


   def get(self, value):
       """
       Retrieve a leaf by sampling a value in [0, total()).
       Args:
           value: float
       Returns:
           (tree_index, priority, data)
       """
       index = 0
       while index < self.capacity - 1:
           left = 2 * index + 1
           right = left + 1
           if value <= self.tree[left]:
               index = left
           else:
               value -= self.tree[left]
               index = right
       data_index = index - (self.capacity - 1)
       return index, self.tree[index], self.data[data_index]




class PrioritizedReplayBuffer:
   """
   Replay buffer that samples transitions proportional
   to their TD-error priority.
   """


   def __init__(self, capacity, alpha=0.6, beta_start=0.4,
                beta_end=1.0, beta_steps=200_000, epsilon=1e-6):
       """
       Args:
           capacity: int
           alpha: float, how much prioritisation to use (0 = uniform)
           beta_start: float, initial importance-sampling exponent
           beta_end: float, final beta value
           beta_steps: int, steps over which beta is annealed
           epsilon: float, small constant to avoid zero priorities
       """
       self.tree = SumTree(capacity)
       self.alpha = alpha
       self.beta_start = beta_start
       self.beta_end = beta_end
       self.beta_steps = beta_steps
       self.epsilon = epsilon
       self.max_priority = 1.0
       self.step_count = 0


   def _get_beta(self):
       """Linearly anneal beta from beta_start to beta_end."""
       fraction = min(self.step_count / self.beta_steps, 1.0)
       return self.beta_start + fraction * (self.beta_end - self.beta_start)


   def store(self, state, action, reward, next_state, done):
       """
       Store a transition with max priority.
       Args:
           state:      np.ndarray
           action:     int
           reward:     float
           next_state: np.ndarray
           done:       bool
       """
       # New transitions get max priority so they are sampled at least once
       priority = self.max_priority ** self.alpha
       self.tree.add(priority, (state, action, reward, next_state, done))


   def sample(self, batch_size):
       """
       Sample a batch proportional to priorities.
       Args:
           batch_size: int
       Returns:
           (states, actions, rewards, next_states, dones,
            indices, is_weights)
       """
       indices = []
       priorities = []
       transitions = []


       # Divide total priority range into batch_size equal segments
       segment = self.tree.total() / batch_size
       beta = self._get_beta()
       self.step_count += 1


       for i in range(batch_size):
           low, high = segment * i, segment * (i + 1)
           value = np.random.uniform(low, high)
           tree_idx, priority, data = self.tree.get(value)
           indices.append(tree_idx)
           priorities.append(priority)
           transitions.append(data)


       # Importance-sampling weights to correct for sampling bias
       priorities = np.array(priorities, dtype=np.float32)
       sampling_probs = priorities / self.tree.total()
       is_weights = (self.tree.size * sampling_probs) ** (-beta)
       is_weights /= is_weights.max()  # normalise so max weight = 1


       states, actions, rewards, next_states, dones = zip(*transitions)
       return (
           np.array(states,      dtype=np.float32),
           np.array(actions,     dtype=np.int64),
           np.array(rewards,     dtype=np.float32),
           np.array(next_states, dtype=np.float32),
           np.array(dones,       dtype=np.float32),
           indices,
           is_weights,
       )


   def update_priorities(self, indices, td_errors):
       """
       Update priorities after learning.
       Args:
           indices:   list of tree indices
           td_errors: np.ndarray of |TD error| values
       """
       for idx, error in zip(indices, td_errors):
           priority = (abs(error) + self.epsilon) ** self.alpha
           self.tree.update(idx, priority)
           self.max_priority = max(self.max_priority, priority)


   def __len__(self):
       return self.tree.size
