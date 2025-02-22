import numpy as np
from config import *

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((GRID_SIZE ** 2, 4))  # Initialize Q-table
        self.best_q_table = None  # To store the best Q-table
        self.best_reward = -np.inf  # Best reward
        self.learning_rate = LEARNING_RATE  # Learning rate
        self.discount_factor = DISCOUNT_FACTOR  # Discount factor
        self.epsilon = INITIAL_EPSILON  # Initial exploration rate
        self.min_epsilon = MIN_EPSILON  # Minimum epsilon

    def get_state_index(self, state):
        """
        Convert state (x, y) to Q-table index
        """
        return state[0] * GRID_SIZE + state[1]

    def choose_action(self, state):
        """
        Choose action based on current state
        """
        state_idx = self.get_state_index(state)
        if np.random.rand() < self.epsilon:  # Exploration
            return np.random.randint(4)
        return np.argmax(self.q_table[state_idx])  # Exploitation

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table
        """
        state_idx = self.get_state_index(state)
        next_idx = self.get_state_index(next_state) if not done else None

        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[next_idx])

        self.q_table[state_idx, action] += self.learning_rate * (target - self.q_table[state_idx, action])

    def decay_epsilon(self):
        """
        Decay epsilon for exploration
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * (1 - EPSILON_DECAY))

    def update_best_model(self, episode_reward):
        """
        Update best Q-table if current episode has better reward
        """
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_q_table = self.q_table.copy()

    def save_model(self):
        """
        Save best Q-table to file
        """
        if self.best_q_table is not None:
            np.save(Q_TABLE_PATH, self.best_q_table)
        else:
            np.save(Q_TABLE_PATH, self.q_table)

    def load_model(self):
        """
        Load Q-table from file
        """
        try:
            self.q_table = np.load(Q_TABLE_PATH)
            self.best_q_table = self.q_table.copy()
        except FileNotFoundError:
            print("Q-table not found. Please train the model first.")
            exit(1)

    def set_epsilon(self, epsilon):
        """
        Set exploration rate epsilon
        """
        self.epsilon = epsilon

