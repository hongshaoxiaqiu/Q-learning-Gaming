# Introduction
   This project is a maze navigation system based on Q-learning algorithm,
    
   which uses reinforcement learning to train an intelligent body to find the optimal path 
   from the starting point to the end point in a randomly generated maze. 
    
   The project includes maze environment construction, Q-learning algorithm implementation,
   training and testing scripts, and real-time visualisation.

# Game Objective.
  The main goal of the game is to move the agent from the starting point (START = (0, 0)) to the end point (END = (7, 7)). agent needs to reach the end point in as few steps as possible while avoiding obstacles. the success of the agent depends on its ability to learn how to avoid obstacles and find the optimal path during the training process.
 
# Environment setting 
 ## Python 3.0

 ## dependency：
 numpy, pygame

## install dependency: 
pip install numpy pygame

## training agent:
python train.py --episodes 1000 --render_every 50

## After training is complete, run the following command to see the optimal path:
python test.py

 # Project Structure

![image](https://github.com/user-attachments/assets/dabdefa3-a69e-4b3a-9a6a-35fac65dc142)

# maze environment
1.Configurable 8x8 grid world.

![image](https://github.com/user-attachments/assets/ff847b4a-97c4-493d-b394-8136f19f46e1)


2.Randomly generated obstacles (5-8).

![image](https://github.com/user-attachments/assets/300f2982-0265-4875-9ec7-230aa6a977e3)

3.Supports visualisation of start point, end point and obstacles.

# Q-learning algorithm：

![image](https://github.com/user-attachments/assets/faebdd2f-560c-4a6f-9243-3c6f21046c8e)

      def update(self, state, action, reward, next_state, done):
          state_idx = self.get_state_index(state)
          next_idx = self.get_state_index(next_state) if not done else None
      
          target = reward
          if not done:
              target += self.discount_factor * np.max(self.q_table[next_idx])
      
          self.q_table[state_idx, action] += self.learning_rate * (target - 
      self.q_table[state_idx, action])

# Action Selection Strategy (ε-greedy Policy):

![image](https://github.com/user-attachments/assets/7d1006fb-f149-45d1-8f6c-6154a3276e58)

          def choose_action(self, state):
              if np.random.rand() < self.epsilon: 
                  return np.random.randint(4)
              return np.argmax(self.q_table[state_idx])  
              
# Q-learning Algorithm Workflow

![image](https://github.com/user-attachments/assets/c40801a6-685f-4191-bb79-2674c63b9e6f)

# Reward Mechanism
### Default Reward:
The agent receives a small penalty (-1) for each step it takes. 

### Boundary Penalty:
If the agent tries to move outside the maze boundaries, it receives a penalty of -2 and does not move.

### Obstacle Penalty:
If the agent tries to move into an obstacle, it receives a penalty of -10 and does not move.

### Goal Reward:
If the agent reaches the goal (endpoint), it receives a large reward of +100, and the episode ends.

# The Best Model

![image](https://github.com/user-attachments/assets/c482edc6-d288-4094-b7b8-86a052e8b815)

![image](https://github.com/user-attachments/assets/a719e12d-236a-4204-a13d-2538291de3f9)





 
 
