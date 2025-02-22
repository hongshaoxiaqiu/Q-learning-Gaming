import pygame
from maze_env import MazeEnv
from q_learning import QLearningAgent
from config import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Maximum steps per episode")
    parser.add_argument("--render_every", type=int, default=1, help="Frequency of rendering (episodes)")
    return parser.parse_args()

def train():
    args = parse_args()
    env = MazeEnv()
    agent = QLearningAgent()

    # Save initial obstacles
    np.save(OBSTACLES_PATH, env.obstacles)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((env.screen_size, env.screen_size))
    pygame.display.set_caption("Training...")

    best_reward = -np.inf
    best_path = None

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        path = [tuple(state)]

        while step < args.max_steps and not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step += 1
            path.append(tuple(state))

            if episode % args.render_every == 0:
                env.render(
                    screen,
                    current_state=tuple(state),
                    current_action=action,
                    current_reward=reward,
                    episode_info={
                        'episode': episode + 1,
                        'total_reward': total_reward,
                        'epsilon': agent.epsilon
                    }
                )
                pygame.display.flip()

            pygame.time.delay(100)

        if total_reward > best_reward:
            best_reward = total_reward
            best_path = path

        agent.decay_epsilon()
        print(f"Episode: {episode + 1}/{args.episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    agent.save_model()
    np.save(PATH_PATH, best_path)
    pygame.quit()
    print("Training completed. Q-table, obstacles, and best path saved.")

if __name__ == "__main__":
    train()