import pygame
import numpy as np
from maze_env import MazeEnv
from q_learning import QLearningAgent
from config import *


def visualize_best_path():
    obstacles = np.load(OBSTACLES_PATH, allow_pickle=True).tolist()
    env = MazeEnv(obstacles=obstacles)
    agent = QLearningAgent()
    agent.load_model()  # Load the best Q-table

    pygame.init()
    screen = pygame.display.set_mode((env.screen_size, env.screen_size))
    pygame.display.set_caption("Optimal Path Visualization")

    state = env.reset()
    done = False
    path = [tuple(state)]
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        state_idx = agent.get_state_index(state)
        action = np.argmax(agent.q_table[state_idx])

        next_state, reward, done = env.step(action)
        state = next_state
        path.append(tuple(state))
        total_reward += reward

        # 渲染界面，显示与训练时相同的信息
        env.render(
            screen,
            path=path,
            current_state=tuple(state),
            current_action=action,
            current_reward=reward,
            episode_info={
                'episode': "Best Model",
                'total_reward': total_reward,
                'epsilon': 0  # 测试阶段不需要探索率
            }
        )
        pygame.display.flip()
        pygame.time.delay(500)  # 延迟以方便观察

        if tuple(state) == END:
            pygame.draw.rect(screen, COLORS["path"],
                             (state[1] * 50 + 15, state[0] * 50 + 15, 20, 20))
            pygame.display.flip()
            break

    np.save(PATH_PATH, np.array(path))
    pygame.time.delay(2000)
    pygame.quit()
    print("Optimal Path Saved:", path)
    print("Total Reward:", total_reward)


if __name__ == "__main__":
    visualize_best_path()



    