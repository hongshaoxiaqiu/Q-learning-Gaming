# import pygame
# import numpy as np
# from config import *
#
# class MazeEnv:
#     def __init__(self, obstacles=None):
#         self.size = GRID_SIZE
#         self.start = START
#         self.end = END
#         self.obstacles = obstacles if obstacles is not None else self._generate_obstacles()
#         self.agent_pos = list(self.start)
#         self.cell_size = 50
#         self.screen_size = self.size * self.cell_size
#
#     def _generate_obstacles(self):
#         obstacles = []
#         num_obstacles = np.random.randint(*OBSTACLE_NUM)
#         while len(obstacles) < num_obstacles:
#             x, y = np.random.randint(0, self.size, 2)
#             pos = (x, y)
#             if pos != self.start and pos != self.end and pos not in obstacles:
#                 obstacles.append(pos)
#         return obstacles
#
#     def reset(self):
#         self.agent_pos = list(self.start)
#         return tuple(self.agent_pos)
#
#     def step(self, action):
#         dx, dy = 0, 0
#         if action == 0:  #  up
#             dx = -1
#         elif action == 1:  # down
#             dx = 1
#         elif action == 2:  # left
#             dy = -1
#         elif action == 3:  # right
#             dy = 1
#
#         new_x = self.agent_pos[0] + dx
#         new_y = self.agent_pos[1] + dy
#         reward = -1
#         done = False
#
#         # Boundary check
#         if not (0 <= new_x < self.size and 0 <= new_y < self.size):
#             return tuple(self.agent_pos), -2, False
#
#         # Obstacle checking
#         if (new_x, new_y) in self.obstacles:
#             return tuple(self.agent_pos), -10, False
#
#         self.agent_pos = [new_x, new_y]
#
#         # endpoint inspection
#         if (new_x, new_y) == self.end:
#             reward = 100
#             done = True
#
#         return tuple(self.agent_pos), reward, done
#
#     def render(self, screen, path=[]):  # 增加 path 参数
#         screen.fill(COLORS["background"])
#
#         # 绘制网格线
#         for i in range(self.size + 1):
#             pygame.draw.line(screen, COLORS["grid"], (0, i * self.cell_size),
#                              (self.screen_size, i * self.cell_size))
#             pygame.draw.line(screen, COLORS["grid"], (i * self.cell_size, 0),
#                              (i * self.cell_size, self.screen_size))
#
#         # 绘制起点终点
#         pygame.draw.rect(screen, COLORS["start"],
#                          (START[1] * self.cell_size, START[0] * self.cell_size,
#                           self.cell_size, self.cell_size))
#         pygame.draw.rect(screen, COLORS["end"],
#                          (END[1] * self.cell_size, END[0] * self.cell_size,
#                           self.cell_size, self.cell_size))
#
#         # 绘制障碍物
#         for x, y in self.obstacles:
#             pygame.draw.rect(screen, COLORS["obstacle"],
#                              (y * self.cell_size, x * self.cell_size,
#                               self.cell_size, self.cell_size))
#
#         # 绘制路径
#         for (x, y) in path:
#             pygame.draw.rect(screen, COLORS["path"],
#                              (y * self.cell_size, x * self.cell_size,
#                               self.cell_size, self.cell_size))
#
#         # 绘制智能体
#         x, y = self.agent_pos
#         pygame.draw.rect(screen, COLORS["agent"],
#                          (y * self.cell_size + 5, x * self.cell_size + 5,
#                           self.cell_size - 10, self.cell_size - 10))
#
import pygame
import numpy as np
from config import *


class MazeEnv:
    def __init__(self, obstacles=None):
        self.size = GRID_SIZE
        self.start = START
        self.end = END
        self.obstacles = obstacles if obstacles is not None else self._generate_obstacles()
        self.agent_pos = list(self.start)
        self.cell_size = 50
        self.screen_size = self.size * self.cell_size

    def _generate_obstacles(self):
        obstacles = []
        num_obstacles = np.random.randint(*OBSTACLE_NUM)
        while len(obstacles) < num_obstacles:
            x, y = np.random.randint(0, self.size, 2)
            pos = (x, y)
            if pos != self.start and pos != self.end and pos not in obstacles:
                obstacles.append(pos)
        return obstacles

    def reset(self):
        self.agent_pos = list(self.start)
        return tuple(self.agent_pos)

    def step(self, action):
        dx, dy = 0, 0
        if action == 0:  # up
            dx = -1
        elif action == 1:  # down
            dx = 1
        elif action == 2:  # left
            dy = -1
        elif action == 3:  # right
            dy = 1

        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        reward = -1
        done = False

        # Boundary check
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            return tuple(self.agent_pos), -2, False

        # Obstacle checking
        if (new_x, new_y) in self.obstacles:
            return tuple(self.agent_pos), -10, False

        self.agent_pos = [new_x, new_y]

        # endpoint inspection
        if (new_x, new_y) == self.end:
            reward = 100
            done = True

        return tuple(self.agent_pos), reward, done

    def render(self, screen, path=[], current_state=None, current_action=None, current_reward=None, episode_info=None):
        screen.fill(COLORS["background"])

        # 绘制网格线
        for i in range(self.size + 1):
            pygame.draw.line(screen, COLORS["grid"], (0, i * self.cell_size),
                             (self.screen_size, i * self.cell_size))
            pygame.draw.line(screen, COLORS["grid"], (i * self.cell_size, 0),
                             (i * self.cell_size, self.screen_size))

        # 绘制起点终点
        pygame.draw.rect(screen, COLORS["start"],
                         (START[1] * self.cell_size, START[0] * self.cell_size,
                          self.cell_size, self.cell_size))
        pygame.draw.rect(screen, COLORS["end"],
                         (END[1] * self.cell_size, END[0] * self.cell_size,
                          self.cell_size, self.cell_size))

        # 绘制障碍物
        for x, y in self.obstacles:
            pygame.draw.rect(screen, COLORS["obstacle"],
                             (y * self.cell_size, x * self.cell_size,
                              self.cell_size, self.cell_size))

        # 绘制路径
        for (x, y) in path:
            pygame.draw.rect(screen, COLORS["path"],
                             (y * self.cell_size, x * self.cell_size,
                              self.cell_size, self.cell_size))

        # 绘制智能体
        x, y = self.agent_pos
        pygame.draw.rect(screen, COLORS["agent"],
                         (y * self.cell_size + 5, x * self.cell_size + 5,
                          self.cell_size - 10, self.cell_size - 10))

        # 新增信息显示区域
        font = pygame.font.SysFont(None, 24)
        info_y = 10  # 信息显示区域的起始Y坐标

        # 绘制训练信息标题
        # title_text = "Training Status:"
        # text_surface = font.render(title_text, True, (0, 0, 0))
        # screen.blit(text_surface, (10, info_y))
        # info_y += 30  # 增加行间距

        if episode_info:
            episode_text = f"Episode: {episode_info.get('episode', '')}"
            total_reward_text = f"Total Reward: {episode_info.get('total_reward', '')}"
            epsilon_text = f"Exploration Rate: {episode_info.get('epsilon', 0):.2f}"

            for text in [episode_text, total_reward_text, epsilon_text]:
                text_surface = font.render(text, True, (0, 0, 0))
                screen.blit(text_surface, (10, info_y))
                info_y += 20

        if current_state is not None:
            state_text = f"State: {current_state}"
            text_surface = font.render(state_text, True, (0, 0, 0))
            screen.blit(text_surface, (10, info_y))
            info_y += 20

        # if current_action is not None:
        #     action_names = ['↑', '↓', '←', '→']
        #     action_text = f"Action: {action_names[current_action]}"
        #     text_surface = font.render(action_text, True, (0, 0, 0))
        #     screen.blit(text_surface, (10, info_y))
        #     info_y += 20

        if current_reward is not None:
            reward_text = f"Reward: {current_reward}"
            text_surface = font.render(reward_text, True, (0, 0, 0))
            screen.blit(text_surface, (10, info_y))