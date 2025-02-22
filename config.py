

import numpy as np

# 迷宫配置
GRID_SIZE = 8
START = (0, 0)
END = (7, 7)
OBSTACLE_NUM = (5, 8)  # 随机障碍物数量范围

# Q-learning参数
LEARNING_RATE = 0.1
MAX_STEPS = 100
EPISODES = 1000  # Increased number of training rounds
DISCOUNT_FACTOR = 0.95  # Increasing the importance of long-term incentives
INITIAL_EPSILON = 1.0  # Initial exploration rate set to 1
EPSILON_DECAY = 0.001  # Slow down the rate of exploration decay
MIN_EPSILON = 0.01  # Added minimum exploration rate

# 颜色配置
COLORS = {
    "background": (255, 255, 255),  # 背景白色
    "grid": (0, 0, 0),  # 网格线黑色
    "start": (0, 255, 0),  # 起点绿色
    "end": (255, 215, 0),  # 终点金色
    "obstacle": (0, 0, 0),  # 障碍物黑色
    "agent": (0, 0, 255),  # 智能体蓝色
    "path": (255, 0, 0)  # 路径红色提高可见性
}

# 文件路径
Q_TABLE_PATH = "q_table.npy"
OBSTACLES_PATH = "obstacles.npy"
PATH_PATH = "best_path.npy"
