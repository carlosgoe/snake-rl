import numpy as np
import random
import math
from time import sleep
from IPython.display import clear_output


class Snake:

    def __init__(self, size=(16, 16), gui=False):
        self.size = size
        self.gui = gui
        self.snake = [(size[0] // 2, 3), (size[0] // 2, 2), (size[0] // 2, 1)]
        self.positions = [(x, y) for x in range(size[0]) for y in range(size[1])]
        self.direction = (0, 1)
        self.food = random.choice([p for p in self.positions if p not in self.snake])
        self.points = 0
        self.game_over = False
        self.visualize()

    def visualize(self):
        if self.gui:
            field = np.zeros(self.size)
            field[self.food] = 3
            field[np.array(self.snake)[:, 0], np.array(self.snake)[:, 1]] = 1
            field[self.snake[0]] = 2
            replacements = {'2.': '◈', '0.': '▢', '1.': '▩', '3.': '◉', '[': '', ']': ''}
            output = str(field)
            for key, val in replacements.items():
                output = output.replace(key, val)
            clear_output(wait=True)
            print('Points: {0}\n\n {1}'.format(self.points, output))
            sleep(0.1)
            if self.game_over:
                sleep(1)
            else:
                sleep(0.1)
                
    def get_vector(self, direct):
        # Vectors to select from
        vecs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # Right
        if direct == 0:
            return vecs[(vecs.index(self.direction) + 1) % 4]
        # Left
        if direct == 1:
            return vecs[(vecs.index(self.direction) - 1) % 4]
        # Forward
        return self.direction

    def obs_and_invalid(self):
        # Get angle to food
        dir_food = tuple(np.array(self.food) - np.array(self.snake[0]))
        angle = math.acos((self.direction[0] * dir_food[0] + self.direction[1] * dir_food[1]) / math.sqrt(dir_food[0] ** 2 + dir_food[1] ** 2)) / math.pi
        # Get positions right, left, and in front of snake
        positions = [tuple(np.array(self.snake[0]) + np.array(self.get_vector(i))) for i in range(3)]
        # Get booleans that indicate whether there is an obstacle right, left, and in front of snake
        obstacles = [float(p not in self.positions or p in self.snake[:-1]) for p in positions]
        # Get invalid actions if there are 1 or 2
        invalid = np.argwhere(np.array(obstacles) == 1).flatten().tolist() if 0 < np.sum(obstacles) < 3 else []
        # Return observation (as numpy array) and invalid actions
        return np.array(obstacles + [angle]), invalid

    def step(self, action):
        # Get new movement direction of snake
        vec = self.get_vector(action)
        # Calculate new position of snake's head
        new_pos = tuple(np.array(self.snake[0]) + np.array(vec))
        # Check if new position is within the borders and not part of snake (else: game over and reward -2)
        if new_pos not in self.positions or new_pos in self.snake[:-1]:
            self.game_over = True
            reward = -2
        else:
            # Calculate the current and the new distance to food
            d_food_prev = np.sum(np.abs(np.array(self.food) - np.array(self.snake[0])))
            d_food_new = np.sum(np.abs(np.array(self.food) - np.array(new_pos)))
            # Apply move to environment
            self.snake = [new_pos] + self.snake
            last = self.snake[-1]
            self.snake = self.snake[:-1]
            self.direction = vec
            # Check if food is reached and increase length of snake
            if new_pos == self.food:
                self.snake.append(last)
                self.points += 1
                self.food = random.choice([p for p in self.positions if p not in self.snake])
            # reward of 1 if food is reached or distance to food has decreased, else -1
            reward = 1 if d_food_new < d_food_prev else -1
        # Visualize environment (only if gui is enabled)
        self.visualize()
        # Get current observation and invalid actions
        obs, invalid = self.obs_and_invalid()
        return obs, reward, self.game_over, invalid
