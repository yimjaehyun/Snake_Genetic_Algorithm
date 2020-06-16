import pygame
import time
import random

class SnakeGame():
    def __init__(self, width, height, snake_speed):
        self.width = width
        self.height = height
        self.snake_speed = snake_speed

        self.snake_list = []
        self.length_of_snake = 1

        self.food_x = round(random.randrange(0, self.width - 10) / 10.0) * 10.0
        self.food_y = round(random.randrange(0, self.height - 10) / 10.0) * 10.0

        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))

        self.x = self.width // 2
        self.y = self.height // 2

        self.game_over = False

        self.x_change = 0
        self.y_change = 0

    def run(self):
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        if self.x_change != 10:
                            self.x_change = -10
                            self.y_change = 0
                    elif event.key == pygame.K_RIGHT:
                        if self.x_change != -10:
                            self.x_change = 10
                            self.y_change = 0
                    elif event.key == pygame.K_UP:
                        if self.y_change != 10:
                            self.y_change = -10
                            self.x_change = 0
                    elif event.key == pygame.K_DOWN:
                        if self.y_change != -10:
                            self.y_change = 10
                            self.x_change = 0

            if self.x >= self.width or self.x <= 0 or self.y >= self.height or self.y <= 0:
                self.game_over = True

            self.x += self.x_change
            self.y += self.y_change

            self.snake_list.append([self.x, self.y])

            # Remove last cube of the snake if we haven't eaten food
            # length_of_snake is true reflection of current length
            # snake_list should only have the number of cubes being displayed so we delete trailing one since we move forward
            if len(self.snake_list) > self.length_of_snake:
                del self.snake_list[0]

            # reset everything to black before drawing
            self.display.fill((0, 0, 0))

            # draw snake
            for pos in self.snake_list:
                pygame.draw.rect(self.display, (255, 0, 0), [pos[0], pos[1], 10, 10])

            # draw food
            pygame.draw.rect(self.display, (255, 0, 255), [self.food_x, self.food_y, 10, 10])

            value = pygame.font.SysFont("comicsansms", 35).render("Score: " + str(self.length_of_snake - 1), True, (255,0,0))
            self.display.blit(value, [0, 0])

            pygame.display.update()

            # Eat food 
            if self.x == self.food_x and self.y == self.food_y:
                self.food_x = round(random.randrange(0, self.width - 10) / 10.0) * 10.0
                self.food_y = round(random.randrange(0, self.height - 10) / 10.0) * 10.0

                # increase length of snake
                self.length_of_snake += 1

            # check collision with self
            for pos in self.snake_list[:-1]:
                if pos[0] == self.x and pos[1] == self.y:
                    self.game_over = True

            pygame.time.Clock().tick(self.snake_speed)

        pygame.quit()
        quit()
    

# SnakeGame(1000,1000, 30).run()


import tensorflow as tf
from tensorflow import keras

import numpy as numpyimport 
import math


current_pool = []
n = 50

def initialize():
    for _ in range(n):
        model = keras.Sequential([
            keras.layers.Dense(24, activation='relu'),  # input layer
            keras.layers.Dense(18, activation='relu'),  # hidden layer 1
            keras.layers.Dense(18, activation='relu'),  # hidden layer 2
            keras.layers.Dense(4, activation='softmax') # output layer
        ])

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        current_pool.append(model)

# looks in 8 directions
# each directions it looks for 
#   1) distance to the food
#   2) distance to itself
#   3) distance to the wall
#             \   |   /
#              \  |  /
#         - - - (x,y) - - -
#              /  |  \
#             /   |   \
# input type:
#  [[#,#,#], [#,#,#], [#,#,#], 
#   [#,#,#], [#,#,#], [#,#,#],
#   [#,#,#], [#,#,#], [#,#,#]] 
#
def predict_action(snake_views, model_index):
    input_data = np.asarray(snake_views)
    input_data = np.atleast_2d(input_data)

    output_probability = current_pool[model_index].predict(input_data)[0]
    return output_probability
    

def fitness(snake):
    pass

def crosover(snake1_index, snake2_index):
    parent_weight1 = current_pool[snake1_index].get_weights()
    parent_weight2 = current_pool[snake2_index].get_weights()

    new_weight1 = parent_weight1
    new_weight2 = parent_weight2

    gene = random.randomint(0, len(new_weight1))

    new_weight1[gene] = parent_weight2[gene]
    new_weight2[gene] = parent_weight1[gene]

    return np.asarray([new_weight1, new_weight2])

def mutate(weights):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0,1) > .90:
                change = random.uniform(-0.5, 0.5)
                weights[i][j] += change
     return weights

def main(number_of_iterations):
    # Generate population

