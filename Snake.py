import pygame
import time
import random

import tensorflow as tf
from tensorflow import keras

import numpy as np
import math

class Snake():
    def __init__(self, width, height):
        self.nn = self.create_model()
        self.snake_list = []
        self.length_of_snake = 1
        self.is_dead = False
        self.width = width
        self.height = height
        self.x = width // 2
        self.y = height // 2
        self.x_change = 0
        self.y_change = 0

        self.fitness = 0

        self.food_x = round(random.randrange(0, self.width - 10) / 10.0) * 10.0
        self.food_y = round(random.randrange(0, self.height - 10) / 10.0) * 10.0


    # create brain of snake
    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation='relu'),  # input layer
            keras.layers.Dense(20, activation='relu'),  # hidden layer 1
            keras.layers.Dense(12, activation='relu'),  # hidden layer 2
            keras.layers.Dense(4, activation='softmax') # output layer
        ])

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        return model
    
    # looks in 8 directions
    # each directions it looks for 
    #   1) distance to the food
    #   2) distance to the wall
    #   3) distance to itself
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
    # returns index of max probability [up, down, left, right]
    # up = 0
    # down = 1
    # left = 2
    # right = 3
    def get_next_move(self, snake_views):
        input_data = np.asarray(snake_views)
        input_data = np.atleast_2d(input_data)

        prediction = self.nn.predict(input_data)[0]
        return np.argmax(prediction)

    def get_snake_views(self):
        def distance(a, b):
            return math.sqrt( ((a[0]-b[0])**2)+((a[1]-b[1])**2) )

        #[food_dist, wall_dist, self_dist]
        def get_top_left_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, x+10, 10):
                current_search_pos = [x-i, y-i]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y], [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        def get_top_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, y + 10, 10):
                current_search_pos = [x, y-i]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y],  current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y], [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        def get_top_right_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, self.width - x + 10, 10):
                current_search_pos = [x+i, y-i]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y],  current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y],  [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        def get_right_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, self.width - x + 10, 10):
                current_search_pos = [x+i, y]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y], [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        def get_bottom_right_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, self.width - x + 10, 10):
                current_search_pos = [x+i, y+i]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y], [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        def get_bottom_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, self.height - y + 10, 10):
                current_search_pos = [x, y+i]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y], [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        def get_bottom_left_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, x + 10, 10):
                current_search_pos = [x-i, y+i]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y], [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        def get_left_view(x, y):
            food_dist = -1
            wall_dist = -1
            self_dist = -1
            for i in range(0, x + 10, 10):
                current_search_pos = [x-i, y]
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if current_search_pos[0] <= 0 or current_search_pos[0] >= self.width or current_search_pos[1] >= self.height or current_search_pos[1] <= 0:
                    wall_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    if self_dist == -1:
                        self_dist = distance([x, y], [current_search_pos])

            return [food_dist, wall_dist, self_dist]

        return [get_top_left_view(self.x, self.y), get_top_view(self.x, self.y), get_top_right_view(self.x, self.y),
                get_left_view(self.x, self.y), get_right_view(self.x, self.y),
                get_bottom_left_view(self.x, self.y), get_bottom_view(self.x, self.y), get_bottom_right_view(self.x, self.y)]

    def get_fitness(self):
        if self.is_dead:
            return self.length_of_snake - 1 + self.fitness
        else:
            return self.length_of_snake + self.fitness

    def is_colliding(self):
        return self.x >= self.width or self.x <= 0 or self.y >= self.height or self.y <= 0

    def is_eating_self(self):
        for pos in self.snake_list[:-1]:
            if pos[0] == self.x and pos[1] == self.y:
                self.is_dead = True
    
    def update_movement(self):
        self.x += self.x_change
        self.y += self.y_change
        
    def reset(self):
        self.snake_list = []
        self.length_of_snake = 1
        self.is_dead = False
        self.x = self.width // 2
        self.y = self.height // 2
        self.x_change = 0
        self.y_change = 0
        self.fitness = 0

        self.food_x = round(random.randrange(0, self.width - 10) / 10.0) * 10.0
        self.food_y = round(random.randrange(0, self.height - 10) / 10.0) * 10.0
    
class SnakeGame():
    def __init__(self, width, height, snake_speed, population, generations, life_span):
        self.width = width
        self.height = height
        self.snake_speed = snake_speed
        self.population = population
        self.snake_list = self.snake_list_init()

        self.generations = generations
        self.life_span = life_span

        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))

        self.x_change = 0
        self.y_change = 0

    def snake_list_init(self):
        snake_list = []
        for _ in range(self.population):
            snake_list.append(Snake(self.width, self.height))
        return snake_list

    def run(self, current_generation, max_score):
        dead_snake_count = 0
        start_ticks = pygame.time.get_ticks()
        seconds = ((pygame.time.get_ticks() - start_ticks) / 1000)
        timer = start_ticks

        while seconds < self.life_span:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()

            dead_snake_count = 0
            for snake in self.snake_list:
                if snake.is_dead:
                    dead_snake_count += 1

            if dead_snake_count >= len(self.snake_list):
                break

            # update snake movements
            for snake in self.snake_list:
                if snake.is_dead:
                    continue

                if snake.is_colliding():
                    snake.is_dead = True
                snake.is_eating_self()


                next_move = snake.get_next_move(snake.get_snake_views())
                if next_move == 0: # up
                    if snake.y_change == 10:
                        snake.is_dead = True
                        continue
                    snake.x_change = 0
                    snake.y_change = -10
                elif next_move == 1: # down
                    if snake.y_change == -10:
                        snake.is_dead = True
                        continue
                    snake.x_change = 0
                    snake.y_change = 10
                elif next_move == 2: # left
                    if snake.x_change == 10:
                        snake.is_dead = True
                        continue
                    snake.x_change = -10
                    snake.y_change = 0
                elif next_move == 3: # right
                    if snake.x_change == -10:
                        snake.is_dead = True
                        continue
                    snake.x_change = 10
                    snake.y_change = 0
                snake.update_movement()

                snake.snake_list.append([snake.x, snake.y])

                # Remove last cube of the snake if we haven't eaten food
                # length_of_snake is true reflection of current length
                # snake_list should only have the number of cubes being displayed so we delete trailing one since we move forward
                if len(snake.snake_list) > snake.length_of_snake:
                    del snake.snake_list[0]

            # update score 
            if pygame.time.get_ticks()-timer > 500:
                timer = pygame.time.get_ticks()
                for snake in self.snake_list:
                    snake.fitness += 1
                #do something every 1.5 seconds

            # draw snake
            # reset everything to black before drawing
            self.display.fill((0, 0, 0))
            for snake in self.snake_list:
                # draw snake
                for pos in snake.snake_list:
                    if not snake.is_dead:
                        pygame.draw.rect(self.display, (255, 0, 0), [pos[0], pos[1], 10, 10])

                # draw food
                pygame.draw.rect(self.display, (255, 0, 255), [snake.food_x, snake.food_y, 10, 10])

            value = pygame.font.SysFont("comicsansms", 35).render("Generation: " + str(current_generation), True, (255,0,0))
            self.display.blit(value, [0, 0])
            value = pygame.font.SysFont("comicsansms", 35).render("Max Score: " + str(max_score), True, (255,0,0))
            self.display.blit(value, [0, 60])

            pygame.display.update()

            #handle snake eating food
            for snake in self.snake_list:
                # Eat food 
                if snake.x == snake.food_x and snake.y == snake.food_y:
                    snake.food_x = round(random.randrange(0, snake.width - 10) / 10.0) * 10.0
                    snake.food_y = round(random.randrange(0, snake.height - 10) / 10.0) * 10.0

                    # increase length of snake
                    snake.length_of_snake += 1

            pygame.time.Clock().tick(self.snake_speed)
            seconds = ((pygame.time.get_ticks() - start_ticks) / 1000)

    def start(self):
        max_score = 0
        for generation_number in range(self.generations):
            self.run(generation_number, max_score)

            # add up total score of generation for weights
            total_score = 0
            for snake in self.snake_list:
                fitness = snake.get_fitness()
                if fitness > max_score:
                    max_score = fitness

                total_score += fitness

            if total_score == 0:
                snake_weights = [(1.0/len(self.snake_list))] * len(self.snake_list)
            else:
                snake_weights = []
                for snake in self.snake_list:
                    snake_weights.append(snake.get_fitness()/total_score)

            new_weights_list = []
            # crossover & mutate
            for i in range(self.population):
                # select 2 parents
                snake1_index = np.random.choice(range(len(self.snake_list)), p=snake_weights)
                snake2_index = np.random.choice(range(len(self.snake_list)), p=snake_weights)

                # crossover those 2 parents
                new_weights = self.crosover(snake1_index, snake2_index)

                # mutate that new child
                mutated_new_weights = self.mutate(new_weights[0])

                new_weights_list.append(mutated_new_weights)

                # reset snakes
                self.snake_list[i].reset()


            # update population weights with new children
            for i in range(len(new_weights_list)):
                self.snake_list[i].nn.set_weights(new_weights_list[i])


    # returns list of new weights
    def crosover(self, snake1_index, snake2_index):
        parent_weight1 = self.snake_list[snake1_index].nn.get_weights()
        parent_weight2 = self.snake_list[snake2_index].nn.get_weights()

        new_weight1 = parent_weight1
        new_weight2 = parent_weight2

        gene = random.randint(0, len(new_weight1)-1)

        new_weight1[gene] = parent_weight2[gene]
        new_weight2[gene] = parent_weight1[gene]

        return np.asarray([new_weight1, new_weight2])

    def mutate(self, weights):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if random.uniform(0,1) > .90:
                    change = random.uniform(-0.5, 0.5)
                    weights[i][j] += change
        return weights

    def quit(self):
        pygame.quit()
        quit()
    

game_speed = 50
population = 500
number_of_generations = 5000
life_span_per_generation = 30

snake_game = SnakeGame(500, 500, game_speed, population, number_of_generations, life_span_per_generation)
snake_game.start()