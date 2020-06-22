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
        self.life_span = 0

        self.fitness = 0

        self.food_x = round(random.randrange(0, self.width - 10) / 10.0) * 10.0
        self.food_y = round(random.randrange(0, self.height - 10) / 10.0) * 10.0


    # create brain of snake
    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation='relu'),  # input layer
            keras.layers.Dense(18, activation='relu'),  # hidden layer 1
            keras.layers.Dense(18, activation='relu'),  # hidden layer 2
            keras.layers.Dense(4, activation='sigmoid') # output layer
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
        input_data = np.asarray(snake_views).flatten()
        input_data = np.atleast_2d(input_data)


        prediction = self.nn.predict(input_data)[0]
        x = ["up", "down", "left", "right"]
        # print(x[np.argmax(prediction)])
        # print(prediction)
        return np.argmax(prediction)

    def get_snake_views(self):
        def distance(a, b):
            return math.sqrt( ((a[0]-b[0])**2)+((a[1]-b[1])**2) ) / self.height

        def diagonal_distance(a,b):
            return math.sqrt( ((a[0]-b[0])**2)+((a[1]-b[1])**2) ) / math.sqrt(self.height ** 2 + self.width ** 2)

        def get_view(x, y, x_add_value, y_add_value):
            food_dist = 1
            wall_dist = 1
            self_dist = 1
            current_search_pos = [x+x_add_value, y+y_add_value]
            while current_search_pos[0] > 0 and current_search_pos[0] < self.width and current_search_pos[1] > 0 and current_search_pos[1] < self.height:
                if current_search_pos[0] == self.food_x and current_search_pos[1] == self.food_y:
                    food_dist = distance([x, y], current_search_pos)
                if (current_search_pos[0], current_search_pos[1]) in self.snake_list:
                    self_dist = distance([x, y], [current_search_pos])

                current_search_pos[0] += x_add_value
                current_search_pos[1] += y_add_value

            #print(current_search_pos, self.x, self.y)
            wall_dist = distance([x, y], current_search_pos)

            return [food_dist, wall_dist, self_dist]



        #endregion
        
        x =  [get_view(self.x, self.y, -10, -10), get_view(self.x, self.y, 0, -10), get_view(self.x, self.y, 10, -10),
                get_view(self.x, self.y, -10, 0), get_view(self.x, self.y, 10, 0),
                get_view(self.x, self.y, -10, 10), get_view(self.x, self.y, 0, 10), get_view(self.x, self.y, 10, 10)]

        # x = [[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)],[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)],[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)],[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)],[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)],[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)],[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)],[random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)]]
        # print("top_left:",x[0][1], " top:", x[1][1], " top_right:", x[2][1], " left", x[3][1], " right", x[4][1], " bottom_left", x[5][1], " bottom", x[6][1], " bottom_right", x[7][1])
        print("top_left:",x[0][0], " top:", x[1][0], " top_right:", x[2][0], " left", x[3][0], " right", x[4][0], " bottom_left", x[5][0], " bottom", x[6][0], " bottom_right", x[7][0])
        #print("top_left:",x[0][2], " top:", x[1][2], " top_right:", x[2][2], " left", x[3][2], " right", x[4][2], " bottom_left", x[5][2], " bottom", x[6][2], " bottom_right", x[7][2])
        return x

    def get_fitness(self):
        return self.life_span + (((2 ** self.length_of_snake) + (self.length_of_snake**2.1) * 500) - ((self.length_of_snake**1.2) * (0.25 * self.length_of_snake) ** 1.3))

    def is_colliding(self):
        return self.x >= self.width or self.x <= 0 or self.y >= self.height or self.y <= 0

    def is_eating_self(self):
        for pos in self.snake_list[:-1]:
            if pos[0] == self.x and pos[1] == self.y:
                return True
        return False
    
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

        self.generations = generations
        self.life_span = life_span

        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))

        self.x_change = 0
        self.y_change = 0
        self.snake_list = self.snake_list_init()

    def snake_list_init(self):
        snake_list = []
        for _ in range(self.population):
            snake_list.append(Snake(self.width, self.height))
        return snake_list

    def run(self, current_generation, best_snake_index, max_score, snake_index):
        dead_snake_count = 0
        start_ticks = pygame.time.get_ticks()
        timer = start_ticks
        clock =pygame.time.Clock()

        snake = self.snake_list[snake_index]
        for step in range(self.life_span):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()

            if snake.is_dead:
                break

            snake.is_dead = snake.is_colliding()
            if not snake.is_dead:
                snake.is_dead = snake.is_eating_self()

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

            # draw snake
            # reset everything to black before drawing
            self.display.fill((0, 0, 0))
            for pos in snake.snake_list:
                if not snake.is_dead:
                    pygame.draw.rect(self.display, (255, 0, 0), [pos[0], pos[1], 10, 10])

            # draw food
            pygame.draw.rect(self.display, (255, 0, 255), [snake.food_x, snake.food_y, 10, 10])

            value = pygame.font.SysFont("comicsansms", 14).render("Generation: " + str(current_generation), True, (255,0,0))
            self.display.blit(value, [0, 0])
            value = pygame.font.SysFont("comicsansms", 14).render("Individual: " + str(snake_index + 1) + "/" + str(population), True, (255,0,0))
            self.display.blit(value, [0, 30])
            value = pygame.font.SysFont("comicsansms", 14).render("Current Score: " + str(snake.get_fitness()), True, (255,0,0))
            self.display.blit(value, [0, 60])
            value = pygame.font.SysFont("comicsansms", 14).render("Max Score: " + str(best_snake_index) + ": " + str(max_score), True, (255,0,0))
            self.display.blit(value, [0, 90])

            pygame.display.update()

            # Eat food 
            if snake.x == snake.food_x and snake.y == snake.food_y:
                snake.food_x = round(random.randrange(0, snake.width - 10) / 10.0) * 10.0
                snake.food_y = round(random.randrange(0, snake.height - 10) / 10.0) * 10.0

                snake.fitness += 1000

                # increase length of snake
                snake.length_of_snake += 1

            clock.tick(self.snake_speed)
            snake.life_span = step

            
        return snake.get_fitness()

    def start(self):
        max_score = 0
        for generation_number in range(self.generations):
            total_score = 0
            is_better_generation = False

            previous_max_score = max_score
            max_score = 0
            best_snake_index = 0
            best_weights = []
            for snake_index in range(population):
                score = self.run(generation_number, best_snake_index, max_score, snake_index)
                if score > max_score:
                    best_snake_index = snake_index
                    max_score = score
                    best_weights = self.snake_list[snake_index].nn.get_weights()

                total_score += score

            if max_score >= previous_max_score:
                is_better_generation = True


            if total_score == 0:
                snake_weights = [(1.0/len(self.snake_list))] * len(self.snake_list)
            else:
                snake_weights = []
                for snake in self.snake_list:
                    snake_weights.append(snake.get_fitness()/total_score)

            #print(snake_weights)

            new_weights_list = []
            # crossover & mutate
            for i in range(self.population):
                # select 2 parents
                snake1_index = np.random.choice(range(len(self.snake_list)), p=snake_weights)
                snake2_index = np.random.choice(range(len(self.snake_list)), p=snake_weights)

                #print(snake1_index, snake2_index)

                # crossover those 2 parents
                new_weights = self.crosover(snake1_index, snake2_index)
                
                # if new population is not better than previous, set to previous best snake
                if not is_better_generation:
                    new_weights[0] = best_weights

                # mutate that new child
                mutated_new_weights1 = self.mutate(new_weights[0])

                new_weights_list.append(mutated_new_weights1)

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
                if random.uniform(0,1) > .85:
                    change = random.uniform(-0.5, 0.5)
                    weights[i][j] += change
        return weights

    def quit(self):
        pygame.quit()
        quit()
    

game_speed = 1000
population = 250
number_of_generations = 10000
life_span_per_generation = 300

snake_game = SnakeGame(500, 500, game_speed, population, number_of_generations, life_span_per_generation)
snake_game.start()
