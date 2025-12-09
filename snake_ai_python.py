import pygame
import numpy as np
import random
import math
from collections import deque

# Window dimensions
WIDTH = 800
HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class NeuralNetwork:
    def __init__(self, input_size=24, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.5
        self.weights2 = np.random.randn(hidden_size, hidden_size) * 0.5
        self.weights3 = np.random.randn(hidden_size, output_size) * 0.5

        # Bias terms
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, hidden_size))
        self.bias3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        # First hidden layer
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)

        # Second hidden layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)

        # Output layer
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.output = self.softmax(self.z3)

        return self.output

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def mutate(self, mutation_rate=0.05):
        # Mutate weights
        if random.random() < mutation_rate:
            self.weights1 += np.random.randn(*self.weights1.shape) * 0.1
        if random.random() < mutation_rate:
            self.weights2 += np.random.randn(*self.weights2.shape) * 0.1
        if random.random() < mutation_rate:
            self.weights3 += np.random.randn(*self.weights3.shape) * 0.1

        # Clamp weights
        self.weights1 = np.clip(self.weights1, -1, 1)
        self.weights2 = np.clip(self.weights2, -1, 1)
        self.weights3 = np.clip(self.weights3, -1, 1)

    def crossover(self, partner):
        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)

        # Simple crossover - mix weights from both parents
        mask1 = np.random.random(self.weights1.shape) < 0.5
        mask2 = np.random.random(self.weights2.shape) < 0.5
        mask3 = np.random.random(self.weights3.shape) < 0.5

        child.weights1 = np.where(mask1, self.weights1, partner.weights1)
        child.weights2 = np.where(mask2, self.weights2, partner.weights2)
        child.weights3 = np.where(mask3, self.weights3, partner.weights3)

        return child

class Snake:
    def __init__(self, brain=None):
        self.reset()
        if brain is None:
            self.brain = NeuralNetwork()
        else:
            self.brain = brain

    def reset(self):
        # Start position in the middle
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)  # Start moving right
        self.food = self.place_food()
        self.score = 0
        self.life_left = 200
        self.dead = False
        self.fitness = 0

    def place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if food not in self.body:
                return food

    def look(self):
        """Look in 8 directions and detect food, body, and wall"""
        vision = []
        directions = [
            (0, -1),   # Up
            (1, -1),   # Up-Right
            (1, 0),    # Right
            (1, 1),    # Down-Right
            (0, 1),    # Down
            (-1, 1),   # Down-Left
            (-1, 0),   # Left
            (-1, -1)   # Up-Left
        ]

        for dx, dy in directions:
            # Look for food, body, wall in this direction
            food_dist, body_dist, wall_dist = self.look_in_direction(dx, dy)
            vision.extend([food_dist, body_dist, wall_dist])

        return vision

    def look_in_direction(self, dx, dy):
        """Look in a specific direction and return distances to food, body, and wall"""
        head_x, head_y = self.body[0]
        distance = 1
        food_found = False
        body_found = False

        while True:
            x = head_x + dx * distance
            y = head_y + dy * distance

            # Check wall collision
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                food_dist = 1/distance if food_found else 0
                body_dist = 1/distance if body_found else 0
                wall_dist = 1/distance
                return [food_dist, body_dist, wall_dist]

            # Check food collision
            if not food_found and (x, y) == self.food:
                food_found = True

            # Check body collision
            if not body_found and (x, y) in self.body[1:]:
                body_found = True

            distance += 1

    def think(self):
        """Use neural network to decide next direction"""
        vision = self.look()
        vision_array = np.array(vision).reshape(1, -1)

        output = self.brain.forward(vision_array)[0]

        # Choose direction with highest output
        direction_index = np.argmax(output)

        # Map index to direction
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        new_direction = directions[direction_index]

        # Prevent reversing
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def move(self):
        """Move the snake"""
        if self.dead:
            return

        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.dead = True
            return

        # Check body collision
        if new_head in self.body:
            self.dead = True
            return

        # Move snake
        self.body.insert(0, new_head)

        # Check food collision
        if new_head == self.food:
            self.score += 1
            self.food = self.place_food()
            self.life_left = min(self.life_left + 100, 500)
        else:
            self.body.pop()

        self.life_left -= 1
        if self.life_left <= 0:
            self.dead = True

    def calculate_fitness(self):
        """Calculate fitness based on score and lifetime"""
        if self.score < 10:
            self.fitness = (200 - self.life_left) ** 2 * (2 ** self.score)
        else:
            self.fitness = (200 - self.life_left) ** 2 * (2 ** 10) * (self.score - 9)

    def clone(self):
        """Create a copy of the snake"""
        new_snake = Snake(self.brain)
        return new_snake

class SnakeGame:
    def __init__(self, human_controlled=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.human_controlled = human_controlled
        self.snake = Snake()
        self.running = True
        self.fps = 10

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.human_controlled:
                    if event.key == pygame.K_UP and self.snake.direction != (0, 1):
                        self.snake.direction = (0, -1)
                    elif event.key == pygame.K_DOWN and self.snake.direction != (0, -1):
                        self.snake.direction = (0, 1)
                    elif event.key == pygame.K_LEFT and self.snake.direction != (1, 0):
                        self.snake.direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and self.snake.direction != (-1, 0):
                        self.snake.direction = (1, 0)
                else:
                    if event.key == pygame.K_r:
                        self.snake.reset()

    def draw(self):
        self.screen.fill(BLACK)

        # Draw snake
        for i, segment in enumerate(self.snake.body):
            color = GREEN if i == 0 else WHITE  # Head is green
            pygame.draw.rect(self.screen, color,
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE, GRID_SIZE))

        # Draw food
        pygame.draw.rect(self.screen, RED,
                        (self.snake.food[0] * GRID_SIZE, self.snake.food[1] * GRID_SIZE,
                         GRID_SIZE, GRID_SIZE))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.snake.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # Draw life left
        life_text = font.render(f"Life: {self.snake.life_left}", True, WHITE)
        self.screen.blit(life_text, (10, 50))

        if self.snake.dead:
            dead_text = font.render("GAME OVER - Press R to restart", True, RED)
            text_rect = dead_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            self.screen.blit(dead_text, text_rect)

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()

            if not self.human_controlled and not self.snake.dead:
                self.snake.think()

            self.snake.move()
            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()

def main():
    print("Snake AI Test")
    print("1. Human Control")
    print("2. AI Control")

    choice = input("Choose mode (1 or 2): ").strip()

    if choice == "1":
        game = SnakeGame(human_controlled=True)
        print("Human control mode - Use arrow keys to control")
    else:
        game = SnakeGame(human_controlled=False)
        print("AI control mode - Watch the AI play")

    game.run()

if __name__ == "__main__":
    main()