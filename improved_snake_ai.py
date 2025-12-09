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
YELLOW = (255, 255, 0)

class ImprovedNeuralNetwork:
    def __init__(self, input_size=24, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Better weight initialization
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.2
        self.weights2 = np.random.randn(hidden_size, hidden_size) * 0.2
        self.weights3 = np.random.randn(hidden_size, output_size) * 0.2

        # Small biases
        self.bias1 = np.random.randn(1, hidden_size) * 0.1
        self.bias2 = np.random.randn(1, hidden_size) * 0.1
        self.bias3 = np.random.randn(1, output_size) * 0.1

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, x):
        # First hidden layer
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = self.relu(z1)

        # Second hidden layer
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.relu(z2)

        # Output layer
        z3 = np.dot(a2, self.weights3) + self.bias3
        output = self.softmax(z3)

        return output

    def mutate(self, mutation_rate=0.1):
        # More aggressive mutation for learning
        if random.random() < mutation_rate:
            self.weights1 += np.random.randn(*self.weights1.shape) * 0.3
        if random.random() < mutation_rate:
            self.weights2 += np.random.randn(*self.weights2.shape) * 0.3
        if random.random() < mutation_rate:
            self.weights3 += np.random.randn(*self.weights3.shape) * 0.3

        # Clamp weights
        self.weights1 = np.clip(self.weights1, -2, 2)
        self.weights2 = np.clip(self.weights2, -2, 2)
        self.weights3 = np.clip(self.weights3, -2, 2)

    def set_heuristic_weights(self):
        """Initialize with some heuristic knowledge"""
        # Set initial weights to prefer food-seeking behavior
        # This gives the AI a starting point
        self.weights1 *= 0.5  # Reduce randomness

class SmartSnake:
    def __init__(self, brain=None, use_heuristics=True):
        self.reset()
        if brain is None:
            self.brain = ImprovedNeuralNetwork()
            if use_heuristics:
                self.brain.set_heuristic_weights()
        else:
            self.brain = brain

    def reset(self):
        # Start position in the middle
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)  # Start moving right
        self.food = self.place_food()
        self.score = 0
        self.life_left = 300  # More starting life
        self.dead = False
        self.fitness = 0
        self.moves_without_food = 0

    def place_food(self):
        while True:
            food = (random.randint(2, GRID_WIDTH-3), random.randint(2, GRID_HEIGHT-3))
            if food not in self.body:
                # Make sure food is not too close to snake
                dist = abs(food[0] - self.body[0][0]) + abs(food[1] - self.body[0][1])
                if dist > 5:
                    return food

    def look(self):
        """Improved vision system with better distance calculation"""
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
            food_dist, body_dist, wall_dist = self.look_in_direction(dx, dy)
            vision.extend([food_dist, body_dist, wall_dist])

        return vision

    def look_in_direction(self, dx, dy):
        """Better distance calculation - closer objects have higher values"""
        head_x, head_y = self.body[0]
        distance = 1
        food_found = False
        body_found = False
        food_distance = 0
        body_distance = 0

        while True:
            x = head_x + dx * distance
            y = head_y + dy * distance

            # Check wall collision
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                # Closer = higher value
                wall_dist = 1.0 / max(distance, 1)
                food_dist = 1.0 / max(food_distance, 20) if food_found else 0
                body_dist = 1.0 / max(body_distance, 20) if body_found else 0
                return [food_dist, body_dist, wall_dist]

            # Check food collision
            if not food_found and (x, y) == self.food:
                food_found = True
                food_distance = distance

            # Check body collision
            if not body_found and (x, y) in self.body[1:]:
                body_found = True
                body_distance = distance

            distance += 1
            if distance > 50:  # Prevent infinite loops
                break

        return [0, 0, 0]

    def think(self):
        """Improved decision making with fallback heuristics"""
        vision = self.look()
        vision_array = np.array(vision).reshape(1, -1)

        output = self.brain.forward(vision_array)[0]

        # Choose direction with highest output
        direction_index = np.argmax(output)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        new_direction = directions[direction_index]

        # Fallback to simple heuristics if AI output is too random
        if max(output) < 0.35:  # If AI is uncertain
            new_direction = self.get_heuristic_direction()

        # Prevent reversing and immediate suicide
        if not self.is_safe_direction(new_direction):
            # Try to find any safe direction
            for dir in directions:
                if (dir[0] * -1, dir[1] * -1) != self.direction and self.is_safe_direction(dir):
                    new_direction = dir
                    break

        # Prevent reversing
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def get_heuristic_direction(self):
        """Simple heuristic: move towards food if safe"""
        head_x, head_y = self.body[0]
        food_x, food_y = self.food

        dx = food_x - head_x
        dy = food_y - head_y

        # Prefer the direction that gets us closer to food
        if abs(dx) > abs(dy):
            preferred = (1, 0) if dx > 0 else (-1, 0)
        else:
            preferred = (0, 1) if dy > 0 else (0, -1)

        # Check if preferred direction is safe
        if self.is_safe_direction(preferred) and (preferred[0] * -1, preferred[1] * -1) != self.direction:
            return preferred

        # If not safe, try other directions
        alternatives = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for alt in alternatives:
            if self.is_safe_direction(alt) and (alt[0] * -1, alt[1] * -1) != self.direction:
                return alt

        return self.direction  # Keep current direction as last resort

    def is_safe_direction(self, direction):
        """Check if moving in this direction is immediately safe"""
        head_x, head_y = self.body[0]
        new_x = head_x + direction[0]
        new_y = head_y + direction[1]

        # Check wall collision
        if new_x < 0 or new_x >= GRID_WIDTH or new_y < 0 or new_y >= GRID_HEIGHT:
            return False

        # Check body collision
        if (new_x, new_y) in self.body:
            return False

        return True

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
            self.life_left = min(self.life_left + 150, 800)  # More life reward
            self.moves_without_food = 0
        else:
            self.body.pop()
            self.moves_without_food += 1

        self.life_left -= 1
        if self.life_left <= 0 or self.moves_without_food > 100:
            self.dead = True

    def calculate_fitness(self):
        """Better fitness function"""
        # Reward score heavily
        self.fitness = self.score * 1000

        # Reward surviving longer
        self.fitness += (300 - self.life_left) * 10

        # Bonus for not starving
        self.fitness -= self.moves_without_food * 5

        # Make sure fitness is positive
        self.fitness = max(1, self.fitness)

    def clone(self):
        """Create a copy of the snake"""
        new_snake = SmartSnake(use_heuristics=False)
        new_snake.brain = ImprovedNeuralNetwork()
        new_snake.brain.weights1 = self.brain.weights1.copy()
        new_snake.brain.weights2 = self.brain.weights2.copy()
        new_snake.brain.weights3 = self.brain.weights3.copy()
        new_snake.brain.bias1 = self.brain.bias1.copy()
        new_snake.brain.bias2 = self.brain.bias2.copy()
        new_snake.brain.bias3 = self.brain.bias3.copy()
        return new_snake

class SimpleEvolution:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_snake = None
        self.best_score = 0

        # Create initial population
        for _ in range(population_size):
            self.population.append(SmartSnake())

    def evolve_one_generation(self):
        """Evolve one generation without graphics"""
        print(f"\n=== Generation {self.generation} ===")

        # Run all snakes
        for i, snake in enumerate(self.population):
            if i % 10 == 0:
                print(f"Running snake {i+1}/{self.population_size}")

            while not snake.dead and snake.life_left > 0:
                snake.think()
                snake.move()

            snake.calculate_fitness()

        # Find best snake
        best_fitness = 0
        best_index = 0
        total_fitness = 0

        for i, snake in enumerate(self.population):
            total_fitness += snake.fitness
            if snake.fitness > best_fitness:
                best_fitness = snake.fitness
                best_index = i

        avg_fitness = total_fitness / len(self.population)
        best_snake = self.population[best_index]

        print(f"Best score: {best_snake.score}")
        print(f"Best fitness: {best_fitness:.1f}")
        print(f"Average fitness: {avg_fitness:.1f}")

        # Create new population
        new_population = []

        # Keep the best snake
        new_population.append(best_snake.clone())

        # Create rest of population through selection and mutation
        for _ in range(self.population_size - 1):
            # Tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Simple crossover
            child = self.crossover(parent1, parent2)
            child.brain.mutate(0.2)  # Higher mutation rate for exploration
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return best_snake

    def tournament_selection(self, tournament_size=5):
        """Select a snake using tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1, parent2):
        """Simple crossover between two parent snakes"""
        child = SmartSnake(use_heuristics=False)

        # Mix weights from parents
        mask1 = np.random.random(parent1.brain.weights1.shape) < 0.5
        mask2 = np.random.random(parent1.brain.weights2.shape) < 0.5
        mask3 = np.random.random(parent1.brain.weights3.shape) < 0.5

        child.brain.weights1 = np.where(mask1, parent1.brain.weights1, parent2.brain.weights1)
        child.brain.weights2 = np.where(mask2, parent1.brain.weights2, parent2.brain.weights2)
        child.brain.weights3 = np.where(mask3, parent1.brain.weights3, parent2.brain.weights3)

        return child

class ImprovedSnakeGame:
    def __init__(self, human_controlled=False, use_evolution=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Improved Snake AI")
        self.clock = pygame.time.Clock()
        self.human_controlled = human_controlled
        self.use_evolution = use_evolution
        self.running = True

        # Set FPS based on mode
        if human_controlled:
            self.fps = 10  # Slower for human control
        elif use_evolution:
            self.fps = 15  # Medium for evolution
        else:
            self.fps = 20  # Faster for AI watching

        if use_evolution:
            self.evolution = SimpleEvolution(population_size=20)
            self.generation = 0
            self.best_snake = None
            self.training_mode = True
            self.frame_counter = 0
            self.training_generations = 10
        else:
            self.snake = SmartSnake()
            self.training_mode = False
            if human_controlled:
                self.snake = SmartSnake()  # For human play, just track position

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
                    elif event.key == pygame.K_r:
                        self.snake.reset()
                else:
                    if event.key == pygame.K_r:
                        if self.use_evolution:
                            self.generation = 0
                            self.evolution = SimpleEvolution(population_size=20)
                            self.training_mode = True
                            self.best_snake = None
                        else:
                            self.snake.reset()
                    elif event.key == pygame.K_t and self.use_evolution:
                        self.training_mode = not self.training_mode
                    elif event.key == pygame.K_SPACE and self.use_evolution and not self.training_mode:
                        # Continue training if space is pressed after training completes
                        self.training_mode = True
                        self.training_generations = 5  # Train 5 more generations

    def draw(self):
        self.screen.fill(BLACK)

        if self.use_evolution:
            if self.training_mode:
                # Show training info with progress
                font = pygame.font.Font(None, 36)
                text = font.render(f"Training Generation {self.generation + 1}/{self.training_generations}", True, WHITE)
                self.screen.blit(text, (WIDTH//2 - 200, HEIGHT//2 - 30))

                font_small = pygame.font.Font(None, 24)
                progress_text = font_small.render("AI is learning... Please wait", True, WHITE)
                self.screen.blit(progress_text, (WIDTH//2 - 140, HEIGHT//2))

                controls_text = font_small.render("Press T to watch current best AI", True, WHITE)
                self.screen.blit(controls_text, (WIDTH//2 - 150, HEIGHT//2 + 30))

                # Show training progress bar
                bar_width = 300
                bar_height = 20
                bar_x = WIDTH//2 - bar_width//2
                bar_y = HEIGHT//2 + 70
                progress = (self.generation % self.training_generations) / self.training_generations

                pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
                pygame.draw.rect(self.screen, GREEN, (bar_x, bar_y, int(bar_width * progress), bar_height))

            else:
                # Show the best snake playing
                if self.best_snake:
                    self.draw_snake(self.best_snake)
                    self.draw_info(self.best_snake)
                else:
                    # Show waiting message
                    font = pygame.font.Font(None, 36)
                    text = font.render("Training complete! Ready to watch AI", True, WHITE)
                    self.screen.blit(text, (WIDTH//2 - 250, HEIGHT//2))
        else:
            # Single snake mode
            self.draw_snake(self.snake)
            self.draw_info(self.snake)

        pygame.display.flip()

    def draw_snake(self, snake):
        # Draw snake
        for i, segment in enumerate(snake.body):
            color = GREEN if i == 0 else WHITE  # Head is green
            pygame.draw.rect(self.screen, color,
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE, GRID_SIZE))

        # Draw food
        pygame.draw.rect(self.screen, RED,
                        (snake.food[0] * GRID_SIZE, snake.food[1] * GRID_SIZE,
                         GRID_SIZE, GRID_SIZE))

    def draw_info(self, snake, mode_text=None):
        font = pygame.font.Font(None, 36)

        # Determine correct mode text
        if mode_text is None:
            if self.human_controlled:
                display_text = "Human Control"
            elif self.use_evolution:
                if self.training_mode:
                    display_text = f"Training Generation {self.generation}"
                else:
                    display_text = f"AI Control - Gen {self.generation}"
            else:
                display_text = "AI Control"
        else:
            display_text = mode_text

        # Mode
        mode_text_render = font.render(display_text, True, YELLOW)
        self.screen.blit(mode_text_render, (10, 10))

        # Score
        score_text = font.render(f"Score: {snake.score}", True, WHITE)
        self.screen.blit(score_text, (10, 50))

        # Life left
        life_text = font.render(f"Life: {snake.life_left}", True, WHITE)
        self.screen.blit(life_text, (10, 90))

        # Controls - show different controls based on mode
        font_small = pygame.font.Font(None, 24)
        if self.human_controlled:
            controls_text = font_small.render("Arrow Keys: Move | R: Reset", True, WHITE)
        elif self.use_evolution:
            if self.training_mode:
                controls_text = font_small.render("Training... | T: Watch Mode | R: Restart Training", True, WHITE)
            else:
                controls_text = font_small.render("T: Training Mode | Space: More Training | R: Restart", True, WHITE)
        else:
            controls_text = font_small.render("R: Reset", True, WHITE)

        self.screen.blit(controls_text, (10, HEIGHT - 30))

        if snake.dead:
            dead_text = font.render("GAME OVER", True, RED)
            text_rect = dead_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            self.screen.blit(dead_text, text_rect)

            restart_text = font_small.render("Press R to restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 30))
            self.screen.blit(restart_text, restart_rect)

    def run(self):
        while self.running:
            self.handle_events()

            if self.use_evolution and self.training_mode:
                # Training mode - evolve with frame delay to show progress
                self.frame_counter += 1
                if self.frame_counter >= 60:  # Evolve every 60 frames (slower)
                    self.best_snake = self.evolution.evolve_one_generation()
                    self.generation += 1
                    self.frame_counter = 0

                    # Stop after initial training generations
                    if self.generation >= self.training_generations:
                        self.training_mode = False
                        print(f"Training completed! Best score: {self.best_snake.score}")
                        # Reset the best snake to start fresh
                        self.best_snake.reset()

            else:
                # Playing mode
                if self.human_controlled:
                    # Human control - just move snake based on input
                    if not self.snake.dead:
                        self.snake.move()
                elif self.use_evolution and self.best_snake:
                    # Show the best snake playing
                    if not self.best_snake.dead:
                        self.best_snake.think()
                        self.best_snake.move()
                    else:
                        # If snake dies, reset it to watch again
                        self.best_snake.reset()
                elif not self.use_evolution:
                    # Single AI snake
                    if not self.snake.dead:
                        self.snake.think()
                        self.snake.move()

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()

def main():
    print("Improved Snake AI")
    print("1. Human Control")
    print("2. Single AI Control")
    print("3. Evolution Training")

    choice = input("Choose mode (1, 2, or 3): ").strip()

    if choice == "1":
        game = ImprovedSnakeGame(human_controlled=True)
        print("Human control mode - Use arrow keys to control")
    elif choice == "2":
        game = ImprovedSnakeGame(human_controlled=False)
        print("Single AI control mode")
    elif choice == "3":
        game = ImprovedSnakeGame(human_controlled=False, use_evolution=True)
        print("Evolution training mode - Press T to toggle training/watching")
    else:
        print("Invalid choice, starting with single AI...")
        game = ImprovedSnakeGame(human_controlled=False)

    game.run()

if __name__ == "__main__":
    main()