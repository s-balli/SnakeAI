#!/usr/bin/env python3

import numpy as np
import random

class DebugNeuralNetwork:
    def __init__(self, input_size=24, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with smaller random values for more stable start
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.weights3 = np.random.randn(hidden_size, output_size) * 0.1

        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, hidden_size))
        self.bias3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, x):
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.weights3) + self.bias3
        output = self.softmax(z3)
        return output

class DebugSnake:
    def __init__(self):
        self.reset()
        self.brain = DebugNeuralNetwork()

    def reset(self):
        # Start in center, facing right
        self.body = [(10, 15)]
        self.direction = (1, 0)  # Start moving right
        self.food = self.place_food()
        self.score = 0
        self.life_left = 200
        self.dead = False
        self.moves_made = 0

    def place_food(self):
        while True:
            food = (random.randint(0, 19), random.randint(0, 29))
            if food not in self.body and abs(food[0] - self.body[0][0]) > 3:
                return food

    def get_vision(self):
        """Get vision data with detailed logging"""
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

        print(f"\n=== Snake Vision Analysis ===")
        print(f"Position: {self.body[0]}")
        print(f"Food: {self.food}")
        print(f"Current direction: {self.direction}")

        for i, (dx, dy) in enumerate(directions):
            food_dist, body_dist, wall_dist = self.look_in_direction(dx, dy)
            vision.extend([food_dist, body_dist, wall_dist])

            dir_names = ["Up", "Up-Right", "Right", "Down-Right", "Down", "Down-Left", "Left", "Up-Left"]
            print(f"{dir_names[i]:<10} - Food: {food_dist:.3f}, Body: {body_dist:.3f}, Wall: {wall_dist:.3f}")

        return vision

    def look_in_direction(self, dx, dy):
        head_x, head_y = self.body[0]
        distance = 1
        food_found = False
        body_found = False

        while True:
            x = head_x + dx * distance
            y = head_y + dy * distance

            # Check wall collision
            if x < 0 or x >= 20 or y < 0 or y >= 30:
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

    def make_decision(self):
        """Make AI decision with detailed logging"""
        vision = self.get_vision()
        vision_array = np.array(vision).reshape(1, -1)

        output = self.brain.forward(vision_array)[0]
        direction_index = np.argmax(output)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        new_direction = directions[direction_index]

        print(f"\n=== AI Decision ===")
        print(f"Neural network outputs: [{output[0]:.3f}, {output[1]:.3f}, {output[2]:.3f}, {output[3]:.3f}]")
        print(f"Chosen direction: {new_direction} (index {direction_index})")
        print(f"Direction names: ['Up', 'Down', 'Left', 'Right']")

        # Prevent reversing
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            print(f"Direction changed from {self.direction} to {new_direction}")
            self.direction = new_direction
        else:
            print(f"Direction kept as {self.direction} (would reverse)")

        return new_direction

    def move(self):
        """Move snake with detailed logging"""
        if self.dead:
            return

        print(f"\n=== Move #{self.moves_made + 1} ===")
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        print(f"Current position: {self.body[0]}")
        print(f"Direction: {self.direction}")
        print(f"New position: {new_head}")

        # Check wall collision
        if new_head[0] < 0 or new_head[0] >= 20 or new_head[1] < 0 or new_head[1] >= 30:
            print(f"💀 DIED: Hit wall at {new_head}")
            self.dead = True
            return

        # Check body collision
        if new_head in self.body:
            print(f"💀 DIED: Hit body at {new_head}")
            self.dead = True
            return

        # Move snake
        self.body.insert(0, new_head)

        # Check food collision
        if new_head == self.food:
            self.score += 1
            print(f"🍎 ATE FOOD! Score: {self.score}")
            self.food = self.place_food()
            self.life_left = min(self.life_left + 100, 500)
        else:
            self.body.pop()

        self.moves_made += 1
        self.life_left -= 1

        print(f"New body length: {len(self.body)}")
        print(f"Life left: {self.life_left}")

        if self.life_left <= 0:
            print(f"💀 DIED: Ran out of life")
            self.dead = True

def run_debug_session():
    print("🐍 Starting Snake AI Debug Session")
    print("=" * 50)

    snake = DebugSnake()

    # Run for 10 steps or until death
    for step in range(10):
        print(f"\n{'='*20} STEP {step + 1} {'='*20}")

        if snake.dead:
            print("Snake is dead, stopping debug session")
            break

        snake.make_decision()
        snake.move()

        # Simple food-seeking behavior for comparison
        print(f"\n--- Simple Logic Analysis ---")
        head_x, head_y = snake.body[0]
        food_x, food_y = snake.food

        # What would a simple food-seeking AI do?
        dx = food_x - head_x
        dy = food_y - head_y

        if abs(dx) > abs(dy):
            suggested_dir = (1, 0) if dx > 0 else (-1, 0)
            suggested_name = "Right" if dx > 0 else "Left"
        else:
            suggested_dir = (0, 1) if dy > 0 else (0, -1)
            suggested_name = "Down" if dy > 0 else "Up"

        print(f"Simple logic suggests: {suggested_name} {suggested_dir}")
        print(f"Food at: {snake.food}, Snake at: {snake.body[0]}")

    print(f"\n{'='*20} SUMMARY {'='*20}")
    print(f"Final score: {snake.score}")
    print(f"Moves made: {snake.moves_made}")
    print(f"Life left: {snake.life_left}")
    print(f"Dead: {snake.dead}")

if __name__ == "__main__":
    run_debug_session()