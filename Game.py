import pygame
import random
import numpy as np
from DQNAgent import DQNAgent

# Fenstergröße
WIDTH = 800
HEIGHT = 600

# Farben
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Initialisierung von Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class Game:
    def __init__(self):
        # Spieler
        self.player_size = 50
        self.player_x = WIDTH // 2 - self.player_size // 2
        self.player_y = HEIGHT - self.player_size
        self.player_y_speed = 0
        self.player_jump_power = -15
        self.player_gravity = 0.6
        self.player_can_jump = True
        self.score_history = []

        # Hindernisse
        self.obstacle_width = self.player_size
        self.obstacle_height = self.player_size
        self.obstacle_x = WIDTH + 5 * self.obstacle_width
        self.obstacle_y = HEIGHT - self.obstacle_height
        self.obstacle_speed = 5
        self.obstacle_speed_increase_interval = 10  # Intervall, in dem die Geschwindigkeit erhöht wird
        self.obstacle_speed_increase_amount = 0.02  # Prozentsatz der Geschwindigkeitserhöhung
        self.obstacle_min_gap = 4  # Abstand nach einem Hindernis
        self.obstacle_max_gap = 10  # Maximale Abstand zwischen Hindernissen
        self.next_obstacle_gap = random.randint(self.obstacle_min_gap, self.obstacle_max_gap)

        # Kamera
        self.camera_offset_x = 0
        self.camera_speed = 3

        # Spielstatus
        self.score = 0
        self.block_counter = 0
        self.is_game_over = False
        self.jumped_over_obstacle = False  # Variable to check if player has jumped over an obstacle

        # AI
        self.state_size = 4
        self.action_size = 2
        self.agent = DQNAgent(self.state_size, self.action_size, replay_interval=100)
        self.batch_size = 32
        self.state = self.get_state()
        self.action = 0

        # Spiel zurücksetzen
        self.reset_game()

    def reset_game(self):
        self.player_y = HEIGHT - self.player_size
        self.player_y_speed = 0
        self.obstacle_x = WIDTH + 5 * self.obstacle_width
        self.score = 0
        self.block_counter = 0
        self.obstacle_speed = 5
        self.is_game_over = False
        self.jumped_over_obstacle = False
        self.score_history.append(self.score)

    def get_state(self):
        return np.array([
            self.player_y,
            self.obstacle_x,
            self.obstacle_y,
            self.obstacle_speed
        ])

    def spawn_obstacle(self):
        self.obstacle_x = WIDTH + self.next_obstacle_gap * self.obstacle_width
        self.obstacle_y = HEIGHT - self.obstacle_height
        self.next_obstacle_gap = random.randint(self.obstacle_min_gap, self.obstacle_max_gap)
        self.jumped_over_obstacle = False  # Reset the variable when a new obstacle is spawned

    def update_obstacles(self):
        if not self.is_game_over:
            self.obstacle_x -= self.obstacle_speed

        if self.obstacle_x < -self.obstacle_width:
            self.spawn_obstacle()

        if self.block_counter % self.obstacle_speed_increase_interval == 0:
            self.obstacle_speed += self.obstacle_speed * self.obstacle_speed_increase_amount

    def update_player(self):
        if not self.is_game_over:
            self.player_y_speed += self.player_gravity
            self.player_y += self.player_y_speed

        if self.player_y >= HEIGHT - self.player_size:
            self.player_y = HEIGHT - self.player_size
            self.player_can_jump = True

        if self.player_y < HEIGHT - self.player_size:
            self.player_can_jump = False

    def jump(self):
        if self.player_can_jump:
            self.player_y_speed = self.player_jump_power
            self.player_can_jump = False

    def check_collision(self):
        if (self.player_x + self.player_size > self.obstacle_x and
                self.player_x < self.obstacle_x + self.obstacle_width and
                self.player_y + self.player_size > self.obstacle_y and
                self.player_y < self.obstacle_y + self.obstacle_height):
            self.is_game_over = True


    def update(self):
        self.update_player()
        self.update_obstacles()
        self.check_collision()

        if not self.is_game_over:
            self.block_counter += 1
            if self.player_y + self.player_size < self.obstacle_y:
                self.score = self.block_counter // 100

    def draw(self):
        screen.fill(WHITE)
        pygame.draw.rect(screen, RED, pygame.Rect(self.player_x, self.player_y, self.player_size, self.player_size))
        pygame.draw.rect(screen, BLACK, pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height))

        font = pygame.font.Font(None, 36)
        text = font.render('Score: ' + str(self.score), True, (0, 0, 0))
        screen.blit(text, (50 - text.get_width() // 2, 30 - text.get_height() // 2))

        pygame.display.flip()

    def ai_play(self):
        done = self.is_game_over
        if not done:
            next_state = self.get_state()
            reward = 0.1

            # If the player has jumped over the obstacle, give a reward
            if self.obstacle_x + self.obstacle_width < self.player_x and not self.jumped_over_obstacle:
                reward = 1.0
                self.jumped_over_obstacle = True

            self.agent.remember(self.state, self.action, reward, next_state, done)
            self.state = next_state
            self.action = self.agent.act(self.state)

            if len(self.agent.memory) > self.batch_size:
                self.agent.replay(self.batch_size)

            if self.action == 1:  # Jump
                self.jump()
        else:
            next_state = self.get_state()
            reward = -1
            self.agent.remember(self.state, self.action, reward, next_state, done)
            self.reset_game()
            if self.block_counter % 5 == 0:  # Print current epsilon value every 5 iterations
                print("Current epsilon:", self.agent.epsilon)

    def play(self):
        running = True
        while running:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.jump()

            self.update()
            self.draw()
            self.ai_play()

        pygame.quit()

game = Game()
game.play()