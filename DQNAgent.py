from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from keras.models import Sequential, load_model

class DQNAgent:
    def __init__(self, state_size, action_size, replay_interval):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.0000001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.replay_interval = replay_interval
        self.steps = 0  # Counter for tracking the number of steps

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if self.steps % self.replay_interval == 0 and len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = self.model.predict(state.reshape(1, -1))
                if done:
                    target[0][action] = reward
                else:
                    target[0][action] = (reward +
                                         self.gamma *
                                         np.amax(self.model.predict(next_state.reshape(1, -1))[0]))
                self.model.fit(state.reshape(1, -1), target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.steps += 1

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = load_model(filename)