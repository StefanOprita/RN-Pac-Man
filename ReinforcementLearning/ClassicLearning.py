import random
from collections import deque

import numpy as np

from ReinforcementLearning.LearningStrategy import LearningStrategy
import tensorflow as tf


class ClassicLearning(LearningStrategy):
    """
    The learning method presented in the 8th course
    """

    def __init__(self):
        super().__init__()
        self.max_records = 10000
        self.records = deque(maxlen=self.max_records)
        self.gamma = 0.9
        self.batch_size = 128
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.max_epsilon= 1
        self.decrease_rate_epsilon = 0.0001
        self.learning_rate = 0.001

    def get_next_action(self, current_state):
        chance = random.random()
        if chance < self.epsilon:
            return random.randint(0, 8)

        reshaped = current_state.reshape(1, current_state.shape[0])
        predict = self.model.model.predict(reshaped)
        return np.argmax(predict)

    def add_record(self, old_state, action, reward, new_state, is_done):
        self.records.append((old_state, action, reward, new_state, is_done))

    def after_action(self, episode):
        self.__reduce_epsilon(episode)

        if len(self.records) >= self.batch_size * 2:
            self.__train_model()

    def __reduce_epsilon(self, episode):
        """
        Made this into a function in case we decide to make some more complicated things in here
        :return:
        """
        self.epsilon = (1 - self.min_epsilon) * np.exp(-self.decrease_rate_epsilon * episode) + self.min_epsilon

    def __train_model(self):
        chosen_records = random.sample(self.records, self.batch_size)
        states, actions, rewards, new_states, done = zip(*chosen_records)
        states = list(states)
        actions = list(actions)
        rewards = list(rewards)
        new_states = list(new_states)
        done = list(done)

        inputs = np.array(states)

        outputs = self.__get_outputs(inputs, actions, rewards, new_states, done)

        # i'm not sure it here it should be epochs=1
        self.model.model.fit(inputs, outputs, batch_size=self.batch_size // 8, epochs=1, verbose=0)

    def __get_outputs(self, states, actions, rewards, new_states, done):

        new_states_reshaped = np.array(new_states)
        predictions = self.target_model.model.predict(new_states_reshaped)
        max_qs = np.max(predictions, axis=1)

        predictions = self.model.model.predict(states)

        for index, action in enumerate(actions):
            if not done:
                predictions[index, action] += rewards[index] + self.learning_rate * \
                                              (self.gamma * max_qs[index] - np.max(predictions[index]))
            else:
                predictions[index, action] += rewards[index]

        return predictions

    def update_target_network(self):
        self.target_model.model.set_weights(self.model.model.get_weights())