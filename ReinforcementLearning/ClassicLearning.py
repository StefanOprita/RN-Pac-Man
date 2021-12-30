import json
import random
import pickle

import numpy as np

from ReinforcementLearning.LearningStrategy import LearningStrategy
import tensorflow as tf


class ClassicLearning(LearningStrategy):
    """
    The learning method presented in the 8th course
    """

    def __init__(self):
        super().__init__()
        self.max_records = 2500
        self.records = []
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1

        self.frames = 0
        self.frame_random_threshold = 1000

        self.min_epsilon = 0.1
        self.max_epsilon = 1
        self.decrease_rate_epsilon = 0.9999
        self.learning_rate = 0.0001
        self.update_target_steps = 10000

    def get_next_action(self, current_state):
        chance = random.random()
        if chance < self.epsilon:
            return random.randint(0, 8)

        reshaped = current_state.reshape(1, current_state.shape[0])
        predict = self.model.model.predict(reshaped)
        return np.argmax(predict)

    def add_record(self, old_state, action, reward, new_state, is_done):
        self.records.append((old_state, action, reward, new_state, is_done))
        if len(self.records) > self.max_records:
            self.records = self.records[1:]

    def after_action(self, episode):
        # self.__reduce_epsilon(episode)

        # if len(self.records) >= self.batch_size * 2:
        #     self.__train_model()
        self.frames += 1
        if self.frames >= self.frame_random_threshold:
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.decrease_rate_epsilon

        if len(self.records) >= self.batch_size*2 and self.frames > self.frame_random_threshold:
            self.__train_model()


    def __reduce_epsilon(self, episode):
        """
        Made this into a function in case we decide to make some more complicated things in here
        :return:
        """
        self.epsilon = (1 - self.min_epsilon) * np.exp(-self.decrease_rate_epsilon * episode) + self.min_epsilon

    def __train_model(self):
        batch_size = random.randint(self.batch_size, self.batch_size * 2)

        chosen_records = random.sample(self.records, batch_size)

        states, actions, rewards, new_states, done = zip(*chosen_records)
        states = list(states)
        actions = list(actions)
        rewards = list(rewards)
        new_states = list(new_states)
        done = list(done)

        inputs = np.array(states)

        outputs = self.__get_outputs(inputs, actions, rewards, new_states, done)

        # i'm not sure it here it should be epochs=1
        self.model.model.fit(inputs, outputs, batch_size=batch_size // 8, epochs=1, verbose=0)

    def __get_outputs(self, states, actions, rewards, new_states, done):

        new_states_reshaped = np.array(new_states)
        predictions = self.model.model.predict(new_states_reshaped)
        max_qs = np.amax(predictions, axis=1)

        predictions = self.model.model.predict(states)

        for index, action in enumerate(actions):
            if not done:
                predictions[index, action] = rewards[index] + (self.gamma * max_qs[index])
            else:
                predictions[index, action] = rewards[index]

        return predictions

    def serialize(self, episode):
        with open(f"info\\episode_records_{episode}", "wb") as file:
            pickle.dump(self.records, file, 0)

        with open(f"configs\\episode_config_{episode}.txt", "w") as file:
            file.write(f"{self.epsilon} {self.frames}")


    def update_target_network(self):
        pass
        # self.target_model.model.set_weights(self.model.model.get_weights())