import random

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from Models.PacManModel import PacManModel
from ReinforcementLearning.LearningStrategy import LearningStrategy
import tensorflow as tf

import cv2


class ConvDQN(LearningStrategy):
    """
    The learning method presented in the 8th course
    """

    def __init__(self):
        super().__init__()
        self.records = list()

        self.resize_shape = (84, 80)

        self.nb_frames_stacked = 4



        self.online_model = self.__generate_model()

        self.target_model = self.__generate_model()

        self.max_records = 10000
        self.gamma = 0.9
        self.batch_size = 128
        self.epsilon = 0
        self.min_epsilon = 0.1

        self.learning_rate = 0.05
        self.decrease_rate_epsilon = 0.99999

        self.stacked_frames = list()

        self.update_target_network_rate = 10

        self.train_step = 0

    def get_next_action(self, current_state):
        chance = random.random()
        current_state = self.__modify_image(current_state)
        self.__add_to_stacked_list(current_state)
        if chance < self.epsilon or len(self.stacked_frames) != self.nb_frames_stacked:
            return random.randint(0, 3) + 1
        reshaped = np.array(self.stacked_frames)
        reshaped = np.reshape(reshaped, (84, 80, 4))
        reshaped = tf.expand_dims(reshaped, axis=0)
        predict = self.online_model.predict(reshaped)

        return np.argmax(predict) + 1

    def add_record(self, old_state, action, reward, new_state):
        self.records.append((old_state, action, reward, new_state))
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def after_action(self):
        self.__reduce_epsilon()

        if len(self.records) >= self.batch_size * 2:
            pass
            # self.__train_model()

    def __reduce_epsilon(self):
        """
        Made this into a function in case we decide to make some more complicated things in here
        :return:
        """
        self.epsilon *= self.decrease_rate_epsilon
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def __train_model(self):
        chosen_records = random.sample(self.records, self.batch_size)
        states, actions, rewards, new_states = zip(*chosen_records)
        states = list(states)
        actions = list(actions)
        rewards = list(rewards)
        new_states = list(new_states)

        inputs = np.array(states)

        outputs = self.__get_outputs(inputs, actions, rewards, new_states)

        # i'm not sure it here it should be epochs=1
        self.model.model.fit(inputs, outputs, batch_size=self.batch_size // 8, epochs=1)

        self.train_step += 1
        if self.train_step >= self.update_target_network_rate:
            self.target_model.set_weights(self.online_model.get_weights())

    def __get_outputs(self, states, actions, rewards, new_states):

        new_states_reshaped = np.array(new_states)
        predictions = self.model.model.predict(new_states_reshaped)
        max_qs = np.max(predictions, axis=1)

        predictions = self.model.model.predict(states)

        for index, action in enumerate(actions):
            predictions[index, action] = rewards[index] + max_qs[index]

        return predictions

    def __generate_model(self):
        model = Sequential()
        input_shape = (self.resize_shape[0], self.resize_shape[1], self.nb_frames_stacked)
        model.add(Conv2D(input_shape=input_shape, strides=4, kernel_size=16, filters=8, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4))

        model.compile(optimizer=Adam(), loss='mse')

        return model

    def set_model(self, model: PacManModel):
        pass

    def __add_to_stacked_list(self, current_state):
        self.stacked_frames.append(current_state)
        if len(self.stacked_frames) > self.nb_frames_stacked:
            self.stacked_frames.pop(0)

    def __modify_image(self, current_state):
        img = current_state[1:168:2, :: 2, :]  # crop and downsize
        img = img - 128  # normalize between -128 and 127
        img = img.astype('int8')  # saves memory
        grey = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        return grey
