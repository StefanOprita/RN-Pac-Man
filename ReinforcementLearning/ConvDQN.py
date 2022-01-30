import random

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

from Models.PacManModel import PacManModel
from ReinforcementLearning.LearningStrategy import LearningStrategy


class ConvDQN(LearningStrategy):

    def __init__(self):
        super().__init__()
        self.records = list()

        self.resize_shape = (84, 80)

        self.nb_frames_stacked = 4

        self.online_model = self.__generate_model()

        self.target_model = self.__generate_model()

        self.max_records = 10000
        self.gamma = 0.9
        self.batch_size = 64
        self.epsilon = 0.3
        self.min_epsilon = 0.1

        self.learning_rate = 0.05
        self.decrease_rate_epsilon = 0.99999

        self.stacked_frames = list()

        self.update_target_network_rate = 3000

        self.train_step = 0

        self.frames = 0
        self.frame_random_threshold = 0

        self.steps_to_train = 10
        self.steps_to_train_waiting = 0

    def get_next_action(self, current_state):
        chance = random.random()
        current_state = self.__modify_image(current_state)
        self.__add_to_stacked_list(current_state)

        if chance < self.epsilon or len(self.stacked_frames) != self.nb_frames_stacked:
            return random.randint(0, 3) + 1
        reshaped = self.__stack_frames(self.stacked_frames)
        reshaped = tf.expand_dims(reshaped, axis=0)
        predict = self.online_model(reshaped)

        return np.argmax(predict) + 1

    def add_record(self, old_state, action, reward, new_state, is_done):
        if len(self.stacked_frames) != self.nb_frames_stacked:
            return

        old_stacked = self.__stack_frames(self.stacked_frames)

        new_stacked = self.stacked_frames[:]

        new_state = self.__modify_image(new_state)
        new_stacked.append(new_state)
        new_stacked.pop(0)

        new_stacked = self.__stack_frames(new_stacked)
        # new_stacked = tf.expand_dims(new_stacked, axis=0)

        self.records.append((old_stacked, action, reward, new_stacked, is_done))

        if len(self.records) > self.max_records:
            self.records.pop(0)

    def after_action(self):
        self.__reduce_epsilon()

        if self.frames < self.frame_random_threshold:
            return

        if len(self.records) >= self.batch_size * 2:
            self.steps_to_train_waiting += 1
            if self.steps_to_train_waiting % self.steps_to_train == 0:
                self.__train_model()

    def __reduce_epsilon(self):
        """
        Made this into a function in case we decide to make some more complicated things in here
        :return:
        """
        self.frames += 1
        if self.frames < self.frame_random_threshold:
            return

        self.epsilon *= self.decrease_rate_epsilon
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

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
        self.online_model.fit(inputs, outputs, batch_size=self.batch_size // 8, epochs=1, verbose=0)

        self.train_step += 1
        if self.train_step >= self.update_target_network_rate:
            self.target_model.set_weights(self.online_model.get_weights())

    def __get_outputs(self, states, actions, rewards, new_states, done):
        new_states_reshaped = np.array(new_states)
        predictions = self.target_model.predict(new_states_reshaped)
        max_qs = np.max(predictions, axis=1)

        predictions = self.online_model.predict(states)

        for index, action in enumerate(actions):
            if not done[index]:
                predictions[index, action - 1] = rewards[index] + self.gamma * max_qs[index]
            else:
                predictions[index, action - 1] = rewards[index]

        return predictions

    def __generate_model(self):
        model = Sequential()
        input_shape = (self.resize_shape[0], self.resize_shape[1], self.nb_frames_stacked)
        model.add(Conv2D(input_shape=input_shape, strides=4, kernel_size=16, filters=8, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4))

        model.compile(optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), loss='mse')

        return model

    def set_model(self, model: PacManModel):
        pass

    def __add_to_stacked_list(self, current_state):
        self.stacked_frames.append(current_state)
        if len(self.stacked_frames) > self.nb_frames_stacked:
            self.stacked_frames.pop(0)

    @staticmethod
    def __modify_image(current_state):
        img = current_state[1:168:2, :: 2, :]  # crop and downsize
        img = img - 128  # normalize between -128 and 127
        img = img.astype('int8')  # saves memory
        grey = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        return grey

    @staticmethod
    def __stack_frames(stacked_frames):
        return np.dstack(tuple(stacked_frames))

    def serialize(self, episode):
        self.online_model.save(f'mspacman_models\\{episode}-online')
        # self.target_model.save(f'mspacman_models\\{episode}-target')

        # with open(f"info\\episode_records_{episode}-ModelMic-OmicronX", "wb") as file:
        #     pickle.dump(self.records, file, 0)
        #
        # with open(f"configs\\episode_config_{episode}-ModelMic-OmicronX.txt", "w") as file:
        #     file.write(f"{self.epsilon} {self.frames}")
