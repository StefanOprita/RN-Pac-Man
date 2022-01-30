import random

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

from Models.PacManModel import PacManModel
from ReinforcementLearning.LearningStrategy import LearningStrategy


def simplify_image(image):
    image_reshaped = image[:172:]
    image_tiles = []
    slices = np.array_split(image_reshaped, 15)
    for slice in slices:
        pieces = np.array_split(slice, 15, axis=1)
        image_tiles.append(pieces)
    image_stripped_down = []

    ms_pacman_color = np.array([210, 164, 74])
    empty_blue_color = np.array([0, 28, 136])
    wall_money_color = np.array([228, 111, 111])

    def count_pixels_of_color(image, color):
        count = 0
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                pixel = image[i, j]
                if np.array_equal(pixel, color):
                    count += 1
        return count

    def flood_fill(image, i, j, original_color, replacement_color):
        print('b')
        if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
            return set()

        if np.array_equal(image[i, j], original_color):
            indexes = {i, j}
            image[i, j] = replacement_color
            indexes |= flood_fill(image, i - 1, j, original_color, replacement_color)
            indexes |= flood_fill(image, i, j - 1, original_color, replacement_color)
            indexes |= flood_fill(image, i + 1, j, original_color, replacement_color)
            indexes |= flood_fill(image, i, j + 1, original_color, replacement_color)
            return indexes
        return set()

    def is_pellete(image, i1, j1, i2, j2):
        def count_blue(image, i, j):
            directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
            for i_direction in range(0, directions.shape[0]):
                direction = directions[i_direction]
                current = np.array([i, j])

                current = current + direction

    def find_pellet(image):
        def surrounded_by_blue(image, i1, j1):
            count = 0
            for i in range(i1 - 1, i1 + 3):
                for j in range(j1 - 1, j1 + 6):
                    if 0 <= i < image.shape[0] and 0 <= j < image.shape[1]:
                        if np.array_equal(image[i, j], empty_blue_color):
                            count += 1
            return count

        replacement_color = [0, 0, 0]
        found = False
        checked = set()
        for i in range(0, image.shape[0] - 1):
            for j in range(0, image.shape[1] - 3):
                if np.array_equal(image[i, j], wall_money_color) and np.array_equal(image[i + 1, j + 3],
                                                                                    wall_money_color):
                    blue = surrounded_by_blue(image, i, j)
                    if blue > 10:
                        return True

        return False

    def contains_ghost(image):
        red_ghost_color = np.array([200, 72, 72])
        pink_ghost_color = np.array([198, 89, 179])
        cyan_ghost_color = np.array([84, 184, 153])
        orange_ghost_color = np.array([189, 120, 60])

        all_ghosts = [red_ghost_color, pink_ghost_color, cyan_ghost_color, orange_ghost_color]

        for ghost in all_ghosts:
            count = count_pixels_of_color(image, ghost)
            if count > 10:
                return True

        return False

    colors = [
        111,
        255,
        74,
        0,
    ]

    i = 0
    for pieces in image_tiles:
        line_stripped_down = []
        j = 0
        for piece in pieces:
            # r = np.mean(piece[:,:,0])
            # g = np.mean(piece[:,:,1])
            # b = np.mean(piece[:,:,2])
            #
            # mean_color = np.array([r, g, b])
            value = 0

            total_pixels = piece.shape[0] * piece.shape[1]

            pixels_wall = count_pixels_of_color(piece, wall_money_color)

            if find_pellet(piece):
                # print(i, j)
                value = 1
            # pill_count = count_pixels_of_color(piece, wall_money_color)
            #
            # if pill_count / total_pixels < 0.4:
            #     value = 255

            # empty_blue = count_pixels_of_color(piece, empty_blue_color)
            #
            # if empty_blue / total_pixels > 0.7:
            #     value = 255
            if contains_ghost(piece):
                value = 3

            count_pacman = count_pixels_of_color(piece, ms_pacman_color)

            if count_pacman > 10:
                value = 2

            # difference = ms_pacman_color - mean_color

            # length = np.linalg.norm(difference)
            line_stripped_down.append(colors[value])
            j += 1

        image_stripped_down.append(line_stripped_down)
        i += 1

    new_image = np.array(image_stripped_down)
    return new_image


class ConvDQNModifiesImage(LearningStrategy):

    def __init__(self):
        super().__init__()
        self.records = list()

        self.resize_shape = (15, 15)

        self.nb_frames_stacked = 4

        self.online_model = self.__generate_model()

        self.target_model = self.__generate_model()

        self.max_records = 50000
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1
        self.min_epsilon = 0.1

        self.learning_rate = 0.05
        self.decrease_rate_epsilon = 0.99999

        self.stacked_frames = list()

        self.update_target_network_rate = 3000

        self.train_step = 0

        self.frames = 0
        self.frame_random_threshold = 0

        self.steps_to_train = 15
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
        # self.records.append((old_state, action, reward, new_state))
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
        model.add(Conv2D(input_shape=input_shape, strides=4, kernel_size=10, filters=8, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=1, strides=2, activation='relu'))
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
        print('a')
        return simplify_image(current_state)

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
