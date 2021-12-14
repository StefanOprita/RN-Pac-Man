import gym
import numpy as np
from gym.envs.classic_control import rendering
import hyperparameters as hp
from Models.Model1 import Model1, PacManModel

from ReinforcementLearning.LearningStrategy import LearningStrategy
from ReinforcementLearning.ClassicLearning import ClassicLearning

import tensorflow as tf


def repeat_upsample(rgb_array, k=1, repeat_times=1):
    # repeat kinda crashes if k/repeat_times are zero
    if k <= 0 or repeat_times <= 0:
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), repeat_times, axis=1)


def train_model(env, model: PacManModel, strategy: LearningStrategy):
    strategy.set_model(model)

    viewer = rendering.SimpleImageViewer()
    for i_episode in range(hp.number_of_episodes):
        observation = env.reset()
        strategy.beginning_of_episode()
        done = False
        total_reward = 0
        while not done:
            rgb = env.render('rgb_array')
            upscale = repeat_upsample(rgb, 4, 4)
            viewer.imshow(upscale)

            strategy.before_action()

            action = strategy.get_next_action(observation)

            old_state = np.copy(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            strategy.add_record(old_state=old_state, action=action, reward=reward, new_state=observation)

            strategy.after_action()
            if done:
                print("Episode finished after {} timesteps".format(i_episode + 1))
                break
        strategy.end_of_episode()
        print(f'The total reward was {reward}')


def main():
    env = gym.make('MsPacman-v0')
    env.reset()
    test_model = Model1()
    strategy = ClassicLearning()

    train_model(env, test_model, strategy)

    env.close()


if __name__ == '__main__':
    main()
