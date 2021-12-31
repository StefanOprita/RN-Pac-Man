import gym
import numpy as np
from gym.envs.classic_control import rendering
import hyperparameters as hp
from Models.Model1 import PacManModel
from Models.ModeluLuNenea import ModelulLuiNenea

from ReinforcementLearning.LearningStrategy import LearningStrategy
from ReinforcementLearning.ClassicLearning import ClassicLearning

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def repeat_upsample(rgb_array, k=1, repeat_times=1):
    # repeat kinda crashes if k/repeat_times are zero
    if k <= 0 or repeat_times <= 0:
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), repeat_times, axis=1)


def train_model(env, model: PacManModel, strategy: LearningStrategy, render_window: bool = False):
    strategy.set_model(model)

    viewer = rendering.SimpleImageViewer()
    for i_episode in range(hp.number_of_episodes):
        observation = env.reset()
        strategy.beginning_of_episode()

        done = False
        total_reward = 0
        score = 0
        timesteps = 0
        lives = 3

        ep_frame_number = 0
        while ep_frame_number < 258:
            observation, _, _, _ = env.step(0)

            ep_frame_number = float(env.ale.getEpisodeFrameNumber())

        while not done:
            timesteps += 1
            if render_window:
                rgb = env.render('rgb_array')
                upscale = repeat_upsample(rgb, 4, 4)
                viewer.imshow(upscale)

            strategy.before_action()

            action = strategy.get_next_action(observation)
            old_state = np.copy(observation)

            observation, reward, done, info = env.step(action)
            score += reward
            if info['lives'] < lives:
                reward = -300
                lives -= 1

            total_reward += reward

            strategy.add_record(old_state=old_state, action=action, reward=reward, new_state=observation, is_done=done)

            strategy.after_action(i_episode)
            if done:
                print("Episode {}finished after {} timesteps".format(i_episode, timesteps + 1))
                break
        strategy.end_of_episode()

        if i_episode % 50 == 0:
            strategy.model.model.save_weights(f"mspacman\\{i_episode}-beta-pacman.h5")
            strategy.model.model.save(f'mspacman_models\\{i_episode}-beta.model')
            strategy.serialize(i_episode)


        print(f'The total reward was {total_reward} . The score was {score}')


def main():
    env = gym.make('MsPacman-ram-v4')
    env.reset()
    test_model = ModelulLuiNenea()
    test_model.model = tf.keras.models.load_model("mspacman_models\\450.model")
    strategy = ClassicLearning()
    strategy.load_info("450")
    train_model(env, test_model, strategy, True)

    env.close()


if __name__ == '__main__':
    main()
