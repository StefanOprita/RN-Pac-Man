from collections import deque
from statistics import mean, stdev

import gym
import numpy as np
import tensorflow
from gym.envs.classic_control import rendering

import hyperparameters as hp
from Models.MainModel import MainModel, PacManModel
from ReinforcementLearning.ConvDQN import ConvDQN
from ReinforcementLearning.LearningStrategy import LearningStrategy

import tensorflow as tf
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config)

def repeat_upsample(rgb_array, k=1, repeat_times=1):
    # repeat kinda crashes if k/repeat_times are zero
    if k <= 0 or repeat_times <= 0:
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), repeat_times, axis=1)


def skip_waiting_sequence(env):
    for i in range(0, 15):
        _, _, _, _ = env.step(0)


def train_model(env, model: PacManModel, strategy: LearningStrategy):
    strategy.set_model(model)

    strategy.online_model = tensorflow.keras.models.load_model('mspacman_models\\3350-online')
    strategy.target_model = tensorflow.keras.models.load_model('mspacman_models\\3350-online')

    scores = deque(maxlen=30)

    modify_action_timer = 3

    for i_episode in range(7801, hp.number_of_episodes):
        observation = env.reset()

        skip_waiting_sequence(env)

        strategy.beginning_of_episode()
        done = False
        total_reward = 0
        lives = 3
        score = 0
        life_steps = 0

        steps_without_reward = 0
        action = 0
        total_reward_this_life = 0
        while not done:
            env.render()

            strategy.before_action()

            if life_steps % modify_action_timer == 0:
                action = strategy.get_next_action(observation)

            old_state = np.copy(observation)

            observation, reward, done, info = env.step(action)
            life_steps += 1

            score += reward
            if info['lives'] < lives:
                reward = -30
                lives -= 1
                life_steps = 0
                skip_waiting_sequence(env)
                total_reward_this_life = 0

            if reward == 0:
                steps_without_reward += 1
                if steps_without_reward % 6:
                    reward = -0.005
            else:
                steps_without_reward = 0
                if reward == 10:
                    reward = 10
                else:
                    reward = 0

            if done and lives != 0:
                reward = 100

            total_reward += reward
            total_reward_this_life += reward

            strategy.add_record(old_state=old_state, action=action, reward=reward, new_state=observation, is_done=done)

            strategy.after_action()
            if done:
                print("Episode finished after {} timesteps".format(i_episode + 1))
                break
        strategy.end_of_episode()

        if i_episode % 50 == 0:
            strategy.serialize(i_episode)

        scores.append(score)
        print(
            f'The total reward was {total_reward} . The score was {score} .Eps: {strategy.epsilon}')

        if len(scores) > 2:
            print(f'Mean: {mean(scores)}.Stdev: {stdev(scores)}')
            print()


def test_model(env, strategy, model_name):
    strategy.online_model = tensorflow.keras.models.load_model(f'mspacman_models\\{model_name}')
    test_episodes = 1000

    viewer = rendering.SimpleImageViewer()
    scores = []

    modify_action_timer = 3

    for i_episode in range(test_episodes):
        observation = env.reset()

        skip_waiting_sequence(env)

        done = False
        total_reward = 0
        score = 0

        action = 0
        life_steps = 0
        while not done:
            # rgb = env.render('rgb_array')
            # upscale = repeat_upsample(rgb, 4, 4)
            # viewer.imshow(upscale)

            if life_steps % modify_action_timer == 0:
                action = strategy.get_next_action(observation)

            observation, reward, done, info = env.step(action)

            score += reward
            life_steps += 1

            total_reward += reward

            if done:
                print("Episode {} finished".format(i_episode))
                break

        scores.append(score)
        print(
            f'The total reward was {total_reward} . The score was {score} .Eps: {strategy.epsilon}')
        print(f'Min :  {min(scores)} . Max : {max(scores)} . Mean : {mean(scores)}')

    print(f"Minimum : {min(scores)}")
    print(f"Maximum : {max(scores)}")
    print(f"Mean : {mean(scores)}")
    print(f"Stdev : {stdev(scores)}")

    with open(f"scores/scores_{model_name}", "w") as file:
        for score in scores:
            file.write(f"{score}\n")


def main():
    # train_model(env, training_model, strategy)
    models_to_test = ['8900-online', '3350-online']

    for model in models_to_test:
        env = gym.make('MsPacman-v0')
        env.reset()

        strategy = ConvDQN()
        strategy.epsilon = 0.1
        test_model(env, strategy, model)

        env.close()


if __name__ == '__main__':
    main()
