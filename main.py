import gym
import numpy as np
from gym.envs.classic_control import rendering


def repeat_upsample(rgb_array, k=1, repeat_times=1):
    # repeat kinda crashes if k/repeat_times are zero
    if k <= 0 or repeat_times <= 0:
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), repeat_times, axis=1)


def main():
    viewer = rendering.SimpleImageViewer()
    env = gym.make('MsPacman-v0')
    env.reset()

    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            rgb = env.render('rgb_array')
            upscale = repeat_upsample(rgb, 4, 4)
            viewer.imshow(upscale)

            print(observation.shape)
            action = 0
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == '__main__':
    main()
