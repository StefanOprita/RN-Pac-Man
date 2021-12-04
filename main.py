import gym

def main():
    env = gym.make('MsPacman-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            print(observation)
            action = 0
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()

if __name__ == '__main__':
    main()