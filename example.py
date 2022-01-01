import gym
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as anime

env = gym.make("MsPacman-v0")

# starting a new game and plotting the frame

obs = env.reset()
print(obs.shape)  # [ht, wd, channels]

plt.figure(figsize=(7, 7))
plt.imshow(obs)
plt.axis('off')

# 9 possible actions in Pacman : 0,...,8
# 0: no change, 1: up, 2: right, 3: left, 4: down, ...
print('action_space :', env.action_space)


def preprocess(obs):
    img = obs[1:168:2, :: 2, :]  # crop and downsize
    img = img - 128  # normalize between -128 and 127
    img = img.astype('int8')  # saves memory
    return img


print('original obs shape :', obs.shape)
print('new obs shape :', preprocess(obs).shape)
print(obs.min(), obs.max(), preprocess(obs).min(), preprocess(obs).max())
print(preprocess(obs).flatten().shape)

plt.imshow((preprocess(obs) + 128).astype('uint8'))  # plt doesn't plot 'int8'
plt.axis('off')

import tensorflow as tf
from tensorflow.keras.initializers import he_uniform

tf.compat.v1.disable_eager_execution()
input_height = 84
input_width = 80
n_channels = 3
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
conv_strides = [4, 2, 1]
conv_pads = ['SAME'] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = conv_n_maps[2] * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu

# we only allow 4 actions : 1: up, 2: right, 3: left, 4: down
n_outputs = 4  # (in output vector, map is changed to 0: up, 1: right, 2: left, 3: down)

# He-initializer
he_init = he_uniform
outputs_initializer = he_uniform


def q_net(X_inp, name):
    X_inp_normalized = tf.cast(X_inp, dtype='float32') / 128.0

    with tf.compat.v1.variable_scope(name) as scope:
        # all vars names will be prefixed with 'name'

        conv1 = tf.compat.v1.layers.conv2d(X_inp_normalized, filters=conv_n_maps[0], kernel_size=conv_kernel_sizes[0],
                                           strides=conv_strides[0], padding=conv_pads[0],
                                           activation=conv_activation[0], kernel_initializer=he_init)
        conv2 = tf.compat.v1.layers.conv2d(conv1, filters=conv_n_maps[1], kernel_size=conv_kernel_sizes[1],
                                           strides=conv_strides[1], padding=conv_pads[1],
                                           activation=conv_activation[1], kernel_initializer=he_init)
        conv3 = tf.compat.v1.layers.conv2d(conv2, filters=conv_n_maps[2], kernel_size=conv_kernel_sizes[2],
                                           strides=conv_strides[2], padding=conv_pads[2],
                                           activation=conv_activation[2], kernel_initializer=he_init)
        hidden = tf.compat.v1.layers.dense(tf.reshape(conv3, shape=(-1, n_hidden_in)),
                                           n_hidden, activation=hidden_activation, kernel_initializer=he_init)
        outputs = tf.compat.v1.layers.dense(hidden, n_outputs, kernel_initializer=outputs_initializer)

        trainbale_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return outputs, trainbale_vars


tf.compat.v1.reset_default_graph()

X_state = tf.compat.v1.placeholder(dtype='int8', shape=(None, input_height, input_width, n_channels), name='X')

q_vals_online, vars_online = q_net(X_state, 'online')
q_vals_target, vars_target = q_net(X_state, 'target')

assign_ops = []

for i, var in enumerate(vars_target):
    assign_ops.append(tf.compat.v1.assign(var, vars_online[i]))

copy_online_to_target = tf.group(*assign_ops)

# defining loss func and training ops

learning_rate = 0.001

X_action = tf.compat.v1.placeholder('uint8', shape=(None))  # actions taken by online q-net
y = tf.compat.v1.placeholder('float32', shape=(None, 1))  # q-val estimates using target q-net

q_val_action = tf.reduce_sum(q_vals_online * tf.one_hot(X_action, n_outputs),
                             axis=1, keepdims=True)  # [batch_size, 1]

# using L1 loss as in L2 loss the gradients are too unstable
loss = tf.reduce_mean(tf.abs(q_val_action - y))  # L1 loss

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# checking for exploding grads
bottom_grads = optimizer.compute_gradients(loss, var_list=[vars_online[0]])
bottom_grads_norm = tf.norm(bottom_grads, ord=1)  # reflects changes in smaller gradients

# saver
saver = tf.compat.v1.train.Saver()

# init
init = tf.compat.v1.global_variables_initializer()

from collections import deque

max_mem_size = 2000
game_memory = deque([], max_mem_size)  # a queue with max len

epsilon0 = 1.0
epsilon_min = 0.9
eps_decay_steps = 2000000


def epsilon_greedy(step, q_vals_eval):
    # this is more relevant if you have the patience for training the net for a LONG time
    # unfortunately we will not be doing so: hence epsilon_min is kept very high
    epsilon = max(epsilon_min, epsilon0 - (epsilon0 - epsilon_min) * (step / eps_decay_steps))
    if np.random.random() < epsilon:
        return np.random.choice(n_outputs)
    else:
        return np.argmax(q_vals_eval)


def get_next_batch(batch_size, n_steps_train):
    indices = np.random.permutation(-np.arange(1, n_steps_train))[:batch_size]
    (batch_state, batch_action, batch_reward, batch_next_state, batch_continue) = ([], [], [], [], [])
    for idx in indices:
        state, action, reward, next_state, cont = game_memory[idx]
        batch_state.append(state)
        batch_action.append(action)
        batch_reward.append(reward)
        batch_next_state.append(next_state)
        batch_continue.append(cont)
    batch_state = np.array(batch_state)
    batch_next_state = np.array(batch_next_state)
    batch_action = np.array(batch_action).reshape(batch_size, 1)
    batch_reward = np.array(batch_reward).reshape(batch_size, 1)
    batch_continue = np.array(batch_continue).reshape(batch_size, 1)
    return batch_state, batch_action, batch_reward, batch_next_state, batch_continue


n_steps = 5000000  # num steps to train for
n_steps_train = 15  # train online q-net after these many steps
n_steps_copy = 3000  # copy online net to target after these many steps
discount_rate = 0.999999
batch_size = 5
skip_start = 85  # skip the start of every game (it's just waiting time in PacMan)
training_start = 60  # start training after these many steps
timer = 8  # update action after these life steps, this was determined using experiments

sess = tf.compat.v1.InteractiveSession()
init.run()

step = 0
done = True  # has a game ended?
game_memory = deque([], max_mem_size)

# vars for keeping track of training progress
opt_actions = deque([], n_steps_train)
game_reward = 0  # total reward scored in a game
n_games = 0
last_10_losses = deque([], 10)  # a queue for storing last 10 training losses

from collections import Counter

while step < n_steps:
    step += 1

    if done:
        # start a  new game
        obs = env.reset()
        # skip the start of each game as in pacman its just waiting
        for skip in range(skip_start):
            obs, reward, done, info = env.step(0)
        state = preprocess(obs)
        lives_left = 3
        game_reward = 0
        life_step = 0
        n_games += 1

    # determine the action using e-greedy policy on online q_net
    q_vals_eval = q_vals_online.eval(feed_dict={X_state: state.reshape(1, input_height, input_width, n_channels)})[0]
    opt_actions.append(1 + np.argmax(q_vals_eval))

    # update the action every 8 life steps
    if life_step % timer == 0:
        action = epsilon_greedy(step, q_vals_eval)

    # execute the action
    (obs, reward, done, info) = env.step(action + 1)
    next_state = preprocess(obs)
    life_step += 1
    game_reward += int(reward)

    # record this step in game memory
    game_memory.append((state, action, reward, next_state, 1 - done))

    # skip the start of each new life as in pacman its just waiting
    if lives_left != info['lives']:
        lives_left = info['lives']
        life_step = 0
        for skip in range(35):
            obs, reward, done, info = env.step(0)

    if step < training_start:
        # there are not enough game memories to train usefully
        continue

    if step % n_steps_train == 0:
        # train the online q_net
        batch_state, batch_action, batch_reward, batch_next_state, batch_continue = get_next_batch(batch_size,
                                                                                                   n_steps_train)
        # estimate the q_online vals using reward and q_target val
        q_target_eval = q_vals_target.eval(feed_dict={X_state: batch_next_state})  # [batch_size, n_outputs]
        q_target_eval_max = np.max(q_target_eval, axis=1).reshape(-1, 1)  # [batch_size, 1]
        q_est = batch_reward + discount_rate * q_target_eval_max * batch_continue  # [batch_size, 1]
        loss_eval, _, b_grads_norm = sess.run([loss, training_op, bottom_grads_norm],
                                              feed_dict={X_state: batch_state,
                                                         X_action: batch_action.reshape(-1), y: q_est}
                                              )
        # queue the loss
        last_10_losses.append(loss_eval)

        if step % 300 == 0:
            # print the mean of last 10 train losses, bottom grads norm , etc
            print(step, ':', '%.1f' % np.mean(last_10_losses), dict(Counter(opt_actions)), b_grads_norm, end='  ')
            opt_actions = deque([], n_steps_train)

        # exploding bottom layer grads
        if loss_eval > 2e14 or b_grads_norm > 2e14:
            print('-------------DIVERGED----------------')
            break

    if step % n_steps_copy == 0:
        # copy online q_net vars to target q_net
        sess.run(copy_online_to_target)
        print('|copy|', end=' ')

# playing using the online q-net

import time

# sess = tf.InteractiveSession()
# saver.restore(sess, './datasets/Pacman/pacman.ckpt')

n_max_steps = 5000
frames = []  # will store the game frames, can be converted to video
game_reward = 0

# start a  new game
obs = env.reset()

# skip the start of each life as in pacman its just waiting
for skip in range(skip_start):
    obs, reward, done, info = env.step(0)

lives_left = 3

for stp in range(n_max_steps):
    state = preprocess(obs)
    # determine the action using online q_net
    q_vals_eval = q_vals_online.eval(feed_dict={X_state: state.reshape(1, input_height, input_width, n_channels)})[0]
    action = np.argmax(q_vals_eval)
    # execute the action
    obs, reward, done, info = env.step(1 + action)

    # skip the start of each life as in pacman its just waiting
    if lives_left != info['lives']:
        lives_left = info['lives']
        for skip in range(35):
            obs, reward, done, info = env.step(0)

    print(1 + action, end=' ')
    frames.append(obs)  # if you want to store the game as a vid
    game_reward += int(reward)
    env.render()
    time.sleep(0.05)  # to slowdown the game
    # check if game ended
    if done:
        break

print('game_score :', game_reward)
