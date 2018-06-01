## Architecture

# Take in inputs from the screen and preprocess them
# Pass them into an NN
# Update the weights of the NN using gradient descent
# weights['1'] - Matrix that holds weights of pixels passing into hidden layer. Dimensions: [200 x 80 x 80] -> [200 x 6400]
# weights['2'] - Matrix that holds weights of hidden layer passing into output. Dimensions: [1 x 200]

# Process is:

# processed_observations = image vector - [6400 x 1] array
# Compute hidden_layer_values = weights['1'] dot processed_observations ([200 x 6400] dot [6400 x 1]) -> [200 x 1] - this gives initial activation values.
# Next we need to transform those either via a sigmoid or an ReLU of some sort. Let's use ReLU
# ReLU(hidden_layer_values)
# Next we need to pass this one layer further
# output_layer_value = weights['2'] dot hidden_layer_values ([1 x 200] dot [200 x 1] -> [1 x 1])
# Now our output layer is the probability of going up or down. Let's make sure this output is between 0 and 1 by passing it through a sigmoid
# p = sigmoid(output_layer_value)

# Learning after round has finished:

# Figure out the result
# Compute the error
# Use the error to calculate the gradient
    # The below dimensions all assume we had exactly 10 frames in the round (not necessarily true!)
    # dC_dw2 = hidden_layer_values^T dot gradient_log_p ([1 x 2000] dot [2000 x 1] -> 1x1)
    # delta_1 = gradient_log_p outer_product weights['2'] = [2000 x 1] outer_product [1 x 200] ([2000 x 200])
    # dC_dw1 = delta_1^T dot input_observations ([200 x 2000]x dot [2000 x 64000] -> [200 x 64000])

# After some batch size of rounds has finished,
    # Use rmsprop to move weights['1'] and weights['2'] in the direction of the gradient
# Repeat!

import gym
import numpy as np
import pickle 
import time 
import os 

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3

def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

def ret_time(filename):
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    toResume = False        # Whether or not to build off saved model
    # hyperparameters
    episode_number = 0
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-3

    episode_number = int(filename[:-2]) if toResume else 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None
    if toResume:
        weights = pickle.load(open(filename, 'rb'), encoding='bytes')
    else:
        weights = {
            '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
            '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
        }

    # This file stores data from each run for plotting later 
    fd = open('time.csv','a')
    count = 1
    start_time = time.time()
    #env.render()
    while (count > 0):

        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        if done: # an episode finished
            count -= 1

    duration = time.time() - start_time
    average_duration = duration 
    myCsvRow = str(episode_number) + "," + str(average_duration) + "\n"
    fd.write(myCsvRow)

def main():
    all_files = os.listdir()
    saved_files = [i for i in all_files if i[-2:] == '.p']
    # for i in saved_files:
    #     ret_time(i)
    ret_time('0.p')




main()