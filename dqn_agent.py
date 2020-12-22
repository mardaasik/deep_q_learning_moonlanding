import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from collections import deque
import random
import gym
import datetime

class DeepQNetwork():
    def __init__(self, learning_rate, gamma, n_actions, 
    epsilon, batch_size, input_dims, epsilon_decay=0.999, epsilon_end=0.01,
    buffer_size=1000, network_hidden_layer_neurons=[256,256]):
        self.__action_space = [i for i in range(n_actions)]
        self.__n_actions = n_actions
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__batch_size = batch_size
        self.__inputs_dims = input_dims
        self.__epsilon_decay = epsilon_decay
        self.__epsilon_end = epsilon_end
        self.__buffer_size = buffer_size
        self.__network_hidden_layer_neurons = network_hidden_layer_neurons

        # create buffer 
        self.__buffer = deque(maxlen=self.__buffer_size)
        
        # settings for network and initiation
        self.__network = self.__create_model(learning_rate, n_actions,
        input_dims, network_params=self.__network_hidden_layer_neurons)

        # create target network
        self.__target_network = self.__create_model(learning_rate, n_actions,
        input_dims, network_params=self.__network_hidden_layer_neurons)
        
        # copy weights
        self.target_network_update()
    
    def target_network_update(self):
        # copy weights
        self.__target_network.set_weights(self.__network.get_weights())

    def store_experience(self, state, action, reward, new_state, done):
        self.__buffer.append([state, action, reward, new_state, done])

    def choose_action(self, state):
        rand = np.random.random_sample()
        if rand < self.__epsilon:
            action = np.random.choice(self.__action_space)
        else:
            action = np.argmax(self.__network.predict(state))
        
        return action

    # chooses the action without randomness
    def choose_greedy_action(self, state):
        return np.argmax(self.__network.predict(state))
    
    def learn(self):
        # if there is no enough samples, then don't learn
        if len(self.__buffer) < self.__batch_size:
            return
        
        # sample from buffer
        samples = random.sample(self.__buffer, self.__batch_size)


        # seperate the samples into its components
        states = np.zeros((self.__batch_size, self.__inputs_dims))
        actions = np.zeros((self.__batch_size, 1), dtype=np.int32)
        rewards = np.zeros((self.__batch_size, 1))
        new_states = np.zeros((self.__batch_size, self.__inputs_dims))
        dones = np.zeros((self.__batch_size, 1))

        # assign values
        for i, sample in enumerate(samples):
            states[i] = samples[i][0]
            actions[i] = samples[i][1]
            rewards[i] = samples[i][2]
            new_states[i] = samples[i][3]
            dones[i] = int(samples[i][4])

        # calculate values for q_current and next states
        q_values_current = self.__network.predict(states)
        q_values_next = self.__target_network.predict(new_states)

        for i, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            if done:
                q_values_current[i, action] = reward
            else:
                q_values_current[i, action] = reward + self.__gamma*np.max(q_values_next[i])

        # self.__network.fit(states, q_values_current, verbose=0) 
        self.__network.train_on_batch(states, q_values_current) 

        # reassign epsilon after each learning
        self.__reassign_epsilon()

    def save_model(self, file_path):
        self.__network.save(file_path)
    
    def __reassign_epsilon(self):
        self.__epsilon *= self.__epsilon_decay
        self.__epsilon = max(self.__epsilon, self.__epsilon_end)

    def get_epsilon(self):
        return self.__epsilon


    def __create_model(self, learning_rate, n_actions, input_dims, network_params=[256,256]):
        model = keras.Sequential()
        model.add(Input(shape=(input_dims,)))
        for layer_neurons in network_params:
            model.add(Dense(layer_neurons, activation="relu"))
        model.add(Dense(n_actions, activation="linear"))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        print(model.summary())
        return model

    def load_model(self, model_path):
        self.__network = keras.models.load_model(model_path)
        self.target_network_update()


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    agent = DeepQNetwork(learning_rate=0.001, gamma=0.99, n_actions=4, epsilon=1,
        batch_size=32, input_dims=8, buffer_size=100000, epsilon_decay=0.99, network_hidden_layer_neurons=[512, 256, 128])

    target_network_update_every_episode = 4
    n_episodes = 2000
    start_learning_after_episode = 50
    scores = []
    # max_steps_episode = 250

    for episode in range(n_episodes):
        current_state = env.reset().reshape(1,8)
        done = False
        score = 0
        step_counter = 0
        action_list = []
        while not done:
            action = agent.choose_action(current_state)
            action_list.append(action)
            new_state, reward, done, log_info = env.step(action)

            new_state = new_state.reshape(1,8)
            agent.store_experience(current_state, action, reward, new_state, done)
            current_state = new_state

            score += reward

            if episode > start_learning_after_episode:
                agent.learn()
        
            step_counter += 1
            
        if episode % target_network_update_every_episode == 0:
                agent.target_network_update()    
        
        # print(action_list)
        
        scores.append(score)
        print("Episode {0} score {1}, __epsilon {2}, total steps {3}".format(episode, score, agent.get_epsilon(), step_counter))

        if episode % 20 == 0:
            agent.save_model("policies/dqn_v2_moonlanding_1_"+str(episode))