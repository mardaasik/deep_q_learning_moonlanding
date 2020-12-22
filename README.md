# deep_q_learning_moonlanding
A deep Q learning agent to solve "Moon Landing Problem" using Reinforcement Learning

Parameters to create the agent

learning_rate
gamma
n_actions
epsilon
batch_size
input_dims
epsilon_decay=0.999 (default)
epsilon_end=0.01 (default)
buffer_size=1000 (default)
network_hidden_layer_neurons=[256,256] (default)


Additional parameters for episode iteration

target_network_update_every_episode # how often update target network
n_episodes # total number of episodes
start_learning_after_episode # start learning after which episode (Considering the initial experience gathering)
