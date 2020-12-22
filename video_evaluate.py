# import libraries

import gym
import numpy as np
import tensorflow as tf

env = gym.make("LunarLander-v2")
model = tf.keras.models.load_model("policies/dqn_v2_moonlanding_1_140")

def create_policy_eval_video(model, env, num_episodes=5):
    for _ in range(num_episodes):
        state = env.reset().reshape(1,8)
        env.render()
        done = False
        while not done:
            action = np.argmax(model.predict(state)[0])
            new_state, reward, done, _  = env.step(action)
            state = new_state.reshape(1,8)
            env.render()

print(model.summary())
create_policy_eval_video(model, env)