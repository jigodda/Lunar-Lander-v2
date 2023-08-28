import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

env = gym.make("LunarLander-v2")

num_observations = env.observation_space.shape[0]
num_actions = env.action_space.n

###############################################
### Building Model ###

model = Sequential()
model.add(Dense(64, input_shape = (1, num_observations)))
model.add(Activation('relu'))

model.add(Dense(num_actions))
model.add(Activation('linear'))

target_model = clone_model(model)

###############################################
### Hyperparameters and other constants ###

epsilon = 1
learning_rate = 0.001 #NOTE: NOT ALPHA
GAMMA = 0.99

BATCH_SIZE = 32
EPOCHS = 1000
EPSILON_REDUCE = .995
MIN_EPSILON = .1

UPDATE_TARGET_MODEL = 1000
PRINT_TIME = 25

totalSteps = 0

#Returns an action to take, either randomly (exploration) or by referencing the model (exploitation)
def epsilonGreedyActionSelection(model, epsilon, observation):
    #exploitation
    if np.random.random() > epsilon: 
        observation = tf.expand_dims(tf.convert_to_tensor(observation), axis=0)
        prediction = model(observation) 
        action = np.argmax(prediction)
    #exploration    
    else:
        action = np.random.randint(0, env.action_space.n)

    return action

replay_buffer = deque(maxlen = 20000)

#replay function
def replay(replay_buffer, batch_size, model, target_model):
    #if you are not at the maximum size of the replay_buffer, exit the function
    if len(replay_buffer) < batch_size:
        return

    samples = random.sample(replay_buffer, batch_size)

    #unpacks variables from samples
    zipped_samples = list(zip(*samples))
    states, actions, rewards, new_states, dones = zipped_samples

    #predicts values
    targets = target_model.predict(np.array(new_states), verbose = 0)
    q_values = model.predict(np.array(states), verbose = 0)
    #print(q_values)
    #print(q_values[0][0][actions[0]])
    #loops over predicted values
    for i in range(batch_size):


        #q_value = max(q_values[i][0])
        target_value = max(targets[i])
        target_value = max(target_value)
        # target = targets[i].copy()
        if dones[i]:
            #target[0][actions[i]] = rewards[i]
            q_values[i][0][actions[i]] = rewards[i]
        else:
            #target[0][actions[i]] + q_value * GAMMA #
            q_values[i][0][actions[i]] = q_values[i][0][actions[i]] + target_value * GAMMA 


        #target_batch.append(target)
            
    model.fit(np.array(states), np.array(q_values), epochs = 1, verbose = 0)       

# Every 10 epochs, updates the weights of target model to the weights of the model's current weight
def update_model_handler(steps, UPDATE_TARGET_MODEL, model, target_model):
    if steps >= UPDATE_TARGET_MODEL:
        target_model.set_weights(model.get_weights())
        return
model.compile(loss = 'huber')
mostSteps = 0

# Calcualtes average reward of the epoch
def avgReward(runningTotal):
    currentRunEpoch = epoch % PRINT_TIME
    avg = runningTotal / (currentRunEpoch+1)
    return avg

totalAvg = 0
realTotalReward = 0
currentPrint = 0
runningTotal = 0 # Total reward of the last printTime runs
peakReward = -9999 # Highest ever reward achieved
lastFourAvg = [0, 0, 0, 0]


for epoch in range(EPOCHS):
    observation = env.reset()
    observation = observation.reshape([1,8])
    done = False

    stepsTaken = 0
    rewardTotal = 0

    while not done:
        action = epsilonGreedyActionSelection(model, epsilon, observation)
        next_observation, reward, done, info = env.step(action)
        next_observation = next_observation.reshape([1,8])

        replay_buffer.append((observation, action, reward, next_observation, done))
        rewardTotal += reward
        observation = next_observation
        stepsTaken += 1
        totalSteps += 1
        if totalSteps >= UPDATE_TARGET_MODEL:
            update_model_handler(totalSteps, UPDATE_TARGET_MODEL, model, target_model)
            totalSteps = 0
        if(rewardTotal > peakReward):
            peakReward = rewardTotal
        #env.render()
        replay(replay_buffer, BATCH_SIZE, model, target_model)
    update_model_handler(totalSteps, UPDATE_TARGET_MODEL, model, target_model)
    runningTotal += rewardTotal

    if stepsTaken > mostSteps:
        mostSteps = stepsTaken

    
    #print results every PRINT_TIME epochs
    if epoch % PRINT_TIME == 0 and epoch != 0:
        avg = avgReward(rewardTotal)
        
        lastFourAvg[int((epoch / PRINT_TIME) - 1) % 4] = avg

        realTotalReward = lastFourAvg[0] + lastFourAvg[1] + lastFourAvg[2] + lastFourAvg[3]
        totalAvg = realTotalReward / 4
        runningTotal = 0
        print(f"{epoch}: Average Reward: {avg} Last 100 Avg: {totalAvg} Peak Reward: {peakReward} eps: {epsilon} ")   

    
    if(epsilon  > MIN_EPSILON):
        epsilon *= EPSILON_REDUCE

    
