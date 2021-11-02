import random
import gym
import pickle
from replay_buffer import ReplayBuffer
from wrappers import *


import torch

from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation

import numpy as np
import sys

def run_episode(env):
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    from MyAgent import MyAgent
    agent = MyAgent(env.observation_space,env.action_space,replay_buffer,use_double_dqn=hyper_params['use-double-dqn'],lr=hyper_params["learning-rate"], batch_size=hyper_params["batch-size"], gamma=hyper_params['discount-factor'])

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    loss = [0.0]
    #torch.save(agent.policy_network.state_dict(),'./policy_network')

#~~~
    done = False
    episode_return = 0.0
    state = env.reset()


#~~~

    for t in range(hyper_params["num-steps"]):
        if(t%500000==0):
            torch.save(agent.policy_network.state_dict(),'./policy_network_'+str(t))
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()
        # TODO 
        #  select random action if sample is less equal than eps_threshold
        # take step in env
        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        # add reward to episode_reward
        if sample > eps_threshold:
            action = agent.act(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        episode_return += reward
        if done:
            state = env.reset()
            episode_return = 0.0
            episode_rewards.append(0.0)

        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            agent.optimise_td_loss()

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params[
            "print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("episode returns: {}".format(episode_return))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
    
    torch.save(agent.policy_network.state_dict(),'./policy_network')
    #with open("dqn_agent_saved.pkl", "wb") as pickle_out:
    #    pickle.dump(agent, pickle_out)
    #    print("pickle done")


    return episode_return

if __name__ == '__main__':

    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(2e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.1,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    #assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    #env = gym.make(hyper_params["env"])
    
    #env.seed(hyper_params["seed"])
    error_occurred = False
    # In this example we use the seeds used for evaluating submissions
    # to the Obstacle Tower Challenge.
    eval_seeds = [1, 2, 3]

    # Create the ObstacleTowerEnv gym and launch ObstacleTower
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    worker_id = 10 # int(np.random.randint(999, size=1))
    print(worker_id)
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', docker_training=False, worker_id=worker_id, retro=True,
                           realtime_mode=False, config=config)

    # Wrap the environment with the ObstacleTowerEvaluation wrapper
    # and provide evaluation seeds.
    



    #env = NoopResetEnv(env, noop_max=30)
    #env = MaxAndSkipEnv(env, skip=4)
    #env = EpisodicLifeEnv(env)
    #env = FireResetEnv(env)
    # TODO Pick Gym wrappers to use
    #
    #
    #
    env = PyTorchFrame(env)
    #env = WarpFrame(env)
    #env = ClipRewardEnv(env)
    #env = FrameStack(env, 4)

    episode_rew = run_episode(env) 
    #env = ObstacleTowerEvaluation(env, eval_seeds)



    '''while not env.evaluation_complete:
        try:
            episode_rew = run_episode(env)
        except Exception as exception:
            with open('error_evaluation.txt', 'a') as error_file:
                error_file.write('\n' + str(exception.msg) + '\n')
                error_occurred = True
            break'''
    
    # Finally the evaluation results can be fetched as a dictionary from the
    # environment wrapper.
    env.close()
    if error_occurred:
        print(-100.0)
    else:
        print(env.results['average_reward']*10000)

