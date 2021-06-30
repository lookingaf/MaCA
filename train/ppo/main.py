import os
import copy
import numpy as np
import torch
import sys
from agent.fix_rule.agent import Agent
from interface import Environment
from train.ppo import ppo
MAP_PATH = 'maps/1000_1000_fighter10v10.map'
RENDER = True
MAX_EPOCH = 200000
BATCH_SIZE = 10
LR = 0.001                   # learning rate
EPSILON = 0.2               # greedy policy
GAMMA = 0.99                # reward discount
TARGET_REPLACE_ITER = 999   # target update frequency
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1 # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM
LEARN_INTERVAL = TARGET_REPLACE_ITER
BETAS = (0.9, 0.999)
EPS_clip = 0.2
K_epochs = 4
max_timesteps = 300

if __name__ == "__main__":
    # create blue agent
    blue_agent = Agent()
    # get agent obs type
    red_agent_obs_ind = 'ppo'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    fighter_model = ppo.PPOFighter(ACTION_NUM,LR,BETAS,GAMMA,EPS_clip,K_epochs,MAX_EPOCH,LEARN_INTERVAL,max_timesteps)
    fighter_model.policy.load_state_dict(torch.load('model/ppo/model_000026500.pkl', map_location='cpu'))
    fighter_model.policy_old.load_state_dict(torch.load('model/ppo/model_000026500.pkl', map_location='cpu'))

    reward_sum = []
    round_sum = []
    # execution
    for x in range(MAX_EPOCH):
        if x % 100 == 0:
            print('第%d局'%x)
        step_cnt = 0
        reward_temp = 0
        env.reset()
        while True:
            obs_list = []
            action_list = []
            red_fighter_action = []
            # get obs
            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()
            # get action
            # get blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)
            # get red action
            obs_got_ind = [False] * red_fighter_num
            for y in range(red_fighter_num):
                true_action = np.array([0, 0, 0, 0], dtype=np.int32)
                if red_obs_dict['fighter'][y]['alive']:
                    obs_got_ind[y] = True
                    tmp_state_obs = red_obs_dict['fighter'][y]['info']
                    tmp_action = fighter_model.choose_action(tmp_state_obs)
                    # action formation
                    true_action = tmp_action
                red_fighter_action.append(true_action)
            red_fighter_action = np.array(red_fighter_action)
            # step
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward
            reward_temp += np.sum(fighter_reward)
            is_terminal = [False] * (len(fighter_reward))
            if env.get_done():
                is_terminal = [True] * (len(fighter_reward))
            elif (step_cnt > 0) and (step_cnt % LEARN_INTERVAL == 0):
                is_terminal = [True] * (len(fighter_reward))
            for y in range(red_fighter_num):
                if not obs_got_ind[y]:
                    is_terminal[y] = True
            fighter_model.store_transition(fighter_reward,is_terminal,obs_got_ind)
            # save repaly
            red_obs_dict, blue_obs_dict = env.get_obs()


            # if done, perform a learn
            if env.get_done():
                # detector_model.learn()
                reward_sum.append(reward_temp)
                round_sum.append(x)
                reward_sum_temp = np.array(reward_sum)
                round_sum_temp = np.array(round_sum)
                if x % 20 == 0:
                    np.save('reward_sum.npy',reward_sum_temp)
                    np.save('round_sum.npy',round_sum_temp)
                fighter_model.learn(step_cnt)
                fighter_model._clear_memory()
                break
            # if not done learn when learn interval
            if (step_cnt > 0) and (step_cnt % LEARN_INTERVAL == 0):
                # detector_model.learn()
                fighter_model.learn(step_cnt)
                fighter_model._clear_memory()
            step_cnt += 1