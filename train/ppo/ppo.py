#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Lookingaf
@software: PyCharm
@file: ppo.py
@time: 2021.1.19.14:48:54
@desc: model of ppo
"""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import math
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#放在GPU上，如果能分配cuda就使用cuda：0，采用的是通过字符串分配方式
#device = torch.device('cpu')
FIGHTER_NUM = 10
REPLACE_TARGET_ITER = 100
action_course_dim = 22
action_target_dim = 21

class ActorCritic(nn.Module):
    def __init__(self,n_actions):
        super(ActorCritic,self).__init__()

        self.action_feature = nn.Sequential(
            nn.Linear(124, 256),  # 层一，statedim为输入样本大小，n_latent_var为输出样本大小，输入层
            nn.Tanh(),  # 激活函数，非线性激活函数的引入，使得模型能解决非线性问题
            nn.Linear(256, 256),  # 层二，隐藏层
            nn.Tanh(),  # 激活函数
        )
        self.action_1 = nn.Sequential(
            nn.Linear(256, action_course_dim),
            nn.Softmax(-1)
                                      )
        self.action_4 = nn.Sequential(
            nn.Linear(256, action_target_dim),
            nn.Softmax(-1)
        )


        # critic
        self.value_layer = nn.Sequential(  # 建立神经网络
            nn.Linear(124, 256),  # 层一输入层
            nn.Tanh(),  # 激活函数
            nn.Linear(256, 256),  # 层二隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(256, 1)  # 层三输出层
        )

    def forward(self):
        raise NotImplementedError #指定抛出的异常名称，以及异常信息的相关描述。
        # 父类中可以预留一个接口不实现，要求在子类中实现。如果一定要子类中实现该方法，可以使用raise NotImplementedError报错
        # 如果子类没有实现父类中指定要实现的方法，则会自动调用父类中的方法，而父类方法又是raise将错误抛出。这样代码编写者就能发现是缺少了对指定接口的实现
        # 没有实际内容，用于被子类方法覆盖

    def action(self,state,states_memory,actions_cou_memory,actions_tar_memory,cou_logprobs_memory,tar_logprobs_memory):
        state = torch.tensor(state).float().to(device)
        feature = self.action_feature(state)

        action_course_probs = self.action_1(feature)
        dist_cou = Categorical(action_course_probs)
        action_cou = dist_cou.sample()

        action_target_probs = self.action_4(feature)

        dist_tar = Categorical(action_target_probs)
        action_tar = dist_tar.sample()
        action_ori_logprobs = dist_cou.log_prob(action_cou)
        action_att_logprobs = dist_tar.log_prob(action_tar)
        action = np.array([int(action_cou.cpu().numpy() * 17), 1, 11, int(action_tar.cpu().numpy())])

        action_ori_logprobs = torch.where(torch.isnan(action_ori_logprobs), torch.full_like(action_ori_logprobs, 0.01),
                                          action_ori_logprobs)
        action_att_logprobs = torch.where(torch.isnan(action_att_logprobs), torch.full_like(action_att_logprobs, 0.01),
                                          action_att_logprobs)
        states_memory.append(state)
        actions_cou_memory.append(action_cou)
        actions_tar_memory.append(action_tar)
        cou_logprobs_memory.append(action_ori_logprobs)#将其加入存储
        tar_logprobs_memory.append(action_att_logprobs)  # 将其加入存储
        return action

    def evaluate(self,state,action_cou,action_tar):  # 评价函数
        state =state.float().to(device)

        feature = self.action_feature(state)

        action_course_probs = self.action_1(feature)
        dist_course = Categorical(action_course_probs)

        action_target_probs = self.action_4(feature)
        dist_target = Categorical(action_target_probs)


        action_ori_logprobs = dist_course.log_prob(action_cou)  # logit回归模型
        action_att_logprobs = dist_target.log_prob(action_tar)  # logit回归模型
        action_ori_logprobs = torch.where(torch.isnan(action_ori_logprobs), torch.full_like(action_ori_logprobs, 0.01),
                                          action_ori_logprobs)
        action_att_logprobs = torch.where(torch.isnan(action_att_logprobs), torch.full_like(action_att_logprobs, 0.01),
                                          action_att_logprobs)
        dist_course_entropy = dist_course.entropy()  # 熵
        dist_target_entropy = dist_target.entropy()  # 熵
        dist_entropy = 0.5*dist_course_entropy + 0.5*dist_target_entropy

        state_value = self.value_layer(state)

        return action_ori_logprobs, action_att_logprobs, state_value.squeeze(), dist_entropy  # squeeze用于压缩维度


class PPOFighter:
    def __init__(
            self,
            n_actions,
            learning_rate,
            betas,
            gamma,
            eps_clip,
            K_epochs,
            max_episodes,
            log_interval,
            max_timesteps
    ):
        self.n_actions = n_actions
        self.lr = learning_rate#
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.epsilon_max = max_episodes
        self.interval = log_interval
        self.max_timesteps = max_timesteps
        self.learn_step_counter = 0
        self.fighter_alive_status = []
        self.actions_course_memory = []
        self.actions_target_memory = []
        self.states_memory = []
        self.course_logprobs_memory = []  # log概率
        self.target_logprobs_memory = []
        self.rewards_memory = []
        self.is_terminals_memory = []
        self.memory_counter = 0

        self.gpu_enable = torch.cuda.is_available()



        self.policy = ActorCritic(n_actions).to(device)
        self.optimirer = torch.optim.Adam(self.policy.parameters(),lr=learning_rate,betas=betas)
        self.policy_old = ActorCritic(n_actions).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()  # 均方误差标准

    def store_transition(self, reward,is_terminal,fighter_alive_status):
        self.rewards_memory.append(reward)
        self.is_terminals_memory.append(is_terminal)
        self.fighter_alive_status.append(fighter_alive_status)

    def _clear_memory(self):
        self.states_memory.clear()
        self.actions_course_memory.clear()
        self.actions_target_memory.clear()
        self.rewards_memory.clear()
        self.course_logprobs_memory.clear()
        self.target_logprobs_memory.clear()
        self.fighter_alive_status.clear()
        self.is_terminals_memory.clear()


    def choose_action(self, state):
        action = self.policy_old.action(state,self.states_memory,self.actions_course_memory,self.actions_target_memory,self.course_logprobs_memory,self.target_logprobs_memory)
        # if self.gpu_enable:
        #     action = action.cpu()
        # action = action.numpy()
        return action


    def discount_reward(self,step):
        rewards = []
        discounted_reward = 0
        temp_rewards = []
        temp_is_terminals = []
        temp_fighter_alive_status = []
        for i in range(10):
            for reward, is_terminal, fighter_alive_status in zip(self.rewards_memory, self.is_terminals_memory,
                                                                 self.fighter_alive_status):
                temp_rewards.append(reward[i])
                temp_is_terminals.append(is_terminal[i])
                temp_fighter_alive_status.append(fighter_alive_status[i])
        self.rewards_memory = copy.deepcopy(temp_rewards)
        self.is_terminals_memory = temp_is_terminals
        self.fighter_alive_status = temp_fighter_alive_status
        for reward, is_terminal, fighter_alive_status in zip(reversed(self.rewards_memory),
                                                             reversed(self.is_terminals_memory), reversed(
                    self.fighter_alive_status)):  # 返回给定序列值的反向迭代器，zip打包成一个个元组
            if not fighter_alive_status:
                rewards.insert(0, 0)
                continue
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)  # 如果是最后一个，discounted为rewards，否则需要乘上γ再加reward
            rewards.insert(0, discounted_reward)
        temp_rewards.clear()
        for temp_step in range(0, step + 1):
            for i in range(10):
                if self.fighter_alive_status[temp_step + i * (step + 1)]:
                    temp_rewards.append(rewards[temp_step + i * (step + 1)])
        rewards = torch.tensor(temp_rewards).float().to(device)
        rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-5) ).squeeze()  # 将reward减去平均数后除以标准差（方差的算术平方根）
        rewards = torch.where(torch.isnan(rewards), torch.full_like(rewards, temp_rewards[0]),
                                          rewards)
        return rewards


    def stack_memory(self):
        return torch.stack(self.states_memory).to(device),torch.stack(self.actions_course_memory).to(device),torch.stack(self.actions_target_memory).to(device),torch.stack(self.course_logprobs_memory).to(device),torch.stack(self.target_logprobs_memory).to(device)

    def learn(self,step):
        if self.learn_step_counter % REPLACE_TARGET_ITER == 0:
            step_counter_str = '%09d' % self.learn_step_counter
            torch.save(self.policy.state_dict(), 'model/ppo/model_' + step_counter_str + '.pkl')

        rewards = self.discount_reward(step)
        # convert list to tensor  沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        old_states,old_course_actions,old_target_actions,old_course_logprobs,old_target_logprobs = self.stack_memory()
        for i in range(self.K_epochs):  # 迭代
            # Evaluating old actions and values :对过去进行评估
            course_logprobs, target_logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_course_actions,old_target_actions)  # 调用AC的评估函数
            advantages = rewards - state_values.detach()
            # Finding the ratio (pi_theta / pi_theta__old):
            ori_ratios = torch.exp(course_logprobs - old_course_logprobs.detach())
            att_ratios = torch.exp(target_logprobs - old_target_logprobs.detach())
            ratios = (0.5 * ori_ratios + 0.5 * att_ratios)
            loss = (-torch.min(ratios * advantages, torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages) + 0.5 * self.MseLoss(rewards.float(),state_values.float() ) - 0.01 * (dist_entropy) )
            self.optimirer.zero_grad()  # 清空所有被优化过的Variable的梯度.
            loss.mean().backward()  # 反向传播计算，根据误差进行反向传播计算，根据参数进行调整，不断迭代，进行收敛
            self.optimirer.step()  # 进行单次优化 (参数更新).
            self.learn_step_counter += 1

            # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class PPODetevtor:
    def __init__(
            self,
            n_actions,
    ):
        self.n_actions = n_actions
        self.gpu_enable = torch.cuda.is_available()

        self.netdetector = ActorCritic(self.n_actions)
        if self.gpu_enable:
            print('GPU Available!!')
            self.netdetector = self.netdetector.cuda()
            self.netdetector.load_state_dict(torch.load('model/ppo/model.pkl'))
        else:
            self.netdetector.load_state_dict(torch.load('model/ppo/model.pkl', map_location=lambda storage, loc: storage))


    def choose_action(self, img_obs, info_obs):
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()
        action = self.netdetector.action(img_obs, info_obs)
        if self.gpu_enable:
            action = action.cpu()
        action = action.numpy()
        return action

