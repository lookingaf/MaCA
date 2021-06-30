import numpy as np
import math
import copy

COURSE_NUM = 20
FIGHTER = 1

class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num
        self.img_obs_reduce_ratio = 10
        #self.last_obs =[]

        self.opp_fighter_num = 10

    def obs_construct(self, obs_raw_dict):
        obs_data_dict = dict()
        fighter_data = []
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
        fighter_recv = self.__get_all_recv(fighter_data_obs_list)   #获得我放观测到的所有敌方飞机特征
        alive_status = self.__get_alive_status(detector_data_obs_list, fighter_data_obs_list)
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                alliance_fighter = self.__get_alliance_fighter(fighter_data_obs_list ,x)    #获取我方飞机信息
                alliance_fighter = alliance_fighter.ravel()#将多维数组转化为一维数组

                temp = self.process_data(fighter_data_obs_list[x], fighter_recv)

                temp = np.r_[temp,alliance_fighter]

                alive_status[x] = True
                fighter_data.append({'info':temp,'alive':True})# , "raw_obs" : obs_raw_dict['fighter_obs_list'][x], "all_recv" : fighter_recv })
            else:
                fighter_data.append({'info':[],'alive':False})
        obs_data_dict['fighter'] = fighter_data
        return obs_data_dict


    def process_data(self,fighter_obs, fighter_recv):
        myfighter_recv = copy.deepcopy(fighter_recv)

        oriention = math.radians(((fighter_obs["last_action"]["course"])//18)*18)#将角度转换为弧度
       # print(oriention)
        last_ori_action = np.zeros(2, dtype=np.float32)
        last_ori_action[0] = math.sin(oriention)
        last_ori_action[1] = math.cos(oriention)
       # print(last_ori_action)

        last_att_action = np.zeros(21)
        if fighter_obs["last_action"]['missile_type'] == 1:
            last_att_index = fighter_obs["last_action"]['hit_target']

        elif fighter_obs["last_action"]['missile_type'] == 2:
            last_att_index = fighter_obs["last_action"]['hit_target'] + 10
        else:
            last_att_index = fighter_obs["last_action"]['hit_target']

        last_att_action[last_att_index] =  1


        fighter_missle = np.zeros(2 ,dtype = np.float32)

        fighter_pos = np.zeros( 2, dtype=np.float32 )
        pos_x = fighter_obs['pos_x']
        pos_y = fighter_obs['pos_y']
      #  print("red position", pos_x, pos_y)
        fighter_pos[0] = pos_x  / self.battlefield_size_x
        fighter_pos[1] = pos_y / self.battlefield_size_y

        fighter_missle[0] = fighter_obs['l_missile_left'] / 2         #导弹数量归一化
        fighter_missle[1] = fighter_obs['s_missile_left'] / 4
        myfighter_recv = self.__get_visible(fighter_obs ,myfighter_recv, pos_x, pos_y)

        fighter_striking_list = self.__get_strick_list(fighter_obs)

        myfighter_recv = myfighter_recv.ravel()         #将所有探测到的飞机的特征矩阵变成一行的

        temp = np.r_[fighter_pos ,fighter_missle, last_ori_action, last_att_action ,myfighter_recv, fighter_striking_list ]          #把处理后的所有数据拼接为一行

        return temp


    def __get_all_recv(self,obs):
        fighter_recv = np.zeros((self.opp_fighter_num, 4))      #我方探测到的所有敌方飞机：id，pos_x, pos_y, distance, direction,5个，最后一个不要了        oriention 6个特征值
        for x in range(self.fighter_num):
            if obs[x]['alive']:
                for j in range(len(obs[x]["r_visible_list"])):
                    id = int(obs[x]["r_visible_list"][j]['id']) - 1#探测到的id
                    fighter_recv[id, 0] = FIGHTER   #战机类型为1
                    fighter_recv[id, 1] = obs[x]["r_visible_list"][j]['pos_x']
                    fighter_recv[id, 2] = obs[x]["r_visible_list"][j]['pos_y']
        return fighter_recv


    #处理我方的全局 敌方飞机信息
    def __get_visible(self, fighter_obs, fighter_recv, my_posx, my_posy):
        tempfighter_recv = np.zeros((self.opp_fighter_num, 6))   #自己是否探测到 方向(sinx, cosx  距离 是否在长导弹射程 是否在短导弹射程
        for i in range(self.opp_fighter_num):
            if fighter_recv[i, 0] != 0:       #id不等于0表示探测到了该飞行器
                relative_x, relative_y = fighter_recv[i,1]- my_posx ,fighter_recv[i][2] - my_posy
                dist = math.sqrt(relative_x * relative_x + relative_y * relative_y)

                sin_ori = 0
                cos_ori = 0
                if dist != 0:
                    sin_ori = relative_y / dist
                    cos_ori = relative_x / dist

                tempfighter_recv[i][0] = sin_ori
                tempfighter_recv[i][1] = cos_ori

                tempfighter_recv[i][2] = dist / self.battlefield_size_x
                if dist <= 120:
                    tempfighter_recv[i][3] = 1
                if dist <= 50:
                    tempfighter_recv[i][4] = 1

        for r_recv in fighter_obs['r_visible_list']:    #自己是否探测到
            tempfighter_recv[r_recv['id']-1 ][5] = 1



        for j_recv in fighter_obs['j_recv_list']:       #被动探测的方向和是否探测到
            tempfighter_recv[j_recv['id'] -1 ][0] = math.sin(  math.radians(j_recv['direction']  ))
            tempfighter_recv[j_recv['id'] -1 ][1] = math.cos(  math.radians(j_recv['direction']  ))

            tempfighter_recv[j_recv["id"] -1 ][5] = 1


        return tempfighter_recv


    def __get_alliance_fighter(self,obs, x):
        alliance_fighter = np.zeros((self.fighter_num - 1 ,3))  # 我方其他飞机的信息
        pos_x = obs[x]["pos_x"]
        pos_y = obs[x]["pos_y"]
        count = 0
        for i in range(self.fighter_num):
            if obs[i]["alive"] and i != x:

                relative_x, relative_y = obs[i]["pos_x"] - pos_x, obs[i]["pos_y"] - pos_y

                dist = math.sqrt(relative_x * relative_x + relative_y * relative_y)

                sin_ori = 0
                cos_ori = 0
                if dist != 0:
                    sin_ori = relative_y / dist
                    cos_ori = relative_x / dist

                alliance_fighter[count][0] = sin_ori
                alliance_fighter[count][1] = cos_ori

                alliance_fighter[count][2] = dist / self.battlefield_size_x

                count += 1

        if count != 0:
            tmp_arr = sorted(alliance_fighter[:count], key= lambda k:k[2])   #按照距离远近进行排序
            alliance_fighter[:count] = tmp_arr

        return alliance_fighter

    def __get_strick_list(self, obs):
        fighter_strick_list = np.zeros(10)
        for striking in obs["striking_dict_list"]:
            fighter_strick_list[ striking["target_id"]-1 ] = 1
        return fighter_strick_list

    def __get_alive_status(self,detector_data_obs_list,fighter_data_obs_list):
        alive_status = np.full((self.detector_num+self.fighter_num,1),True)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                alive_status[x][0] = False
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x+self.detector_num][0] = False
        return alive_status
