from audioop import avg
from sys import api_version
from .Parameter1 import Param
from .State_Machine import reward_traffic
import gym
import os
import random as rand
import numpy as np
import math
import scipy.io as scio
from scipy.interpolate import interp1d, interp2d
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

import csv

#####################  PATH   ###########################
# reward_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'reward')
reward_dir = 'D:/RL/ElegantRL-master/PPO_Prius_Simplified/reward'
if not os.path.exists(reward_dir):
	os.makedirs(reward_dir)
# data_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data')
data_dir = 'D:/RL/ElegantRL-master/PPO_Prius_Simplified/data'
if not os.path.exists(data_dir):
	os.makedirs(data_dir)
# SOC_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'soc')
SOC_dir = 'D:/RL/ElegantRL-master/PPO_Prius_Simplified/soc'
if not os.path.exists(SOC_dir):
	os.makedirs(SOC_dir)
# result_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'Result')
result_dir = 'D:/RL/ElegantRL-master/PPO_Prius_Simplified/Result'
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
#########################################################

param = Param()

class My_Env_simp(gym.Env):
	def __init__(self):
		# 道路类对象实例化
		self.RoadSegmentList = []
		# # 训练的数据
		# self.RoadSegmentList.append(Road(1, 60, 0, 525, 31, 49, -20))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(2, 60, 525, 1480, 28, 51, -20))       # 白龙江东街
		# self.RoadSegmentList.append(Road(3, 60, 1480, 2005, 30, 82, -20))      # 河西大街
		# 随机相位尝试训练
		# self.RoadSegmentList.append(Road(1, 60, 0, 525, 30, 50, rand.randint(-50,50)))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(2, 60, 525, 1480, 40, 35, rand.randint(-50,50)))       # 白龙江东街
		# self.RoadSegmentList.append(Road(3, 60, 1480, 2005, 20, 60, rand.randint(-50,50)))      # 河西大街
		# # 参陈浩师兄多路口数据
		# self.RoadSegmentList.append(Road(1, 60, 0, 1200, 40, 60, 0))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(2, 80, 1200, 2200, 40, 65, 0))       # 白龙江东街
		# self.RoadSegmentList.append(Road(3, 80, 2200, 3700, 42, 65, 0))      # 河西大街
		# self.RoadSegmentList.append(Road(4, 80, 3700, 5100, 45, 55, 0))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(5, 80, 5100, 6400, 40, 65, 0))       # 白龙江东街
		# self.RoadSegmentList.append(Road(6, 80, 6400, 8000, 55, 67, 0))      # 河西大街
		# 东大实验路径
		self.RoadSegmentList.append(Road(1, 20, 0, 550, 60, 40, 0))         # 嘉陵江东街 2.5
		self.RoadSegmentList.append(Road(2, 20, 550, 1060, 50, 50, 30))       # 白龙江东街 7.3
		self.RoadSegmentList.append(Road(3, 20, 1060, 1650, 40, 65, 20))      # 河西大街 22.5
		# 庐山路实际数据
		# self.RoadSegmentList.append(Road(1, 60, 0, 326, 31, 49, -20))         # 嘉陵江东街
		# self.RoadSegmentList.append(Road(2, 65, 326, 679, 28, 51, -20))       # 白龙江东街
		# self.RoadSegmentList.append(Road(3, 60, 679, 1005, 30, 82, -20))      # 河西大街
		# self.RoadSegmentList.append(Road(4, 60, 1005, 1320, 38, 50, -30))      # 楠溪江东街
		# self.RoadSegmentList.append(Road(5, 60, 1320, 1606, 38, 50, 0))     # 富春江大街
		# RoadSegmentList.append(Road(6, 60, 1606, 1935, 27, 61, -30))     # 奥体大街
		# RoadSegmentList.append(Road(7, 60, 1935, 2232, 38, 50, 0))     # 新安江街
		# # RoadSegmentList.append(Road(8, 60, 2232, 2500, 99, 1, 0))     # 终点路段  
		# 车辆对象实例化
		self.Prius = Prius_model()

		# 环境量初始化
		self.s_dim = 7
		self.a_dim = 1
		self.t_total = 0
		self.STEP_SIZE = param.step_size
		self.displacement = 0
		self.dis2inter = self.RoadSegmentList[0].endpoint                   # 初始化成刚好在通信范围外，是否合适有待验证
		self.dis2termin = self.RoadSegmentList[len(self.RoadSegmentList) - 1].endpoint
		self.travellength = self.RoadSegmentList[len(self.RoadSegmentList) - 1].endpoint
		self.location = self.RoadSegmentList[0].segment_id                      # 目前车辆所在路段编号
		self.speedlimit = self.RoadSegmentList[0].maxspeed                  # 道路限速初始化
		self.signal_flag = 0                   # 能否收到相位配时的标志位
		self.max_step = int(self.travellength/self.STEP_SIZE)
		# 观测量数组初始化
		self.s = np.zeros(self.s_dim)

		# 重要状态初始化
		self.SOC_min = 0.4
		self.SOC_max = 0.8
		self.SOC_origin = 0.75
		self.isdone = 0
		self.target_return = 999999
		self.if_discrete = False
		self.reset_flag = 1

		# 需要导出的数据存储列表
		self.car_spd_list = []
		self.car_spd_list.append("Speed")
		self.tar_spd_list = []
		self.tar_spd_list.append("Tar_spd")
		self.car_a_list = []
		self.car_a_list.append("Acce")
		self.SOC_data = []
		self.SOC_data.append("SOC")

		self.Eng_spd_list = []
		self.Eng_spd_list.append("Eng_spd")
		self.Eng_trq_list = []
		self.Eng_trq_list.append("Eng_trq")
		self.Gen_spd_list = []
		self.Gen_spd_list.append("Gen_spd")
		self.Gen_trq_list = []
		self.Gen_trq_list.append("Gen_trq")
		self.Mot_spd_list = []
		self.Mot_spd_list.append("Mot_spd")
		self.Mot_trq_list = []
		self.Mot_trq_list.append("Mot_trq")

		self.Phase_list = []
		self.Phase_list.append("Phase")
		self.remaining_time_list = []  
		self.remaining_time_list.append("RemainTime")
		self.signal_flag_list = []
		self.signal_flag_list.append("SignalFlag")
		self.dis2inter_list = []
		self.dis2inter_list.append("dis2inter")

		self.location_list = []
		self.location_list.append("Location")

		self.displacement_list = []
		self.displacement_list.append("s")
		self.t_list = [] 
		self.t_list.append("time")

		self.action_list = []   
		self.action_list.append("action")  

		self.eq_fuel_cost_list = []
		self.eq_fuel_cost_list.append("eq-fuel-cost")
		self.Reward_list_all = []
		self.Reward_list_all.append("AllReward")

		self.I_list = []
		self.I_list.append("illegal")

		# 记录每个episode里每一step的单项reward,需要在episode开始时清空
		self.r_fuel_list = []
		self.r_fuel_list.append("r_fuel")
		self.r_soc_list = []
		self.r_soc_list.append("r_soc")
		self.r_tra_list = []
		self.r_tra_list.append("r_tra")
		self.r_spd_list = []
		self.r_spd_list.append("r_spd")
		self.r_illegal_list = []
		self.r_illegal_list.append("r_ill")
		self.r_success_list = []
		self.r_success_list.append("r_suc")

		# 用来画单项reward变化的数组
		self.mean_fuel_list = []
		self.mean_soc_list = []
		self.mean_tra_list = []
		self.mean_spd_list = []		
		self.mean_ill_list = []
		self.mean_suc_list = []	

		# DRL-EMS源码中的列表，我不一定用到的
		self.cost_Engine_list = []
		self.cost_all_list = []
		self.cost_Engine_100Km_list = []
		self.mean_reward_list = [] # 他不能随着episode增加而reset，他要一直记录到最后
		self.list_even = []
		self.list_odd = []
		self.mean_discrepancy_list = []
		self.SOC_final_list = []

	
	def reset(self): # 初始化
		# # 随机相位配时分布初始化
		# self.reset_flag += 1
		# # 随机相位范围
		# random_range = 15
		# if self.reset_flag % 1 == 0 :
		# 	print("Changing SPaT")
		# 	self.RoadSegmentList[0].delay = rand.randint(-random_range,random_range)
		# 	self.RoadSegmentList[1].delay = rand.randint(-random_range,random_range)
		# 	self.RoadSegmentList[2].delay = rand.randint(-random_range,random_range)
		# 	self.RoadSegmentList[3].delay = rand.randint(-random_range,random_range)
		# 	self.RoadSegmentList[4].delay = rand.randint(-random_range,random_range)
		# 	self.RoadSegmentList[5].delay = rand.randint(-random_range,random_range)

		car_spd = 0.36                      # KM/h
		car_spd = car_spd/3.6             # m/s
		car_a = 0
		self.Prius.car_spd = car_spd
		self.Prius.car_a = car_a
		SOC = self.SOC_origin
		self.t_total = 0
		self.displacement = 0

		# 初始化普锐斯
		self.Prius.car_spd = car_spd
		self.Prius.car_a = car_a
		# self.Prius.Eng_spd = self.Eng_spd
		# self.Prius.Gen_spd = self.Gen_spd
		# self.Prius.Mot_spd = self.Mot_spd
		self.Prius.SOC = SOC
		self.Prius.Eng_spd = 1000 * math.pi / 30

		# 计算初始状态下合理的转速分配
		s = np.zeros(self.s_dim)
		# normalized input
		self.s[0] = car_spd/self.RoadSegmentList[0].maxspeed
		self.s[1] = np.clip(abs(car_a)/2.5, 0., 1.)
		self.s[2] = 0
		self.s[3] = self.RoadSegmentList[0].endpoint / (self.RoadSegmentList[0].endpoint) # dis2inter/路段长
		self.s[4] = self.RoadSegmentList[len(self.RoadSegmentList) - 1].endpoint / self.travellength # dis2termin/总长
		self.s[5], timing = self.RoadSegmentList[0].SPaT(0)
		self.s[6] = (s[5] == 1)*timing/self.RoadSegmentList[0].GreenTiming + (s[5] == 0)*timing/self.RoadSegmentList[0].RedTiming
		self.isdone = 0
		self.target_return = 0
		# 清空数据记录列表
		self.car_spd_list = []
		self.car_spd_list.append("Speed")
		self.tar_spd_list = []
		self.tar_spd_list.append("Tar_spd")
		self.car_a_list = []
		self.car_a_list.append("Acce")
		self.SOC_data = []
		self.SOC_data.append("SOC")

		self.Eng_spd_list = []
		self.Eng_spd_list.append("Eng_spd")
		self.Eng_trq_list = []
		self.Eng_trq_list.append("Eng_trq")
		self.Gen_trq_list = []
		self.Gen_trq_list.append("Gen_trq")
		self.Mot_trq_list = []
		self.Mot_trq_list.append("Mot_trq")

		self.Phase_list = []
		self.Phase_list.append("Phase")
		self.remaining_time_list = []  
		self.remaining_time_list.append("RemainTime")
		self.signal_flag_list = []
		self.signal_flag_list.append("SignalFlag")
		self.dis2inter_list = []
		self.dis2inter_list.append("dis2inter")

		self.location_list = []
		self.location_list.append("Location")

		self.displacement_list = []
		self.displacement_list.append("s")
		self.t_list = [] 
		self.t_list.append("time")

		self.action_list = []   
		self.action_list.append("action")  

		self.cost_Engine_list = []
		self.cost_Engine_list.append("fuel_cost")
		self.eq_fuel_cost_list = []
		self.eq_fuel_cost_list.append("eq-fuel-cost")
		self.Reward_list_all = []
		self.Reward_list_all.append("AllReward")

		self.I_list = []
		self.I_list.append("illegal")

		self.r_fuel_list = []
		self.r_fuel_list.append("r_fuel")
		self.r_soc_list = []
		self.r_soc_list.append("r_soc")
		self.r_tra_list = []
		self.r_tra_list.append("r_tra")
		self.r_spd_list = []
		self.r_spd_list.append("r_spd")
		self.r_illegal_list = []
		self.r_illegal_list.append("r_ill")
		self.r_success_list = []
		self.r_success_list.append("r_suc")

		# DRL-EMS源码中的列表，我不一定用到的
		# self.cost_Engine_list = [] # 这些都是以episode为单位记录的，不可以被reset方法在每个episode开始时重置
		# self.cost_all_list = []
		# self.cost_Engine_100Km_list = []
		# self.mean_reward_list = []
		# self.list_even = []
		# self.list_odd = []
		# self.mean_discrepancy_list = []
		# self.SOC_final_list = []

		return s
	
	def step(self, action): # 离散步长内的所有计算都发生在这里
		# 调整转矩动作
		# 智能体使用tanh激活，动作范围是-1到1
		self.Prius.car_a = action * self.Prius.car_a_max
		##### 确定上一步长末端的道路限速 #####
		for h in range(len(self.RoadSegmentList)):                                     # 定位车辆在哪段路上，返回该段路的最高时速
			# print("displacement = ",self.displacement)
			# print("h段终点：", self.RoadSegmentList[h].endpoint)
			if (self.displacement <= self.RoadSegmentList[h].endpoint) and (self.displacement > self.RoadSegmentList[h].startpoint):
				speedlimit_last = self.RoadSegmentList[h].maxspeed
				#print("目前车辆行驶在ROAD ", h)
				break
			if self.displacement == self.RoadSegmentList[h].startpoint:
				speedlimit_last = self.RoadSegmentList[h].maxspeed
				break
			h = h + 1
		##### 更新车辆运动学动力学参数 #####
		fuel_cost_rate, P_req, t = self.Prius.run(self.STEP_SIZE)
		car_a = float(self.Prius.car_a)
		car_spd = float(self.Prius.car_spd)
		# 用更新值覆盖原速度

		##### 获得油耗率 #####
		fuel_cost_rate = float(fuel_cost_rate)

		##### 更新位移 #####
		# 排除车速异常，如果出现需要修正一下时间和位移
		stop_flag = 0
		if (t == 0) :
			# print("停车了，妈的")
			stop_flag = 1
			self.displacement = self.displacement
		else:
			self.displacement = self.displacement + self.STEP_SIZE
		
		self.dis2termin = self.travellength - self.displacement
		
		##### 车辆定位 #####
		for h in range(len(self.RoadSegmentList)):                                     # 定位车辆在哪段路上，返回该段路的最高时速
			# print("displacement = ",self.displacement)
			# print("h段终点：", self.RoadSegmentList[h].endpoint)
			# if (self.displacement < self.RoadSegmentList[h].endpoint) and (self.displacement >= self.RoadSegmentList[h].startpoint):
			if (self.displacement <= self.RoadSegmentList[h].endpoint) and (self.displacement > self.RoadSegmentList[h].startpoint):
				location = self.RoadSegmentList[h]
				# print("目前车辆行驶在ROAD ", h)
				break
			if self.displacement == self.travellength :
				location = self.RoadSegmentList[len(self.RoadSegmentList)-1]
				break

			h = h + 1
		speedlimit = location.maxspeed

		##### 更新时间 #####
		self.t_total = self.t_total + t
		self.t_total = float(self.t_total)

		##### 更新SPaT #####
		Phase, remaining_time = location.SPaT(self.t_total)
		dis2inter_ini = self.dis2inter # 这一步长初始的 到路口距离
		self.dis2inter = location.endpoint - self.displacement # 这一步长结束的 到路口距离
		#signal_flag_list.append(signal_flag)

		##### 计算reward #####
		# # 求r_fuel
		fuel_cost = get_fuel(fuel_cost_rate, t) 
		fuel_cost = float(fuel_cost) # g
		fuel_cost = fuel_cost / 0.72 / 1000 # L
		cf = param.r_fuel_cf
		ct = 0 # param.r_fuel_ct
		r_fuel = cf * (fuel_cost + ct) 

		# # 求r_soc
		# cbat1 = param.r_soc_bat1
		# cbat2 = param.r_soc_bat2
		# r_SOC = (self.displacement < self.travellength) * (cbat1) * (max(SOC_new - self.SOC_max, 0) + max(self.SOC_min - SOC_new, 0)) +\
		# 		(self.displacement == self.travellength) * (cbat2) * max(SOC_new - self.SOC_origin, 0)
		# if SOC_new < self.SOC_min :
		# 	# 惩罚SOC过低，避免负数
		# 	r_SOC = param.illegal_soc_punish

		###### 速度跟踪方案 ######
		# 目标车速应该以该步长的 初始速度、初始到路口距离 和 步长内的加速度计算，用来评价该步长动作的可靠性
		tar_spd = reward_traffic(location, Phase, remaining_time, speedlimit_last, self.s[0]*speedlimit_last, car_a, dis2inter_ini)
		r_moving = self.travellength*(self.s[4] - self.dis2termin/self.travellength)*param.r_spd_cv1 +\
				param.r_spd_cv2 * 0.025 * ((car_spd - tar_spd)**2) +\
				param.r_spd_cv3 * ((car_a - self.s[1]*2.5)**2 + (car_spd - self.s[0]*speedlimit_last)**2) +\
				(car_spd > speedlimit) * (-10) + (car_spd < 1/3.6)*(-10)
		r_moving = 0.8 * r_moving

		# ###### 日常奖励 ######
		# r_moving = 200*(self.s[4] - self.dis2termin/self.travellength)*param.r_spd_cv1 +\
		# 	 	((car_spd - speedlimit)**2 + (car_spd - 10/3.6)**2)* param.r_spd_cv2 +\
		# 		param.r_spd_cv3 * ((car_a - self.s[1]*2.5)**2+ (car_spd - self.s[0]*speedlimit)**2) # 加速度和速度的一致性约束

		# # 结算奖励
		# 求r_success
		# 计算得通过三个路口的奖励应该分别为3 20 55
		r_sucess = 0
		if Phase == 1 and self.displacement == location.endpoint and car_spd != 0 :
			r_sucess = (location.endpoint == 550)*5 + (location.endpoint == 1060)*8.5 + (location.endpoint == 1650)*27.5
		if Phase == 0 and self.displacement == location.endpoint and car_spd != 0 :
			r_sucess = -1*((location.endpoint == 550)*3 + (location.endpoint == 1060)*3 + (location.endpoint == 1650)*3)

		# 求r_illegal
		# 普锐斯模型输出I记录了模型内部的转速、加速度 越界情况；取值1 2 4 9分别对应发动机、发电机、电动机、加速度越界
		# I为和式，如取值为3则是发动机和发电机都发生了越界
		ci1 = param.r_ille_ci1 # 参考Automated eco-driving in urban scenarios using deep reinforcement learning 
		ci2 = param.r_ille_ci2
		# ci3 = param.r_ille_ci3
		# r_illegal = (car_spd > speedlimit) * (-1) + (car_spd == 0.1) * (-1) # ci1 * ((car_a > 0)*(car_a - 2.5)**2 + (car_a < 0)*(car_a + 2.5)**2) 
		
		r_illegal = (car_spd > speedlimit) * (0)

		self.isdone = self.isdone + r_sucess

		# 求总reward
		# r_fuel = float(r_fuel)
		# r_SOC = float(r_SOC)
		# r_spd = float(r_spd)
		# r_tra = float(r_tra)
		# r_spd = float(r_spd)
		r_illegal = float(r_illegal)
		r_sucess = float(r_sucess)
		r = r_moving + r_fuel + r_sucess + r_illegal 
		r = float(r)

		##### 存新状态 #####
		s_ = np.zeros(self.s_dim)
		s_[0] = car_spd/speedlimit
		s_[1] = car_a/2.5
		s_[2] = P_req/self.Prius.Prius_max_pwr
		s_[3] = self.dis2inter / (location.endpoint - location.startpoint)
		s_[4] = self.dis2termin / self.travellength
		s_[5] = Phase
		s_[6] = (Phase == 1) * remaining_time / location.GreenTiming + (Phase == 0) * remaining_time / location.RedTiming

		###### 更新类属性 #####
		self.s[0] = s_[0]
		self.s[1] = s_[1]
		self.s[2] = s_[2]
		self.s[3] = s_[3]
		self.s[4] = s_[4]
		self.s[5] = s_[5]
		self.s[6] = s_[6]

		##### 结束标志 #####
		# done = self._get_done()
		done = False
		# if self.isdone == 78 and np.mean(self.car_spd_list) <= 17:
		# 	self.target_return = 1

		info = {} # 用于记录训练过程中的信息,便于观察训练状态
		info['fuel_cost'] = fuel_cost
		info['car_spd'] = car_spd
		info['tar_spd'] = tar_spd
		info['car_a'] = car_a
		info['phase'] = Phase
		info['timing'] = remaining_time
		info['dis2inter'] = self.dis2inter
		info['displacement'] = self.displacement
		info['t'] = self.t_total
		info['delta_t'] = t
		info['dis2termin'] = self.dis2termin
		info['r'] = r
		info['r_fuel'] = r_fuel
		# info['r_soc'] = r_SOC
		# info['r_tra'] = r_tra
		# info['r_spd'] = r_spd
		info['r_moving'] = r_moving # 这个量写到原来r_spd_list里面
		info['r_ill'] = r_illegal
		info['r_suc'] = r_sucess
		info['location'] = location.segment_id

		return s_, r, done, info

	def out_info(self, info): # 将所有需要输出到csv的量append到列表并导出
		# action已经在主程序里面append了
		self.car_spd_list.append(info['car_spd'])
		self.tar_spd_list.append(info['tar_spd'])
		self.car_a_list.append(info['car_a'])
		# self.SOC_data.append(info['SOC'])
		# self.Eng_spd_list.append(info['eng_spd'])
		# self.Eng_trq_list.append(info['eng_trq'])
		# self.Gen_spd_list.append(info['gen_spd'])
		# self.Gen_trq_list.append(info['gen_trq'])
		# self.Mot_spd_list.append(info['mot_spd'])
		# self.Mot_trq_list.append(info['mot_trq'])
		# self.Eng_alp_list.append(info['eng_alp'])
		# self.Gen_alp_list.append(info['gen_alp'])
		# self.Mot_alp_list.append(info['mot_alp'])

		self.Phase_list.append(info['phase'])
		self.remaining_time_list.append(info['timing'])
		self.dis2inter_list.append(info['dis2inter'])
		self.displacement_list.append(info['displacement'])
		self.t_list.append(info['t'])
		self.cost_Engine_list.append(info['fuel_cost'])
		self.Reward_list_all.append(info['r'])

		self.r_fuel_list.append(info['r_fuel'])
		# self.r_soc_list.append(info['r_soc'])
		# self.r_tra_list.append(info['r_tra'])
		# self.r_spd_list.append(info['r_spd'])
		self.r_spd_list.append(info['r_moving'])
		self.r_illegal_list.append(info['r_ill'])
		self.r_success_list.append(info['r_suc'])
		self.location_list.append(info['location'])
	
	def write_info(self, i, is_training):
		global data_dir
		if is_training :
			print("under training mode")
		if is_training == False and i == 0 : # 目录只重写一次，避免反复调用write造成的反复目录重写
			print("under testing mode, select new dir")
			data_dir = data_dir + '/testing_data'
			if not os.path.exists(data_dir):
				os.makedirs(data_dir)

		filename = "/eposide " + str(i) + ".csv"
		f = open(data_dir + filename ,'w')
		csv_write = csv.writer(f)
		csv_write.writerow(self.action_list)
		csv_write.writerow(self.Eng_trq_list)
		# csv_write.writerow(self.Eng_alp_list)
		csv_write.writerow(self.Eng_spd_list)
		csv_write.writerow(self.Gen_trq_list)
		# csv_write.writerow(self.Gen_alp_list)
		csv_write.writerow(self.Gen_spd_list)
		csv_write.writerow(self.Mot_trq_list)
		# csv_write.writerow(self.Mot_alp_list)
		csv_write.writerow(self.Mot_spd_list)
		csv_write.writerow(self.car_spd_list)
		csv_write.writerow(self.tar_spd_list)
		csv_write.writerow(self.car_a_list)
		csv_write.writerow(self.SOC_data)
		csv_write.writerow(self.displacement_list)
		csv_write.writerow(self.location_list)
		# csv_write.writerow(signal_flag_list)
		csv_write.writerow(self.t_list)
		csv_write.writerow(self.Phase_list)
		csv_write.writerow(self.remaining_time_list)
		csv_write.writerow(self.Reward_list_all)
		csv_write.writerow(self.cost_Engine_list)

		csv_write.writerow(self.r_fuel_list)
		# csv_write.writerow(self.r_soc_list)
		# csv_write.writerow(self.r_tra_list)
		csv_write.writerow(self.r_spd_list)
		csv_write.writerow(self.r_illegal_list)
		csv_write.writerow(self.r_success_list)
		f.close()
	
	def write_mean_reward(self):
		filename = "/mean_reward_list " + ".csv"
		f = open(data_dir + filename ,'a')
		csv_write = csv.writer(f)
		csv_write.writerow(self.mean_reward_list)
		f.close()

	def write_sum_success_reward(self, cost_engine): # cost_engine是测试代码中的一个局部变量
		# # RL agent记录-无速度跟踪
		# del self.r_success_list[0]
		# sum_success = np.sum(self.r_success_list)
		# if sum_success == 105 :
		# 	duration = self.t_total
		# 	distribution_data = [duration, float(cost_engine)]
		# 	filename = "/RL_no_follow_list_for_distribution " + ".csv"
		# 	f = open(data_dir + filename ,'a')
		# 	csv_write = csv.writer(f)
		# 	csv_write.writerow(distribution_data) # 写入一个episode的r_success和
		# 	f.close()		
		# # IDM记录
		# 	duration = self.t_total
		# 	if cost_engine < 10 :
		# 		distribution_data = [duration, cost_engine]
		# 		filename = "/IDM_list_for_distribution " + ".csv"
		# 		f = open(data_dir + filename ,'a')
		# 		csv_write = csv.writer(f)
		# 		csv_write.writerow(distribution_data) # 写入一个episode的r_success和
		# 		f.close()	
		# RL agent记录-速度跟踪
		del self.r_success_list[0]
		sum_success = np.sum(self.r_success_list)
		if sum_success == 105 :
			duration = self.t_total
			distribution_data = [duration, float(cost_engine)]
			filename = "/RL_follow_list_for_distribution " + ".csv"
			f = open(data_dir + filename ,'a')
			csv_write = csv.writer(f)
			csv_write.writerow(distribution_data) # 写入一个episode的r_success和
			f.close()

	def render(self, i, is_training):
		global result_dir
		global reward_dir
		global SOC_dir
		if is_training :
			print("Under training mode")
		if is_training == False and i == 0 : # 目录只重写一次，避免反复调用Render造成的反复目录重写
			print("Under testing mode, select new dir")
			result_dir = result_dir + '/testing_result'
			if not os.path.exists(result_dir):
				os.makedirs(result_dir)
			reward_dir = reward_dir + '/testing_reward'
			if not os.path.exists(reward_dir):
				os.makedirs(reward_dir)
			SOC_dir = SOC_dir + '/testing_soc'
			if not os.path.exists(SOC_dir):
				os.makedirs(SOC_dir)

		episode_num = i
		t_list = self.t_list
		displacement_list = self.displacement_list
		# delete text in the front of the list
		del t_list[0]
		del displacement_list[0]
		time = int(max(t_list))+10
		# # 调试
		# print(time)
		###################################### 画运动轨迹 ###########################################
		# draw the environment
		for t in range(time):
			for j in range(len(self.RoadSegmentList)):
				Phasej, remainingj = self.RoadSegmentList[j - 1].SPaT(t)
				y = self.RoadSegmentList[j - 1].endpoint
				if Phasej == 0:
					plt.scatter(t, y, s=2*2, marker = "s", c = '#FF0000')
				else:
					plt.scatter(t, y, s=2*2, marker = "s",c = '#008000')
		# draw the trajectory
		plt.plot(t_list, displacement_list, linewidth = '1', linestyle = '-', c = 'b')
		print("画完运动轨迹了")

		plt.xlabel("Time/s")
		plt.ylabel("Distance/m")
		picname = result_dir + '/' + str(episode_num) + ".png"
		plt.savefig(picname, dpi = 500, bbox_inches='tight')
		plt.close('all')
		###################################### 画SOC情况 ###########################################
		temp_soc_data = self.SOC_data
		del temp_soc_data[0]
		x = np.arange(0, len(temp_soc_data), 1)
		y = temp_soc_data
		plt.plot(x,y)
		plt.xlabel('distance')
		plt.ylabel('SOC')
		picname = SOC_dir + '/' + str(episode_num) + '.png'
		plt.savefig(picname, dpi = 500, bbox_inches='tight')
		# 自动调整坐标轴
		plt.tight_layout()
		plt.close() 
		###################################### 画reward情况 ###########################################		
		x = np.arange(0, len(self.mean_reward_list), 1)
		y_spd = self.mean_spd_list
		y_all = self.mean_reward_list
		y_fuel = self.mean_fuel_list
		line_spd, = plt.plot(x, y_spd, linewidth = '1', linestyle = '-', label ='r_spd', c = 'black')
		line_all, = plt.plot(x, y_all, linewidth = '1', linestyle = '-', label ='r', c = 'red')
		line_fuel, = plt.plot(x, y_fuel, linewidth = '1', linestyle = '-', label ='r_fuel', c = 'yellow')
		plt.xlabel('Epsoide')
		plt.ylabel('Mean_Reward')
		plt.legend([line_all, line_spd, line_fuel], ['r', 'r_spd', 'r_fuel'], loc = 'lower right')
		picname = reward_dir+ '/' + str(episode_num) + '.png'
		plt.savefig(picname, dpi = 500, bbox_inches='tight')
		# 自动调整坐标轴
		plt.tight_layout()
		plt.close() 

class Road(object):
	# 规定路的编码，速度上限，全局坐标下的起点、终点，绿灯时间，红灯时间
	def __init__(self, id, maxspeed, startpoint, endpoint, GreenTiming, RedTiming, delay):
		self.segment_id = id
		self.maxspeed = maxspeed/3.6
		self.startpoint = startpoint
		self.endpoint = endpoint
		self.GreenTiming = GreenTiming
		self.RedTiming = RedTiming
		self.delay = delay 

	def SPaT(self, t):
		# 一个红灯相位和一个绿灯相位组成一个SPaT,先红后绿
		t = t + self.delay
		Green = self.GreenTiming
		Red = self.RedTiming
		time = t % (Green + Red)
		if time <= Red:
			# 现在是红灯相位
			Phase = 0
			remaining_time = Red - time
		else:
			# 现在是绿灯相位
			Phase = 1
			remaining_time = Red + Green - time

		return Phase, remaining_time

class Prius_model():
	# Gen3 Prius with a gear retio in MG2
    def __init__(self):
        # # Prius parameters
        # # paramsmeters of car
        # self.Wheel_R = 0.287
        # self.mass = 1450
        # self.C_roll  = 0.015
        # self.density_air = 1.2
        # self.area_frontal = 2.52
        # self.G = 9.81
        # # self.C_d = 0.28
        # self.C_d = 0.25
        # # the factor of F_roll
        # self.T_factor = 1
        # # 发动机怠速转速 100rad/s -> 955 rpm
        # self.idling_speed = 100
        # # 传动系效率
        # self.trans_eta = 0.95
        # # 旋转质量换算系数
        # self.rotating_index = 1 # 可变
		# # 最大整车输出功率
        # self.Prius_max_pwr = 100 * 1000

        # LittleAnt parameters
        # paramsmeters of car
        self.Wheel_R = 0.29775
        self.mass = 1080
        self.C_roll  = 0.015
        self.density_air = 1.2
        self.area_frontal = 2.5885
        self.G = 9.81
        # self.C_d = 0.28
        self.C_d = 0.29
        # the factor of F_roll
        self.T_factor = 1
        # 发动机怠速转速 100rad/s -> 955 rpm
        self.idling_speed = 100
        # 传动系效率
        self.trans_eta = 0.95
        # 旋转质量换算系数
        self.rotating_index = 1 # 可变
		# 最大整车输出功率
        self.Prius_max_pwr = 30 * 1000

        # 车辆运动学状态
        self.car_spd = 0
        self.car_a = 0
        self.car_a_max = 2.5
        self.SOC = 0

        # 匹配下层转矩控制器，定义动力总成属性
        self.Eng_spd = 0
        self.Gen_spd = 0
        self.Mot_spd = 0
        self.Eng_trq = 0
        self.Gen_trq = 0
        self.Mot_trq = 0
        
    def run(self, STEP_SIZE):                         
        # 计算需求功率
        F_roll = self.mass * self.G * self.C_roll * (self.T_factor if self.car_spd > 0 else 0)                    # 行驶方程式考虑滚动阻力，空气阻力，加速阻力，坡度为0
        F_drag = 0.5 * self.density_air * self.area_frontal * self.C_d *(self.car_spd ** 2)
        F_a = self.mass * self.car_a
        F_req = F_roll + F_drag + F_a
        P_req = F_req * self.car_spd

        # # Prius等效能耗模型
        # # 七工况DP数据的等效油耗模型 参考鞠飞师兄
        # p1 = -3.084e-20
        # p2 = -3.149e-15
        # p3 = 4.247e-10
        # p4 = 5.819e-05
        # p5 = 0.07485

        # fuel_cost_rate = p1 * P_req **4 + p2 * P_req ** 3 + p3 * P_req **2 + p4 * P_req + p5

        # # 小蚂蚁纯电动汽车能耗模型
        # # 参考刘昊吉师兄ITSC论文
        P_motor = self.car_a * self.car_spd + (0.87*self.Wheel_R**2/9.7**2)*self.car_a**2
        fuel_cost_rate = P_motor * 216 / 371.6  / 9.8

        car_spd_ini = self.car_spd
        if car_spd_ini < 0.1 :
            car_spd_ini = 0.1

        # 更新末速度
        if (2*self.car_a*STEP_SIZE + car_spd_ini**2) > 0 :                                                                  # 先存当前步长初速度
            self.car_spd = math.sqrt(2*self.car_a*STEP_SIZE + car_spd_ini**2)
        else:
            self.car_spd = 0.1

        if self.car_spd < 0.1 :
            self.car_spd = 0.1

        if self.car_a != 0 :
            t = (self.car_spd - car_spd_ini)/self.car_a
        else :
            t = (STEP_SIZE/car_spd_ini)

        return fuel_cost_rate, P_req, t

def get_fuel(fuel_cost_rate, t):
	# # 取等效因子
	# s = get_EF(delta_SOC)
	# # 取SOC惩罚
	# K_soc = SOC_punishment(delta_SOC)
	# 计算离散步长内发动机油耗
	fuel_cost = fuel_cost_rate * t # 单位g
	# # 计算离散步长内电池能量和等效油耗
	# bat_cost = bat_pwr * t # 单位j
	# bat2fuel_cost = K_soc * s * (bat_cost / 42600) # 单位g
	# 离散步长内总等效油耗
	# eq_fuel_cost = fuel_cost + bat2fuel_cost
	return fuel_cost

# 建立SOC惩罚函数，使用S型函数
def SOC_punishment(delta_SOC):
	# SOC惩罚函数参数
	a = 1
	b = 0.95
	K_soc = 1 - a * delta_SOC**3 + b * delta_SOC**4 # 曲线参数参考：华南理工大学李晓甫学位论文
	return K_soc

# 建立等效因子
def get_EF(delta_SOC):
	# if delta_SOC > 0 :
	#     s = 2.0
	# else:
	#     s = 0.8
	s = 2.6 # 参考：Self-Adaptive Equivalent Consumption Minimization Strategy for Hybrid Electric Vehicles 常数ECMS方案

	return s

def calculate_speed(car_spd):
	# 基本参数
	Wheel_R = 0.287
	mass = 1449
	C_roll  = 0.013
	density_air = 1.2
	area_frontal = 2.23
	G = 9.81
	C_d = 0.26
	# the factor of F_roll
	T_factor = 0.015
	
	# paramsmeters of transmission system
	# number of teeth
	Pgs_R = 2.6                        # 齿圈齿数
	Pgs_S = 1                        # 太阳轮齿数
	# speed ratio from ring gear to wheel  从齿圈到车轮的传动比(主减速器传动比)
	Pgs_K = 3.93  
	Pgs_M = 2.63

	Eng_spd = 1500 * 2 * math.pi /60
	Wheel_spd = car_spd/Wheel_R
	ring_spd = Wheel_spd * Pgs_K
	mg2_spd = ring_spd * Pgs_M
	mg1_spd = (1 + Pgs_R/Pgs_S)*Eng_spd - ring_spd * Pgs_R/Pgs_S 

	# print("Mg2-spd",ring_spd)
	# print("Eng-spd",Eng_spd)
	# print("mg1-spd",mg1_spd)

	return Eng_spd, mg1_spd, mg2_spd, Wheel_spd
