# -*- coding: utf-8 -*-

import re
import numpy as np
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio
from .Parameter import Param
param = Param()
    
def reward_traffic(location, Phase, remaining_time, speedlimit, car_spd, car_a, dis2inter):
    cur_phase = Phase
    next_phase = - (cur_phase - 1)
    brake_max = -2.5
    a_max = 2.5
    r_light = 0
    
    # 初始化
    segment_len = location.endpoint - location.startpoint
    # d_cri = car_spd**2/2/(-brake_max)
    # d_max = min(car_spd*remaining_time + 0.5*a_max*(remaining_time**2), speedlimit * remaining_time)
    d_max = min(car_spd*remaining_time + 0.5*a_max*(remaining_time**2), speedlimit * remaining_time)
    # d_miss = segment_len - dis2inter
    c1 = param.r_tra_ct1
    c2 = param.r_tra_ct2
    c3 = param.r_tra_ct3


    # 然后对靠近路口的状态进行reward
    if cur_phase == 1 :
        # 当前是绿灯，判断最大可行驶距离是否大于到路口距离
        if dis2inter > d_max:
            # 此绿灯过不去
            spd_flag = 2
        else:
            # 此绿灯能过去
            spd_flag = 1
    else:
        # 当前相位为红灯，直接找下一个绿灯窗口
        spd_flag = 0

    # # 根据标志位求不同情况的reward
    # 排除除数为0
    if remaining_time < 0.01 :
        remaining_time = 0.01
        
    if spd_flag == 1:
        # 绿灯过得去应该尽快过
        spd_min = dis2inter/remaining_time
        tar_spd = max(spd_min * c1, speedlimit) 
    elif spd_flag == 2:
        # 绿灯过不去，找下一个绿灯
        hold_time = location.RedTiming + remaining_time
        spd_max = dis2inter/hold_time
        spd_min = dis2inter/(hold_time + location.GreenTiming)
        tar_spd = np.clip(max(spd_max * c2, spd_min), 0, speedlimit)
    else:
        # 红灯，等待变绿灯
        hold_time = location.GreenTiming + remaining_time
        spd_max = dis2inter/remaining_time
        spd_min = dis2inter/hold_time
        # print("===================")
        # print("min", spd_min)
        # print("max", spd_max)
        tar_spd = np.clip(max(spd_max * c3, spd_min), 0, speedlimit)
    
    # if (car_spd > tar_spd) and (car_spd - tar_spd > 1):
    #     tar_spd = car_spd - 3

    # # 调试
    # print("=========================")
    # print("car_spd(ini) input：",car_spd)
    # print("remaining time:", remaining_time)
    # print("dis2inter:",dis2inter)
    # print("tar_spd:",tar_spd)

    error_threshold = 2.5

    if tar_spd > car_spd :
        tar_spd = (tar_spd - car_spd < error_threshold)*tar_spd + (tar_spd - car_spd >= error_threshold)*(car_spd + error_threshold)
    else:
        tar_spd = (car_spd - tar_spd < error_threshold)*tar_spd + (car_spd - tar_spd >= error_threshold)*(car_spd - error_threshold)
    
    if tar_spd > speedlimit :
        tar_spd = speedlimit
    
    return tar_spd

