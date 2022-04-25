class Param():
	def __init__(self):
		# 训练参数
		self.step_size = 1
		self.var = 3.0
		self.exploration_decay_start_step = 40000
		self.MAX_EPISODE = 999999
		self.var_threshold = 0.009 # 结束训练的var条件
		self.var_update_step = 20
		self.var_multi = 0.9999

		# DDPG参数
		self.replay_buffer_size = 4000000
		self.replay_buffer_start = 4000
		self.batch_size = 256
		self.gamma = 0.99

		# 数据存储参数
		self.write = 1
		self.pltsoc = 50
		self.pltenv = 50
		self.pltreward = 100

		# reward参数
		self.r_fuel_cf = -0.8
		self.r_fuel_ct = 5
		self.r_soc_bat1 = -0.5
		self.r_soc_bat2 = -10
		self.illegal_soc_punish = -10
		self.r_spd_cv1 = 0.002
		self.r_spd_cv2 = -0.05
		self.r_spd_cv3 = -0.2
		self.illegal_spd_punish = -10
		self.r_ille_ci1 = -0.6
		self.r_ille_ci2 = -0.4
		self.r_ille_ci3 = -10 # 惩罚SOC为0的
		self.r_tra_ct1 = -0.25
		self.r_tra_ct2 = -0.2
		self.r_tra_ct3 = -0.1
		self.r_suc_cs = 0.01

		# 网络参数
		self.layer1_size = 400
		self.layer2_size = 300
		self.layer3_size = 300
		self.learning_rate = 0.0001
		self.TAU = 0.001