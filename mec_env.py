import numpy as np
import queue
from scipy.special import jv

class Offload:

    def __init__(self, args):

        # INPUT DATA
        self.epsilon = 0.9
        self.n_iot = args.num_iot
        self.n_edge = args.num_edge
        self.n_time = args.num_time
        self.n_size = args.task_size
        self.current_time = 0
        self.last_action = 0
        self.current_iot = 0
        self.queue_limit = 1000
        self.wait_max = 10
        self.gamma = args.gamma
        self.ddpg_gamma = 1.5
        self.LOGNORMAL = False
        if args.FRACTION == 1:
            self.FRACTION = True
        else:
            self.FRACTION = False

        self.wait_state = np.zeros(1)
        self.STATE_STILL = args.STATE_STILL


        # CONSIDER A SCENARIO RANDOM IS NOT GOOD
        # LOCAL CAP SHOULD NOT BE TOO SMALL, OTHERWISE, THE STATE MATRIX IS TOO LARGE (EXCEED THE MAXIMUM)

        self.comp_cap_iot = args.comp_iot
        if args.comp_edge < 0.001:
            self.comp_cap_edge = args.comp_cap_edge
        else:
            self.comp_cap_edge = args.comp_edge * np.ones(self.n_edge)
        self.comp_density  = args.comp_density
        self.n_size = args.task_size # TASK SIZE / M bits

        # ACTION: 0, local; 1, edge 0; 2, edge 1; ...; n, edge n - 1
        self.n_actions = 1 + self.n_edge
        self.n_features = self.n_edge
        self.wait_features = 1

    def execute_action(self, RL, state):
        action = RL.choose_action(state)
        return action

    def random_action(self):
        action = np.random.randint(self.n_actions)
        return action

    def auto_action(self, state):
        index = np.argmin(state)
        if state[index] < 4:
            action = index + 1
        else:
            action = 0
        return action


    def iot_process(self, size, compacity, density):
        expected_time = size * density / compacity
        if self.LOGNORMAL:
            actual_time = np.random.lognormal(np.log(expected_time), self.lognormal_variance)
        else:
            actual_time = np.random.exponential(expected_time)
        # actual_time = expected_time
        
        return actual_time, expected_time

    def edge_process(self, size, compacity, density):
        expected_time = size * density / compacity
        if self.LOGNORMAL:
            actual_time = np.random.lognormal(np.log(expected_time), self.lognormal_variance)
        else:
            actual_time = np.random.exponential(expected_time)
        # actual_time = expected_time

        return actual_time, expected_time

    def Rayleigh(self, h_last):
        T = 1
        fd = 10
        rou = jv(0, 2*np.pi*fd*T)
        alpha = 1
        p = 1
        B = 1
        data_size = 0.1
        device_num = self.n_iot

        e = complex(np.random.randn(), np.random.randn()/np.sqrt(2))
        h = rou*h_last + np.sqrt(1-rou*rou)*e
        g = abs(h) * abs(h) * alpha
        sigma2 = 0.1
        gamma = np.zeros(device_num)
        for i in range(device_num):
            gamma[i] = g[i] * p / (sum(g)*p - g[i]*p + sigma2)

        C = B * np.log2(1+gamma)   
        transmit_time = data_size / C
        return transmit_time, h

    def reset(self, RL):
        self.current_time = 0
        self.action_store = np.zeros(self.n_iot, dtype=int)

        #AoI
        self.duration_store = np.zeros([self.n_iot, 2])
        self.aoi = np.zeros(self.n_iot)
        self.aoi_sum = np.zeros(self.n_iot)
        self.aoi_average = np.zeros(self.n_iot)
        self.wait_time = np.zeros(self.n_iot)
        self.old_aoi = 0.0
        self.new_aoi = 0.0
        
        

        self.queue_indicator = np.zeros(self.n_iot, dtype=int)

        # QUEUE INITIALIZATION: 0 -> iot; 1 -> remain time
        # edge节点等待计算队列
        self.Queue_edge_wait = list()
        for edge in range(self.n_edge):
            self.Queue_edge_wait.append(queue.Queue())

        # TASK INDICATOR
        # iot task 0:remain 1:iot 2:edge+1 3:start_time
        self.task_iot = list()
        for iot in range(self.n_iot):
            self.task_iot.append(np.zeros(4))

        # edge处理中的iot
        self.task_edge = -1 * np.ones(self.n_edge, dtype=int)
        #edge队列长度
        self.queue_length_edge = np.zeros(self.n_edge, dtype=int)
        self.current_state = np.hstack((self.queue_length_edge))        
        self.task_edge_next = self.task_edge 
        for edge in range(self.n_edge):
            self.queue_length_edge[edge] = self.Queue_edge_wait[edge].qsize()

        self.current_duration = 0

        #初始化动作
        for iot in range(self.n_iot):

            iot_state = np.hstack((self.queue_length_edge))
            action = self.auto_action(iot_state)
            self.action_store[iot] = action
             
            self.task_iot[iot][1] = iot
            self.task_iot[iot][2] = action
            self.task_iot[iot][3] = self.current_time

            if action > 0: #分配到edge
                edge = action - 1
                process_duration, expected_time = self.edge_process(self.n_size, self.comp_cap_edge[edge], self.comp_density)
                if self.task_edge[edge] == -1:              
                    self.task_edge[edge] = iot
                else:
                    self.Queue_edge_wait[edge].put(iot)
                    process_duration *= self.queue_limit
                    self.queue_indicator[iot] = 1
                self.queue_length_edge[edge] += 1

            else: #分配到iot
                process_duration, expected_time = self.iot_process(self.n_size, self.comp_cap_iot, self.comp_density)

            self.task_iot[iot][0] = process_duration


        #task_iot 排序
        self.task_iot = sorted(self.task_iot, key=lambda x:x[0])


        #state store initial        
        self.state_store_now = list()
        for iot in range(self.n_iot):
            current_state = np.hstack((self.queue_length_edge))
            self.state_store_now.append(current_state)

        self.state_store_last = self.state_store_now.copy()

        self.wait_state_store_now = list()
        for iot in range(self.n_iot):
            wait_state = [self.current_duration]
            self.wait_state_store_now.append(wait_state)

        self.wait_state_store_last = self.wait_state_store_now.copy()

        self.wait_mode = np.zeros(self.n_iot, dtype=int)

    #allocate step
    def render(self):
        self.task_iot = sorted(self.task_iot, key=lambda x:x[0])
        self.current_iot = round(self.task_iot[0][1])

        #current_time update
        time_passed = self.task_iot[0][0]
        self.current_time = self.current_time + time_passed

        #iot remain update
        for iot in range(self.n_iot):
            iot_index = round(self.task_iot[iot][1])
            if self.queue_indicator[iot_index] == 0:
                self.task_iot[iot][0] -= time_passed

        #如果当前任务不在等待模式
        if self.wait_mode[self.current_iot] == 0:

            #如果该任务在edge上运行
            if self.task_iot[0][2] > 0:
                current_edge = round(self.task_iot[0][2]) - 1
                if self.Queue_edge_wait[current_edge].empty():
                    self.task_edge[current_edge] = -1
                else:
                    #对应edge等待序列释放一个至运行状态
                    task_iot = self.Queue_edge_wait[current_edge].get()
                    for index in range(self.n_iot):
                        if self.task_iot[index][1] == task_iot:  
                            self.task_iot[index][0] /= self.queue_limit
                            iot_index = round(self.task_iot[index][1])
                            self.queue_indicator[iot_index] = 0
                    self.task_edge[current_edge] = task_iot
                self.queue_length_edge[current_edge] -= 1
        

            # aoi update

            self.current_duration = self.current_time - self.task_iot[0][3]
            self.last_duration = self.duration_store[self.current_iot,1]
            self.duration_store[self.current_iot, 0] = self.last_duration
            self.duration_store[self.current_iot, 1] = self.current_duration
            if self.last_duration + self.current_duration != 0:
                self.wait = self.wait_time[self.current_iot]
                self.aoi = 0.5 * ((self.last_duration + self.wait + self.current_duration) ** 2
                                     - self.current_duration ** 2)/(self.last_duration + self.wait + self.current_duration)
                self.aoi_sum[self.current_iot] += (self.last_duration + self.wait + self.current_duration) ** 2 - self.current_duration ** 2
                self.aoi_average[self.current_iot] = 0.5 * self.aoi_sum[self.current_iot] / self.current_time
                self.old_aoi = self.new_aoi
                self.new_aoi = abs(self.aoi_sum[self.current_iot] - self.gamma * self.current_time)
                # self.ddpg_aoi = abs(self.aoi_sum[self.current_iot] - self.ddpg_gamma * self.current_time)

            # state_store update

            self.current_state = np.hstack((self.queue_length_edge))
            self.state_store_last[self.current_iot] = self.state_store_now[self.current_iot]
            self.state_store_now[self.current_iot] = self.current_state

            
            
            if self.STATE_STILL:
                self.wait_state = [1]
            else:
                self.wait_state = [self.current_duration]
            #self.wait_state = np.hstack([min(self.queue_length_edge)])
            self.wait_state_store_last[self.current_iot] = self.wait_state_store_now[self.current_iot]
            self.wait_state_store_now[self.current_iot] = self.wait_state


            # 训练元组
            self.observation = self.state_store_last[self.current_iot]
            self.action = self.action_store[self.current_iot]
            if self.FRACTION:
                self.reward = - self.new_aoi / 1000
            else:
                self.reward = - self.aoi / 100
            self.observation_next = self.state_store_now[self.current_iot]

            self.wait_observation = self.wait_state_store_last[self.current_iot]
            self.wait_action = self.wait_time[self.current_iot]
            if self.FRACTION:
                self.wait_reward = - self.new_aoi / 10000
            else:
                self.wait_reward = - self.aoi / 100
            self.wait_observation_next = self.wait_state_store_now[self.current_iot]       

        return self.current_iot, self.current_state, self.wait_state

    def execute_offload(self, action, process_duration):

        self.wait_mode[self.current_iot] = 0

        #action_store update
        self.action_store[self.current_iot] = action

        #task main list update       
        self.task_iot[0][2] = action
        self.task_iot[0][3] = self.current_time        

        # 计算相应的持续时间
        if action == 0:
            self.task_iot[0][0] = process_duration
        else:
            current_edge = action - 1
            self.task_iot[0][0] = process_duration
            if self.task_edge[current_edge] == -1:
                self.task_edge[current_edge] = self.current_iot
                self.task_iot[0][0] = process_duration
            else:
                self.Queue_edge_wait[current_edge].put(self.current_iot)
                self.task_iot[0][0] = self.queue_limit * process_duration
                iot_index = round(self.task_iot[0][1])
                self.queue_indicator[iot_index] = 1
            self.queue_length_edge[current_edge] += 1

        self.delay = self.current_duration
        return None

        


    # wait time step
    def execute_wait(self, wait_action):

        self.wait_mode[self.current_iot] = 1
        self.task_iot[0][0] = wait_action
        self.wait_time[self.current_iot] = wait_action
        return None
