import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from scheduling import *
import os
#from RL_brain import ReplayBuffer
#from RL_brain import SAC
import wandb
from scheduling import PROCESS_Task
from environment import Environment2

GLABLEDATA ='data_.txt'
def readtime(i):
    time = []
    with open(GLABLEDATA, 'r', encoding="utf-8") as f:
        for line in f:
            info = list(line.strip(' ').split())
            # task.append(Task(float(info[4]),float(info[10]) ))
            # print(info[0])
            # index.append(info[0])
            time.append(info[12])
    return time[i]

# 阶段三
def readfile2(i):
    """
    Read the input job file
    All task are initialized to ready status
    """
    num_task = 0
    # self.job = []
    index = []
    data_rate = []
    atmospheric_temperature = []
    with open(GLABLEDATA, 'r', encoding="utf-8") as f:
        for line in f:
            info = list(line.strip(' ').split())
            # task.append(Task(float(info[4]),float(info[10]) ))
            #print(info[0])
            #index.append(info[0])
            data_rate.append(float(info[4]))
            atmospheric_temperature.append(float(info[10]))

            # task.append(info[0],info[4],info[10])
            # if len(task) != 0:
            # self.job.append(task)
            # task = []
        # print("length job:", len(self.job))
    # print(task)
    #print("data_rate的长度：",len(data_rate))
    return  data_rate[i], atmospheric_temperature[i]


def progress_bar_runtime(finish_tasks_number, tasks_number, complete_time, total_steps):
    """
    进度条

    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :param complete_time: float, 已完成的任务所消耗的总时间
    :return:
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    finished_label = "▓" * (percentage // 2)
    unfinished_label = "-" * (100 - percentage)
    arrow = "->"
    if not finished_label or not unfinished_label:
        arrow = ""
    print("\r当前第{}轮训练进度: {}% [{}{}{}] {:.2f}s".format((total_steps+1),percentage, finished_label, arrow, unfinished_label, complete_time),
          end="")
'''
加入三阶段的评估过程
'''



def main(args1, args2, number, seed):
    #wandb.init("SAC")
    # se_small = [100, 200, 300, 400, 500]
    se_large = [100]
    se = se_large[0]
    data_list = [ "data_","data_10000", "data_1500", "data_2000"]#需要加载的文件
    for se in se_large:
        dataname = 'data_500-'
        file_name = "PPO_servers_" + str(se) + dataname
        #创建环境
        p1 = environment('large', GLABLEDATA, 1000, se)
        p1_evaluate = environment('large', 'data_2000.txt', 1000, se)
        #阶段三环境
        data_rate, atmospheric_temperature=readfile2(0)


        env = Environment2( optimal_temperature=(18.0, 24.0),atmospheric_temperature=atmospheric_temperature,initial_number_users=20,
                           initial_rate_data=data_rate)#阶段三创建的环境叫做env
        '''
        env_evaluate = Environment2(optimal_temperature=(18.0, 24.0), initial_month=0, initial_number_users=20,
                           initial_rate_data=30)
        '''

        # 创建对象
        p1.setFarm()
        #STEFARM的结果就是生成了一个观察值，是一个列表
        # 随机为每个服务器场设置服务器，每个场至少有一个服务器至多2*m/n-1个服务器，初始化每个服务器和服务器场的能耗使用
        p1_evaluate.setFarm()
        LENGTH = len(p1.custom_reshape(p1.remainFarm))  # 计算长度
        N_STATES = LENGTH
        N_ACTIONS1 = p1.farmNum
        N_ACTIONS2 = int(p1.severNum / p1.farmNum)#这里是为了得到输入维度和输出维度
        #stage1
        #自定义参数
        num_epochs = 100  # 训练回合数
        capacity = 500  # 经验池容量
        min_size = 200  # 经验池训练容量
        batch_size = 64
        n_hiddens = 64
        actor_lr = 1e-3  # 策略网络学习率
        critic_lr = 1e-2  # 价值网络学习率
        alpha_lr = 1e-2  # 课训练变量的学习率
        target_entropy = -1
        tau = 0.005  # 软更新参数
        gamma = 0.9  # 折扣因子
        device = torch.device('cuda')
        #定义了阶段三的动作空间，但是没有定义状态空间,状态空间在策略中被定义了
        epsilon = .3
        number_actions = 5
        direction_boundary = (number_actions - 1) / 2
        # number_epochs = 100
        # max_memory = 3000
        #batch_size = 512
        temperature_step = 1.5

        args1.state_dim = N_STATES
        args1.action_dim = N_ACTIONS1
        args1.max_episode_steps = 200  # Maximum number of steps per episode
        #replay_buffer1 = ReplayBuffer(500) #capacity = 500
        replay_buffer1 = ReplayBuffer(args1)
        agent1 = PPO_discrete(args1)
        #agent1 = SAC(n_states=N_STATES,n_actions=N_ACTIONS1,critic_lr=critic_lr,n_hiddens=64,actor_lr=actor_lr,alpha_lr=alpha_lr,target_entropy=target_entropy,tau=tau,gamma=gamma,device=device)
        #stage2
        args2.state_dim = N_STATES
        args2.action_dim = N_ACTIONS2
        args2.max_episode_steps = 200  # Maximum number of steps per episode
        replay_buffer2 = ReplayBuffer(args2)
        agent2 = PPO_discrete(args2)
        #replay_buffer2 = ReplayBuffer(500)
        #agent2 =SAC(n_states=N_STATES,n_actions=N_ACTIONS2,critic_lr=critic_lr,n_hiddens=64,actor_lr=actor_lr,alpha_lr=alpha_lr,target_entropy=target_entropy,tau=tau,gamma=gamma,device=device)
        #stage3
        args3.state_dim= 3
        args3.action_dim = number_actions
        #agent3 = SAC(n_states=3,n_actions=number_actions,critic_lr=critic_lr,n_hiddens=64,actor_lr=actor_lr,alpha_lr=alpha_lr,target_entropy=target_entropy,tau=tau,gamma=gamma,device=device)
        #这一句相当于源代码中的BRAIN
        #replay_buffer3 = ReplayBuffer(500)#这一句就相当于DQN
        agent3 = PPO_discrete(args3)
        # 这一句相当于源代码中的BRAIN
        replay_buffer3 = ReplayBuffer(args3)  # 这一句就相当于DQN
        #print("env={}".format("p1"))
        #print("state1_dim={} stage2_dim={}".format(args1.state_dim, args2.state_dim))
        #print("action1_dim={} action2_dim={}".format(args1.action_dim, args2.action_dim))
        #print("max_episode_steps={}".format(args2.max_episode_steps))

        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards1 = []  # Record the rewards during the evaluating
        evaluate_rewards2 = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training
        # Build a tensorboard
        #writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format("p3", number, seed))
        writer = SummaryWriter(log_dir='runs/env_{}_number_{}_seed_{}'.format("p1",number,seed))

        state_norm1 = Normalization(shape=args1.state_dim)  # Trick 2:state normalization
        state_norm2 = Normalization(shape=args2.state_dim)  # Trick 2:state normalization
        if args1.use_reward_norm:  # Trick 3:reward normalization
            reward_norm1 = Normalization(shape=1)
        elif args1.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling1 = RewardScaling(shape=1, gamma=args1.gamma)

        if args2.use_reward_norm:  # Trick 3:reward normalization
            reward_norm2 = Normalization(shape=1)
        elif args2.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling2 = RewardScaling(shape=1, gamma=args2.gamma)

        energy = []
        ep_time = []
        rej_rate = []
        total_rewards1, avg_rewards1, epsilon_history1 = [], [], []
        total_rewards, avg_rewards, epsilon_history = [], [], []
        total_rewards3,avg_rewards3,epsilon_history3=[],[],[]
        reward1_record,reward2_record,reward3_record=[], [], []
        #三阶段的基本的变量设置


        env.train = train
        #model = brain.model
        early_stopping = True
        patience = 10
        best_total_reward = -np.inf
        patience_count = 0

        while total_steps < args2.max_train_steps:#训练次数




            start = time.perf_counter()#统计时间
            '''
            在第一个回合的时候，什么都是空的，问题是第二个回合开始的时候。
            加入reset仅仅只负责重置状态，负责将状态送回到远点
            另外建一个方法，负责将资源列表等清空呢？
            '''
            p1.reset()  # 初始化状态，每一步插曲开始的时候，都需要重新启动
            p1.setFarm()  # 随机为每个服务器场设置服务器，每个场至少有一个服务器至多2*m/n-1个服务器，初始化每个服务器和服务器场的能耗使用
            #任务数量怎么来的
            input_stage1 = input_stage2 = p1.custom_reshape(p1.remainFarm).astype(np.float32)

            s1_current_state = input_stage1.astype(np.float32)
            s2_current_state = input_stage2.astype(np.float32)
            episode_steps = 0
            r1 = 0
            r2 = 0
            r3=0
            price1=0
            price2=0
            acc = 0
            episode_reward1 = 0
            episode_reward2 = 0
            done = False
            task_length = len(p1.task)

            # 控温阶段的代码设置
            total_reward = 0
            loss = 0.
            new_month = np.random.randint(0, 12)  # 改掉之后不需要月份
            _,at = readfile2(0)
            env.reset(atmospheric_temperature=at)  # 环境的重置
            game_over = False
            current_state, _, _ = env.observe()#S3阶段的当前状态
            #timestep = 0
            cpu_liyonglv=[]
            ram_liyonglv=[]
            #print("p1.task的长度：",len(p1.task))
            while len(p1.task) != 0:
                # Take an action using probabilities from policy network output.

                for t in p1.task:
                    a=t.index
                    #print(a)
                    p1.time_reset()
                    if t.status == -1:  # rejected
                        p1.task.remove(t)
                    elif t.status == 1:  # ready


                        f_action, f_logprob = agent1.choose_action(s1_current_state)#第一阶段动作
                        #print("f_action",f_action)
                        s_action,s_logprob = agent2.choose_action(s2_current_state)  # 选择服务器
                        #print("s_action", s_action)
                        vm = random.randint(0, p1.VMNum - 1)  # 随机选择这个服务器上的某一个虚拟机
                        #print("cccccccccccc",p1.remainFarm[f_action][s_action][vm])
                        '''
                        选择任务之后，按道理来说服务器状态应该要发生改变。在执行完任务之后，按道理要返还资源
                        '''
                        #p1.releaseByTime(f_action, s_action, vm)  # release by time
                        rej = p1.checkRej(f_action, s_action, vm, t)

                        if rej == -1:  # rejected due to ddl
                            t.status = -1
                            continue
                        elif rej == 0:
                            t.endtime = time.time() + t.runtime
                            #状态更新以及奖励值的更新
                            #print("打印s1", s1_current_state)
                            even_index_sum_cpu = sum(s1_current_state[i] for i in range(0, len(s1_current_state), 2))
                            #print("打印s1sdcvdsvsdv",  len(s1_current_state))
                            odd_index_sum_RAM = sum(s1_current_state[i] for i in range(1, len(s1_current_state), 2))
                            cpu_liyonglv.append(1-even_index_sum_cpu/100)

                            ram_liyonglv.append(1-odd_index_sum_RAM/100)
                            t.CPU=round(t.CPU,3)
                            t.RAM=round(t.RAM,3)
                            s1_next_state = s2_next_state = p1.UpdateServerState(f_action, s_action, vm,
                                                                                       t).astype(np.float32)
                            #更新到前两个阶段的下一个状态
                            data_time = readtime(int(t.index.strip('\ufeff'))-1)
                            #print("aaaaaaa",data_time)
                            data_rate, _ = readfile2(int(t.index.strip('\ufeff')) - 1)
                            reward_stage1, price_stage1= p1.rewardFcn1(t.CPU , t.RAM,data_time,f_action,data_rate,s_action,vm)#

                            if reward_stage1 >1000:
                                reward_stage1=0
                            #print("显示一阶段奖励值",reward_stage1)
                            #reward_stage1 = p1.rewardFcn1(t.RAM, t.BANDWIDTH)
                            #print("显示每分配一个任务的一阶段奖励值", reward_stage1)

                            reward_stage2,price_stage2 = p1.rewardFcn2(f_action, s_action,data_rate,data_time)
                            p1.releaseByTime(f_action, s_action, vm,t)  # release by time


                            #确定refarm的值和表是一样的才可以
                            #print("打印s1nnnn", s1_current_state)
                            #print("打印剩余资源表", t.remainFarm)
                            #print("显示每分配一个任务的二阶段奖励值", reward_stage2)



                            # PLAYING THE NEXT ACTION BY EXPLORATION 应该只要ELSE的内容就可以

                                #q_values = agent3.predict(current_state)
                                #action3 = np.argmax(q_values[0])  # 选择最可能的动作，这里我使用了SAC
                            action3, f_logprob3 = agent3.choose_action(current_state)
                            if (action3 - direction_boundary < 0):
                                direction = -1
                            else:
                                direction = 1
                            energy_ai = abs(action3 - direction_boundary) * temperature_step#计算出这个值，这个值后面用来计算奖励值

                            # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
                            #奖励值的计算放在了函数环境状态更新里面

                            data_rate,atmospheric_temperature=readfile2(int(t.index.strip('\ufeff'))-1)

                            #data_rate=readfile2(t.index)
                            next_state, reward, game_over = env.update_env(direction, energy_ai,
                                                                          atmospheric_temperature,
                                                                           data_rate)


                            # 这部分代码负责将新的状态转移存储到记忆中，并使用这些记忆来训练模型
                            #dqn.remember([current_state, action, reward, next_state], game_over)

                            # GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
                            #inputs, targets = dqn.get_batch(model, batch_size=batch_size)
                            # COMPUTING THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
                            #loss += model.train_on_batch(inputs, targets)
                            #timestep += 1
                            current_state = next_state
                            #



                            p1.VMtask[f_action][s_action][vm].append(t)
                            t.status = 2
                            p1.task.remove(t)
                            '''
                            按道理，在移除任务之后，用的资源需要返回
                            '''
                            # episode_frames.append(set_image_context(p1.true, s2_current_state, episode_reward, s_action, 99))
                            acc += 1
                            #上面已经选择好了，下面再进行计算奖励值

                            #state2
                            #print("reward_stage1*****************",reward_stage1)
                            #print("reward_stage2*****************",reward_stage2)
                            #print("r1的类型",type(r1))
                            #print("r2的类型",type(r2))
                            r1 += reward_stage1#执行好一次所有任务所输出的一阶段奖励值的总和
                            r2 += reward_stage2#执行一次所有任务输出的能耗的总和  就是执行一次任务列表后服务器产生的能耗和增加的冷却能耗。这里没有减去空闲的能耗
                            price1 +=price_stage1
                            price2 +=price_stage2

                            #r1,r2是总回报
                            total_reward += reward#三阶段的总回报

                            reward_3=reward
                            #print("sdavsdvadfvdfvfadvfv",reward_3)
                            episode_reward1 += reward_stage1
                            episode_reward2 += reward_stage2
                            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                            # dw means dead or win,there is no next state s';
                            # but when reaching the max_episode_steps,there is a next state s' actually.
                            if len(p1.task) == 0:
                                done = True
                            # When the number of transitions in buffer reaches batch_size,then update
                            if done and episode_steps != args2.max_episode_steps:
                                dw = True
                            else:
                                dw = False
                            #done这个变量表示整个任务调度过程是否结束。
                            #任务调度结束 (done 为 True)


                            #replay_buffer1.store(s1_current_state, f_action, f_logprob, reward_stage1, s1_next_state, dw, done)
                            #replay_buffer2.store(s2_current_state, s_action, s_logprob, reward_stage2, s2_next_state, dw, done)
                            replay_buffer1.store(s1_current_state, f_action, f_logprob, reward_stage1, s1_next_state,
                                                 dw, done)
                            replay_buffer2.store(s2_current_state, s_action, s_logprob, reward_stage2, s2_next_state,
                                                 dw, done)
                            replay_buffer3.store(current_state, action3, f_logprob3, reward,
                                                 next_state, dw, done)
                            s1_current_state = s1_next_state
                            s2_current_state = s2_next_state
                            current_state = next_state
                            #设置进度条
                            duration = time.perf_counter() - start
                            task_length_now = task_length - len(p1.task)
                            progress_bar_runtime(task_length_now, task_length, duration, total_steps)  # 输出调度进度
                            # 更新模型
                            if replay_buffer1.count == args1.batch_size:
                                agent1.update(replay_buffer1, total_steps)
                                replay_buffer1.count = 0
                                # s, a, r, ns, d = replay_buffer1.sample(batch_size)
                                # transition_dict = {'states': s,
                                # 'actions': a,
                                # 'rewards': r,
                                # 'next_states': ns,
                                # 'dones': d}
                                # 模型训练
                                # agent1.update(transition_dict)

                            if replay_buffer2.count == args2.batch_size:
                                agent2.update(replay_buffer2, total_steps)
                                replay_buffer2.count = 0
                                '''
                                s2, a2, r22, ns2, d2 = replay_buffer2.sample(batch_size)
                                transition_dict2 = {'states': s2,
                                                   'actions': a2,
                                                   'rewards': r22,
                                                   'next_states': ns2,
                                                   'dones': d2}
                                '''
                                # 模型训练
                                # agent2.update(transition_dict2)
                            if replay_buffer3.count == args3.batch_size:
                                agent3.update(replay_buffer3, total_steps)
                                replay_buffer3.count = 0

                                '''
                                s3,a3,r3,ns3,d3 = replay_buffer3.sample(batch_size)
                                s3 = np.squeeze(s3,axis=1)
                                transition_dict3 = {
                                            'states': s3,
                                                   'actions': a3,
                                                   'rewards': r3,
                                                   'next_states': ns3,
                                                   'dones': d3

                                }
                                #print(transition_dict2['states'].shape)
                                #print("###################")
                                #print(transition_dict3['states'].shape)
                                # 模型训练
                                agent3.update(transition_dict3)
                                '''
            print("cpu的利用率",cpu_liyonglv)
            print("ram的利用率", ram_liyonglv)

            # 保存模型
            if (total_steps == args2.max_train_steps - 1):
                #换成为se或者data
                save_path1 = os.path.join('models1', str(se) + data_list[0] + os.path.basename(args1.model_path))

                save_path2 = os.path.join('models2', str(se) + data_list[0] + os.path.basename(args1.model_path))
                save_path3 = os.path.join('models3', str(se) + data_list[0] + os.path.basename(args1.model_path))
                agent1.save_model(save_path1)
                agent2.save_model(save_path2)
                agent3.save_model(save_path3)
            total_steps += 1
            # 运行时间
            end = time.perf_counter() - start
            ep_time.append(end)
            # 拒绝率
            rej_rate.append(p1.rej / p1.allscheduling)
            #stage1_reward
            total_rewards1.append(r1)#r1是一次循环的情况，这里表示所有循环

            avg_reward1 = np.mean(total_rewards1[-100:])#计算 total_rewards1 列表最后 100 个元素的平均值，就是最后一百次循环的能耗每一次循环的平均值
            avg_rewards1.append(avg_reward1)
            # stage2 reward
            total_rewards.append(r2)

            avg_reward = np.mean(total_rewards[-100:])
            avg_rewards.append(avg_reward)
            # epsilon_history.append(agent.epsilon)
            energy.append(p1.totalcost)#
            #阶段3
            total_rewards3.append(reward_3)

            print("total_rewards3",total_rewards3)
            #avg_reward3=np.mean(total_rewards3[-100:])
            avg_reward3 = np.mean(total_rewards3)

            avg_rewards3.append(avg_reward3)

            print("\n", 'EP:{} reward1:{} Avg_reward1:{} reward2:{} Avg_reward2:{} Energy Consumption:{} rejrate:{} reward3:{} Avg_reward3:{}'.
                  format(total_steps + 1, r1, avg_reward1, r2, avg_reward, p1.totalcost, p1.rej ,total_reward,avg_reward3))
            reward1_record.append(r1)
            print("\n","奖励值1记录",reward1_record)
            reward2_record.append(r2)
            print("\n", "奖励值2记录", reward2_record)
            reward3_record.append(total_reward)
            print("\n", "奖励值3记录", reward3_record)
            #print("\n",'p1.paclist',p1.paclist)
            print("\n","总冷却功率",p1.totalcoolsum)
            print("\n","总服务器功率",p1.totalwebsum)
            print("\n", "总冷却功率电价",price1)
            print("\n", "总服务器功率电价", price2)
            '''
            wandb.log({
                "reward1":r1,
                "avg_reward1":avg_reward1,
                "avg_reward2":avg_reward,
                "reward2":r2,
                "Energy Consumption":p1.totalcost,
                "rejrate":p1.allscheduling
            })
            '''
            '''
            if total_steps % args2.evaluate_freq == 0:
                evaluate_num += 1
                #stage1
                evaluate_reward1, evaluate_reward2 = evaluate_policy(args1, p1_evaluate, agent1, state_norm1,
                                                                     args2, agent2, state_norm2)
                evaluate_rewards1.append(evaluate_reward1)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward1))
                writer.add_scalar('step_rewards_{}'.format("P1"), evaluate_rewards1[-1],
                                  global_step=total_steps)
                # Save the rewards
                if evaluate_num % args2.save_freq == 0:
                    np.save(
                        './data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format("p2", number,
                                                                                        seed),
                        np.array(evaluate_rewards1))
                #stage2
                evaluate_rewards2.append(evaluate_reward2)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward2))
                writer.add_scalar('step_rewards_{}'.format("P1"), evaluate_rewards2[-1],
                                  global_step=total_steps)
                # Save the rewards
                if evaluate_num % args2.save_freq == 0:
                    np.save(
                        './data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format("p2", number,
                                                                                        seed),
                        np.array(evaluate_rewards2))
        '''

        # energy
        file = open(file_name + "energy.txt", 'a')
        mid = str(energy).replace('[', '').replace(']', '')
        # # 删除单引号并用字符空格代替逗号
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()
        # time
        file = open(file_name + "runtime.txt", 'a')
        mid = str(ep_time).replace('[', '').replace(']', '')
        # # 删除单引号并用字符空格代替逗号
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()
        # reward
        file = open(file_name + "reward1.txt", 'a')
        mid = str(total_rewards1).replace('[', '').replace(']', '')
        # # 删除单引号并用字符空格代替逗号
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()
        # loss26
        file = open(file_name + "reward2.txt", 'a')
        mid = str(total_rewards).replace('[', '').replace(']', '')
        # # 删除单引号并用字符空格代替逗号
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()

        # rejrate
        file = open(file_name + "rejrate.txt", 'a')
        mid = str(rej_rate).replace('[', '').replace(']', '')
        # # 删除单引号并用字符空格代替逗号
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()



if __name__ == '__main__':
    #stage1
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument('--model_path', type=str, default='./my_model.pth')
    parser.add_argument("--max_train_steps", type=int, default=int(1), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=10000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=8e-2")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    #stage2
    parser2 = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser2.add_argument('--model_path', type=str, default='./my_model.pth')
    parser2.add_argument("--max_train_steps", type=int, default=int(250), help=" Maximum number of training steps")
    parser2.add_argument("--evaluate_freq", type=float, default=2.5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser2.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser2.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser2.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser2.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser2.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser2.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser2.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser2.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser2.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser2.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser2.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser2.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser2.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser2.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser2.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser2.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser2.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser2.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser2.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser2.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    # stage3
    parser3 = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser3.add_argument('--model_path', type=str, default='./my_model.pth')
    parser3.add_argument("--max_train_steps", type=int, default=int(100), help=" Maximum number of training steps")
    parser3.add_argument("--evaluate_freq", type=float, default=2.5e2,
                         help="Evaluate the policy every 'evaluate_freq' steps")
    parser3.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser3.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser3.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser3.add_argument("--hidden_width", type=int, default=64,
                         help="The number of neurons in hidden layers of the neural network")
    parser3.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser3.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser3.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser3.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser3.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser3.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser3.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser3.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser3.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser3.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser3.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser3.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser3.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser3.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser3.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser3.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args1 = parser.parse_args()
    args2 = parser2.parse_args()
    args3 = parser3.parse_args()
    train = 0
    test = 1
    main(args1, args2, number=1, seed=0)

