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

def main(args1, args2, number, seed):
    #wandb.init("SAC")
    # se_small = [100, 200, 300, 400, 500]
    se_large = [100]
    se = se_large[0]
    data_list = [ "data_","data_10000", "data_1500", "data_2000"]#
    for se in se_large:
        dataname = 'data_500-'
        file_name = "PPO_servers_" + str(se) + dataname
   
        p1 = environment('large', GLABLEDATA, 1000, se)
        p1_evaluate = environment('large', 'data_2000.txt', 1000, se)
    
        data_rate, atmospheric_temperature=readfile2(0)


        env = Environment2( optimal_temperature=(18.0, 24.0),atmospheric_temperature=atmospheric_temperature,initial_number_users=20,
                           initial_rate_data=data_rate)
        '''
        env_evaluate = Environment2(optimal_temperature=(18.0, 24.0), initial_month=0, initial_number_users=20,
                           initial_rate_data=30)
        '''

    
        p1.setFarm()
        p1_evaluate.setFarm()
        LENGTH = len(p1.custom_reshape(p1.remainFarm)) 
        N_STATES = LENGTH
        N_ACTIONS1 = p1.farmNum
        N_ACTIONS2 = int(p1.severNum / p1.farmNum)
        #stage1
        num_epochs = 100  
        capacity = 500  
        min_size = 200 
        batch_size = 64
        n_hiddens = 64
        actor_lr = 1e-3  
        critic_lr = 1e-2 
        alpha_lr = 1e-2  
        target_entropy = -1
        tau = 0.005  
        gamma = 0.9  
        device = torch.device('cuda')
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
        #replay_buffer3 = ReplayBuffer(500)
        agent3 = PPO_discrete(args3)
        # 这一句相当于源代码中的BRAIN
        replay_buffer3 = ReplayBuffer(args3) 
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
  


        env.train = train
        #model = brain.model
        early_stopping = True
        patience = 10
        best_total_reward = -np.inf
        patience_count = 0

        while total_steps < args2.max_train_steps



            start = time.perf_counter()
            p1.reset()  
            p1.setFarm()  
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

            total_reward = 0
            loss = 0.
            new_month = np.random.randint(0, 12) 
            _,at = readfile2(0)
            env.reset(atmospheric_temperature=at)  
            game_over = False
            current_state, _, _ = env.observe()
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


                        f_action, f_logprob = agent1.choose_action(s1_current_state)
                        #print("f_action",f_action)
                        s_action,s_logprob = agent2.choose_action(s2_current_state)  
                        #print("s_action", s_action)
                        vm = random.randint(0, p1.VMNum - 1) 
                        #print("cccccccccccc",p1.remainFarm[f_action][s_action][vm])
                
                        #p1.releaseByTime(f_action, s_action, vm)  # release by time
                        rej = p1.checkRej(f_action, s_action, vm, t)

                        if rej == -1:  # rejected due to ddl
                            t.status = -1
                            continue
                        elif rej == 0:
                            t.endtime = time.time() + t.runtime
                          
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
                           
                            data_time = readtime(int(t.index.strip('\ufeff'))-1)
                            #print("aaaaaaa",data_time)
                            data_rate, _ = readfile2(int(t.index.strip('\ufeff')) - 1)
                            reward_stage1, price_stage1= p1.rewardFcn1(t.CPU , t.RAM,data_time,f_action,data_rate,s_action,vm)#

                            if reward_stage1 >1000:
                                reward_stage1=0
                           
                    

                            reward_stage2,price_stage2 = p1.rewardFcn2(f_action, s_action,data_rate,data_time)
                            p1.releaseByTime(f_action, s_action, vm,t)  # release by time


                           

                            action3, f_logprob3 = agent3.choose_action(current_state)
                            if (action3 - direction_boundary < 0):
                                direction = -1
                            else:
                                direction = 1
                            energy_ai = abs(action3 - direction_boundary) * temperature_step#计算出这个值，这个值后面用来计算奖励值

                         
                            data_rate,atmospheric_temperature=readfile2(int(t.index.strip('\ufeff'))-1)

                            #data_rate=readfile2(t.index)
                            next_state, reward, game_over = env.update_env(direction, energy_ai,
                                                                          atmospheric_temperature,
                                                                           data_rate)


                            


                            p1.VMtask[f_action][s_action][vm].append(t)
                            t.status = 2
                            p1.task.remove(t)
                            '''
                            按道理，在移除任务之后，用的资源需要返回
                            '''
                            
                            r1 += reward_stage1
                            r2 += reward_stage2
                            price1 +=price_stage1
                            price2 +=price_stage2

                      
                            total_reward += reward#三阶段的总回报

                            reward_3=reward
                            
                            episode_reward1 += reward_stage1
                            episode_reward2 += reward_stage2
                           
                            if len(p1.task) == 0:
                                done = True
                            # When the number of transitions in buffer reaches batch_size,then update
                            if done and episode_steps != args2.max_episode_steps:
                                dw = True
                            else:
                                dw = False
                          


                           
                            replay_buffer1.store(s1_current_state, f_action, f_logprob, reward_stage1, s1_next_state,
                                                 dw, done)
                            replay_buffer2.store(s2_current_state, s_action, s_logprob, reward_stage2, s2_next_state,
                                                 dw, done)
                            replay_buffer3.store(current_state, action3, f_logprob3, reward,
                                                 next_state, dw, done)
                            s1_current_state = s1_next_state
                            s2_current_state = s2_next_state
                            current_state = next_state
                           
                            duration = time.perf_counter() - start
                            task_length_now = task_length - len(p1.task)
                            progress_bar_runtime(task_length_now, task_length, duration, total_steps)  # 输出调度进度
                           
                            if replay_buffer1.count == args1.batch_size:
                                agent1.update(replay_buffer1, total_steps)
                                replay_buffer1.count = 0
                               
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
                                
                                agent3.update(transition_dict3)
                                '''
            print("cpu的利用率",cpu_liyonglv)
            print("ram的利用率", ram_liyonglv)

            
            if (total_steps == args2.max_train_steps - 1):
                #换成为se或者data
                save_path1 = os.path.join('models1', str(se) + data_list[0] + os.path.basename(args1.model_path))

                save_path2 = os.path.join('models2', str(se) + data_list[0] + os.path.basename(args1.model_path))
                save_path3 = os.path.join('models3', str(se) + data_list[0] + os.path.basename(args1.model_path))
                agent1.save_model(save_path1)
                agent2.save_model(save_path2)
                agent3.save_model(save_path3)
            total_steps += 1
 
            end = time.perf_counter() - start
            ep_time.append(end)
 
            rej_rate.append(p1.rej / p1.allscheduling)
            #stage1_reward
            total_rewards1.append(r1)

            avg_reward1 = np.mean(total_rewards1[-100:])#计算 total_rewards1 列表最后 100 个元素的平均值，就是最后一百次循环的能耗每一次循环的平均值
            avg_rewards1.append(avg_reward1)

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
           
        # energy
        file = open(file_name + "energy.txt", 'a')
        mid = str(energy).replace('[', '').replace(']', '')
       
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()
        # time
        file = open(file_name + "runtime.txt", 'a')
        mid = str(ep_time).replace('[', '').replace(']', '')
       
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()
        # reward
        file = open(file_name + "reward1.txt", 'a')
        mid = str(total_rewards1).replace('[', '').replace(']', '')
       
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()
        # loss26
        file = open(file_name + "reward2.txt", 'a')
        mid = str(total_rewards).replace('[', '').replace(']', '')
       
        mid = mid.replace("'", '').replace(',', '') + '\n'
        file.write(mid)
        file.close()

        # rejrate
        file = open(file_name + "rejrate.txt", 'a')
        mid = str(rej_rate).replace('[', '').replace(']', '')
     
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

