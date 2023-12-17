from mec_env import Offload
import numpy as np
import argparse
from RL_brain import D3QN

import os


# 设置随机数种子


def train(RL, RL_wait, args):
    reward_average_list = list()
    aoi_average_list = list()
    wait_average_list = list()
    ori_average_list = list()
    gamma_list = list()
    dqn_cost_list = list()
    drop_ratio_list = list()
    duration_list = np.zeros([env.n_iot, env.n_actions])
    duration_average_list = np.zeros([env.n_iot, env.n_actions])
    duration_count_list = np.zeros([env.n_iot, env.n_actions], dtype=int)

    for episode in range(args.num_episode):
        # initial observation
        env.reset(RL)
        step_wait = np.zeros(env.n_iot, dtype=int)
        reward_all = np.zeros(env.n_iot)
        step = np.zeros(env.n_iot, dtype=int)
        step0 = 0
        drop_all = 0
        count_all = 0
        wait_action_list = list()
        ori_action_list = list()
        done = False

        for iot in range(env.n_iot):
            if args.D3QN_NOISE == 1 and episode < 301:
                RL[iot].epsilon = episode / 300
            # if episode > 270:
            #     RL[iot].new_lr = 3e-4 / (episode-270)
            # if episode > 0:
            #     RL_wait[iot].a_lr = 0.0001 / episode
            #     RL_wait[iot].c_lr = 0.001 / episode

        while True:
            if args.iot_step > 0.001:
                if step0 > args.iot_step:
                    break
            elif env.current_time > args.num_time:
                break
            # RL take action and get next observation and reward
            current_iot, current_state, wait_state = env.render()
            if env.wait_mode[current_iot] == 0:
                RL[current_iot].store_transition(
                    env.observation, env.action, env.reward, env.observation_next
                )

                if WAITING:
                    RL_wait[current_iot].perceive(
                        env.wait_observation,
                        env.wait_action,
                        env.wait_reward,
                        env.wait_observation_next,
                        done,
                    )
                    reward_all[env.current_iot] += env.reward

                    if args.DDPG_NOISE == 1:
                        if episode > 300:
                            wait_action = RL_wait[current_iot].action(
                                env.wait_state_store_now[env.current_iot]
                            )
                            ori_action = wait_action
                        else:
                            ori_action, wait_action = RL_wait[current_iot].noise_action(
                                env.wait_state_store_now[env.current_iot]
                            )
                    else:
                        # ori_action, wait_action = RL_wait[current_iot].noise_action(env.wait_state_store_now[env.current_iot] ) * args.action_range
                        wait_action = RL_wait[current_iot].action(
                            env.wait_state_store_now[env.current_iot]
                        )
                        ori_action = wait_action

                    wait_action = (wait_action / 2 + 0.5) * args.action_range
                    ori_action = (ori_action / 2 + 0.5) * args.action_range

                    if wait_action < 0:
                        wait_action = 0
                else:
                    wait_action = args.wait_time
                    ori_action = wait_action
                    reward_all[env.current_iot] += env.reward

                wait_action_list.append(wait_action)
                ori_action_list.append(ori_action)

                env.execute_wait(wait_action)
                step_wait[current_iot] += 1

            else:
                action = RL[current_iot].choose_action(current_state)

                if action == 0:
                    process_duration, expected_time = env.iot_process(
                        env.n_size, env.comp_cap_iot, env.comp_density
                    )
                else:
                    current_edge = action - 1
                    process_duration, expected_time = env.edge_process(
                        env.n_size, env.comp_cap_edge[current_edge], env.comp_density
                    )

                if (
                    drop_ratio_set < 0.001
                    or episode == 0
                    or process_duration
                    < drop_ratio_set * duration_average_list[current_iot][action]
                ):
                    env.execute_offload(action, process_duration)
                    drop_all += 1
                    count_all += 1

                else:  # task drop
                    original_wait = env.wait_time[current_iot]
                    env.execute_wait(process_duration)
                    env.wait_time[current_iot] += original_wait
                    count_all += 1
                duration_list[current_iot][action] += process_duration
                duration_count_list[current_iot][action] += 1

                if env.current_time > args.num_time / 2:
                    duration_average_list[current_iot][action] = (
                        duration_list[current_iot][action]
                        / duration_count_list[current_iot][action]
                    )

            if (step[env.current_iot] > 10) and (
                step[env.current_iot] % args.d3qn_step == 0
            ):
                RL[env.current_iot].learn()

            if (
                WAITING
                and RL_wait[env.current_iot].replay_buffer.count() > 64
                and episode % 30 == 0
            ):
                RL_wait[env.current_iot].train()

            step[env.current_iot] += 1
            step0 += 1

        # print("episdoe------" ,episode,"------")
        aoi_average = 0
        reward_average = 0

        for iot in range(env.n_iot):
            reward_average += reward_all[iot] / step[iot]

        reward_average /= env.n_iot

        aoi_average = np.mean(env.aoi_average)
        reward_average_list.append(reward_average)

        # if episode > 0 and aoi_average > aoi_average_list[-1] + 0.5 or episode % 375 == 0:
        #     duration_path = folderpath+'/duration_'+str(episode)+'.txt'
        #     np.savetxt(duration_path, duration_average_list)

        aoi_average_list.append(aoi_average)
        gamma_list.append(env.gamma)
        drop_ratio = drop_all / count_all
        drop_ratio_list.append(drop_ratio)
        if len(wait_action_list) > 0:
            wait_average = sum(wait_action_list) / len(wait_action_list)
            wait_average_list.append(wait_average)
            ori_average = sum(ori_action_list) / len(ori_action_list)
            ori_average_list.append(ori_average)

        print(episode)
        print(
            "ORI:"
            + str(ori_average)
            + " WAITING:"
            + str(wait_average)
            + " AoI:"
            + str(aoi_average)
            + " gamma:"
            + str(env.gamma)
        )
        print("Time:" + str(env.current_time) + " Step:" + str(step0))

        # gmma更新
        if episode > 0 and episode % 50 == 0:
            aoi50 = np.mean(aoi_average_list[-50:])
            env.gamma += 0.5 * (aoi50 - env.gamma)

        # end of game

        if episode % 500 == 0:
            np.savetxt(filepath1, aoi_average_list)
            np.savetxt(filepath2, reward_average_list)
            np.savetxt(filepath3, gamma_list)
            np.savetxt(filepath4, drop_ratio_list)
            np.savetxt(filepath5, wait_average_list)
            np.savetxt(filepath6, ori_average_list)
    print("game over")


parser = argparse.ArgumentParser(description="MEC-DRL")
parser.add_argument(
    "--model", type=str, default="D3QN", help="choose a model: D3QN, DDPG"
)
parser.add_argument(
    "--comp_iot", type=float, default=2.5, help="Computing capacity of mobile device"
)
parser.add_argument(
    "--comp_edge", type=float, default=41.8, help="Computing capacity of edge device"
)
parser.add_argument(
    "--comp_cap_edge",
    type=float,
    nargs="+",
    default=[3, 8],
    help="Computing capacity of edge device",
)
parser.add_argument(
    "--comp_density",
    type=float,
    default=0.297,
    help="Computing capacity of edge device",
)
parser.add_argument(
    "--num_iot", type=int, default=20, help="The number of mobile devices"
)
parser.add_argument(
    "--num_edge", type=int, default=2, help="The number of edge devices"
)
parser.add_argument("--num_time", type=float, default=200, help="Time per episode")
parser.add_argument("--num_episode", type=int, default=1501, help="number of episode")
parser.add_argument(
    "--drop_coefficient", type=float, default=1.5, help="number of episode"
)
parser.add_argument("--task_size", type=float, default=30, help="Task size (M)")
parser.add_argument("--gamma", type=float, default=5, help="gamma for fractional")
parser.add_argument(
    "--folder", type=str, default="standard", help="The folder name of the process"
)
parser.add_argument(
    "--subfolder", type=str, default="test", help="The sub-folder name of the process"
)
parser.add_argument("--iot_step", type=int, default=0, help="step per iot")
parser.add_argument("--wait_time", type=float, default=0, help="Fixed waiting time")
parser.add_argument(
    "--action_range", type=float, default=3, help="Waiting action range"
)
parser.add_argument(
    "--FRACTION", type=int, default=1, help="Have fractional AoI or not"
)
parser.add_argument("--D3QN_NOISE", type=int, default=1, help="Have D3QN noise or not")
parser.add_argument("--DDPG_NOISE", type=int, default=1, help="Have DDPG noise or not")
parser.add_argument("--cuda", type=str, default="0", help="Using GPU")
parser.add_argument("--STATE_STILL", type=int, default=0, help="STILL STATE?")
parser.add_argument("--d3qn_step", type=int, default=10, help="D3QN Step")
parser.add_argument("--d3qn_batch", type=int, default=32, help="D3QN Batch Size")
parser.add_argument("--d3qn_lr", type=float, default=3e-4, help="D3QN Learning Rate")
parser.add_argument(
    "--actor_lr", type=float, default=3e-4, help="DDPG Actor Learning Rate"
)
parser.add_argument(
    "--critic_lr", type=float, default=3e-3, help="DDPG Critic Learning Rate"
)
parser.add_argument("--ddpg_batch", type=int, default=64, help="DDPG Batch Size")
args = parser.parse_args()

print(args)

folder = args.folder
sub_folder = args.subfolder
filename1 = "aoi_average_list.txt"
filename2 = "reward_average_list.txt"
filename3 = "gamma_list.txt"
filename4 = "drop_ratio_list.txt"
filename5 = "wait_average_list.txt"
filename6 = "ori_average_list.txt"
folderpath = "./storage/" + folder + "/" + sub_folder

if not os.path.exists(folderpath):
    os.makedirs(folderpath)

args_file = folderpath + "/args.txt"
# np.savetxt(args_file, args)
args_dict = args.__dict__
with open(args_file, "w") as f:
    f.writelines("------------------ start ------------------" + "\n")
    for eachArg, value in args_dict.items():
        f.writelines(eachArg + " : " + str(value) + "\n")

    f.writelines("------------------- end -------------------")
filepath1 = folderpath + "/" + filename1
filepath2 = folderpath + "/" + filename2
filepath3 = folderpath + "/" + filename3
filepath4 = folderpath + "/" + filename4
filepath5 = folderpath + "/" + filename5
filepath6 = folderpath + "/" + filename6

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    TASK_SIZE = args.task_size  # M bits
    NUM_TIME = args.num_time

    drop_ratio_set = args.drop_coefficient

    # GENERATE ENVIRONMENT
    env = Offload(args)
    if args.model == "D3QN":
        WAITING = False
    else:
        WAITING = True
    # GENERATE MULTIPLE CLASSES FOR RL
    RL = list()
    RL_wait = list()
    for iot in range(args.num_iot):
        # RL.append(Agent(lr=3e-4, discount_factor=0.99, num_actions=env.n_actions, epsilon=1.0, batch_size=64, input_dim=[env.n_features]))
        RL.append(D3QN(args, env.n_actions, env.n_features))
    if args.model == "DDPG":
        from RL_brain import DDPG

        for iot in range(args.num_iot):
            RL_wait.append(DDPG(args, env))

    # TRAIN THE SYSTEM
    train(RL, RL_wait, args)

    print("Training Finished")
