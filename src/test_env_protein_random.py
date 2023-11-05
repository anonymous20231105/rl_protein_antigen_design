import time
from torch.utils.tensorboard import SummaryWriter

from src.utils_protein_env import ProteinEnv
from src.utils_basic import get_time_str, wait_gpu_cool_writer


GPU_TEMPERATURE_LIMIT = 58  # None 55 58 60 65 70


def random_search(env):
    obs = env.reset()
    return_val = 0
    return_plddt_specify = 0
    while True:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        return_val += reward
        return_plddt_specify += info["plddt_specify"]

        if done:
            break
    return return_val, return_plddt_specify


def main():
    env = ProteinEnv(render_flag=False)
    opt_obj_max = 0
    start_time = time.time()
    last_epoch_time = start_time
    date_str, time_str = get_time_str()
    writer = SummaryWriter("../data/interim/tensorboard_search/" + date_str + "_" + time_str)
    total_step = 0
    while True:
        total_step += 1
        return_val, return_plddt_specify = random_search(env)
        opt_obj = return_val * return_plddt_specify
        if opt_obj > opt_obj_max:
            opt_obj_max = opt_obj
            print("time: ", time.time()-start_time)
            print("opt_obj_max: ", opt_obj_max)
            print("return_val: ", return_val)
            print("return_plddt_specify: ", return_plddt_specify)
            print("---------------------------------------------------")
            writer.add_scalar("result/opt_obj_max", opt_obj_max, total_step)
            writer.add_scalar("result/return_val", return_val, total_step)
            writer.add_scalar("result/return_plddt_specify", return_plddt_specify, total_step)
        now_time = time.time()
        epoch_time = now_time-last_epoch_time
        writer.add_scalar("time/epoch_time", epoch_time, total_step)
        last_epoch_time = now_time

        if GPU_TEMPERATURE_LIMIT is not None:
            wait_gpu_cool_writer(total_step, writer, class_name="time", gpu_temperature_limit=GPU_TEMPERATURE_LIMIT)


if __name__ == "__main__":
    main()
