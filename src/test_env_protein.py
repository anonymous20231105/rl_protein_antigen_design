import numpy as np

from src.utils_protein_env import ProteinEnv, ProteinEnvDDG, ProteinEnvDouble


def main_env():
    env = ProteinEnv(render_flag=True)
    obs = env.reset()
    print("obs: ", obs)
    return_val = 0
    plddt_specify = 0
    while True:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        return_val += reward
        if info["plddt_specify"] is not None:
            plddt_specify = info["plddt_specify"]

        if done:
            print("return_val: ", return_val)
            print("return_plddt_specify: ", plddt_specify)
            break
    print("Finished...")


def main_env_double():
    env = ProteinEnvDouble(render_flag=True)
    obs = env.reset()
    print("obs: ", obs)
    return_val = 0
    plddt_specify = 0
    while True:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        return_val += reward
        if info["plddt_specify"] is not None:
            plddt_specify = info["plddt_specify"]

        if done:
            print("return_val: ", return_val)
            print("return_plddt_specify: ", plddt_specify)
            break
    print("Finished...")


def main_env_ddg():
    env = ProteinEnvDDG()
    obs = env.reset(set_index=0)
    print("obs: ", obs)
    return_val = 0
    while True:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        return_val += reward

        if done:
            print("return_val: ", return_val)
            break
    print("Finished...")


def main_env_ddg_multi():
    env = ProteinEnvDDG()
    return_val_list = []
    for i in range(53):
        obs = env.reset(set_index=i)
        return_val = 0
        while True:
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            return_val += reward

            if done:
                print("return_val: ", return_val)
                return_val_list.append(return_val)
                break
    print("np.mean(return_val_list): ", np.mean(return_val_list))
    print("Finished...")


def main():
    # main_env()
    main_env_double()
    # main_env_ddg()
    # main_env_ddg_multi()


if __name__ == "__main__":
    main()
