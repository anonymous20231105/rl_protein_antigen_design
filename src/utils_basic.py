import time

import GPUtil
import numpy as np


def get_time_str():
    local_time = time.localtime(time.time())
    # date_str = str(local_time[0]) + str(local_time[1]) + str(local_time[2])
    date_str1 = str(local_time[0])
    if len(date_str1) == 1:
        date_str1 = "0" + date_str1
    date_str2 = str(local_time[1])
    if len(date_str2) == 1:
        date_str2 = "0" + date_str2
    date_str3 = str(local_time[2])
    if len(date_str3) == 1:
        date_str3 = "0" + date_str3
    date_str = date_str1 + date_str2 + date_str3

    time_str1 = str(local_time[3])
    if len(time_str1) == 1:
        time_str1 = "0" + time_str1
    time_str2 = str(local_time[4])
    if len(time_str2) == 1:
        time_str2 = "0" + time_str2
    time_str3 = str(local_time[5])
    if len(time_str3) == 1:
        time_str3 = "0" + time_str3
    time_str = time_str1 + time_str2 + time_str3
    return date_str, time_str


def compress_list(origin_list, goal_length=100):
    compress_ratio = int(len(origin_list) / goal_length)
    if compress_ratio > 1:
        compressed_list = []
        for i in range(int(len(origin_list) / compress_ratio)):
            compressed_list.append(np.mean(origin_list[i * compress_ratio : compress_ratio * (i + 1) - 1]))
    else:
        compressed_list = origin_list
    return compressed_list


def wait_gpu_cool_writer(total_step, writer, class_name, gpu_temperature_limit):
    start_time = time.time()
    gpus = GPUtil.getGPUs()  # about 0.1 second
    # print("GPUtil.getGPUs() time: ", time.time()-start_time)
    gpu_temperature = gpus[0].temperature
    writer.add_scalar(class_name + "/GpuTemperature", gpu_temperature, total_step)
    while gpu_temperature > gpu_temperature_limit:
        gpus = GPUtil.getGPUs()
        gpu_temperature = gpus[0].temperature
        total_step += 1
        writer.add_scalar(class_name + "/GpuTemperature", gpu_temperature, total_step)
        time.sleep(0.2)
    # writer.add_scalar(class_name + "/time_gpu_cool", time.time()-start_time, total_step)
    time_gpu_cool = time.time()-start_time
    return time_gpu_cool


def wait_gpu_cool(gpu_temperature_limit):
    gpus = GPUtil.getGPUs()
    gpu_temperature = gpus[0].temperature
    while gpu_temperature > gpu_temperature_limit:
        gpus = GPUtil.getGPUs()
        gpu_temperature = gpus[0].temperature
        time.sleep(0.2)
