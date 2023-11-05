import numpy as np


def cut_list(cut_steps, list_train_return, list_train_steps):
    i_train_step = None
    for i_train_step in list_train_steps:
        if i_train_step > cut_steps:
            break
    temp_index = list_train_steps.index(i_train_step)
    list_train_return = list_train_return[:temp_index]
    list_train_steps = list_train_steps[:temp_index]
    return list_train_return, list_train_steps


def draw_aver(n_curves, ax, name="average", color='b', total_time=None, step=1000, line_width=1.5):
    # assert len(curve1) == len(curve2), "The input list length is not equal!"
    len_list = []
    for i_curve in n_curves:
        len_list.append(len(i_curve))
    length = min(len_list)
    curve_list = []
    for i_curve in n_curves:
        curve_list.append(i_curve[0:length])
    curve_list_t = np.array(curve_list).T
    curve_avr = []
    curve_max = []
    curve_min = []
    for i in curve_list_t:
        # curve_avr.append(np.median(i))
        curve_avr.append(i.mean())
        curve_max.append(i.max())
        curve_min.append(i.min())
    # 多边形
    if total_time is None:
        x = list(range(length)) + list(range(length - 1, -1, -1))
    else:
        x = [(total_time * i / step) for i in range(step)] + [(total_time * i / step) for i in range(step-1, -1, -1)]
    curve_min.reverse()
    y = curve_max + curve_min
    # ax = plt.gca()
    if color == 'r' or color == 1:
        ax.fill(x, y, 'lightcoral', alpha=0.3)  # plum
        plot_color = 'deeppink'
    elif color == 'y' or color == 2:
        ax.fill(x, y, 'moccasin', alpha=0.5)
        plot_color = 'orange'
    elif color == 'g' or color == 3:
        ax.fill(x, y, 'lightgreen', alpha=0.5)
        plot_color = 'green'
    elif color == 'c' or color == 4:
        ax.fill(x, y, 'paleturquoise', alpha=0.5)  #  aquamarine
        plot_color = 'c'
    elif color == 'b' or color == 5:
        ax.fill(x, y, 'lightblue', alpha=0.5)
        plot_color = 'dodgerblue'
    elif color == 'm' or color == 6:
        ax.fill(x, y, 'thistle', alpha=0.5)
        plot_color = 'purple'
    else:
        print("Unknown color!")
        plot_color = None
    if total_time is None:
        ax.plot(curve_avr, plot_color, label=name, linewidth=line_width, alpha=0.7)
    else:
        time_list = [(total_time * i / step) for i in range(step)]
        ax.plot(time_list, curve_avr, plot_color, label=name, linewidth=line_width, alpha=0.7)
    print('show...')
    return length, ax


def draw_time_aver(time_list, n_curves, ax, name="default_legend", color='b', step=1000):
    total_times = [i[-1] for i in time_list]
    min_time = min(total_times)
    new_n_curves = []
    for i_curve in range(len(n_curves)):
        new_temp_curves = []
        index = 0
        last_value = n_curves[i_curve][0]
        for j in range(step):
            j_time = min_time / step * (j + 1)
            temp_rewards = []
            while time_list[i_curve][index] < j_time:
                temp_rewards.append(n_curves[i_curve][index])
                index += 1
            if len(temp_rewards) > 0:
                new_value = np.mean(temp_rewards)
            else:
                new_value = last_value
            new_temp_curves.append(new_value)
            last_value = new_value
        new_n_curves.append(new_temp_curves)

    draw_aver(new_n_curves, ax, name, color, total_time=min_time, step=step)
