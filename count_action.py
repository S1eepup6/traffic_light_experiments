import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from signal_config import signal_configs

import pickle

# log_path = "./past_logs/log_colight/compare/" + "action_logs.pickle"
# log_path = "./past_logs/log_colight/intas7/" + "action_logs.pickle"
# log_path = "./log_bm/MPLight-tr0-ingolstadt21-21-mplight-pressure/" + "action_logs.pickle"
# log_path = "./log_bm/CoLight-tr0-ingolstadt21-21-colight_state-wait_norm/" + "action_logs.pickle"
log_path = "./logs_0728/co_art44/CoLight0-tr0-arterial4x4-0-colight_state-wait_norm/" + "action_logs.pickle"
log_path = "./logs_0728/mp_art44/" + "action_logs.pickle"

with open(log_path,"rb") as f:
    al = pickle.load(f)

keys = list(al.keys())

for k in keys:
    change_action = 0
    for i in range(1, len(al[k])):
        if al[k][i - 1] != al[k][i]:
            change_action += 1
    display_key = k if len(k) < 15 else k[:15]
    # print("{} {} / {:3}".format(display_key, change_action, len(al[k])))
    tmp = []
    cnt = 1
    for i in range(1, len(al[k])):
        if al[k][i - 1] != al[k][i]:
            # tmp.append([al[k][i - 1], cnt])
            tmp.append(cnt)
            cnt = 1
        else:
            cnt += 1
    tmp.append(cnt)
    print("{} : {} , {:.3f} (/{})".format(display_key, np.max(tmp), np.mean(tmp), len(tmp)))
    # print("{} - {} :  ({}, {:.3f}) (/{})".format(display_key, tmp, np.max(tmp), np.mean(tmp), len(tmp)))


print()

for k in keys:
    display_key = k if len(k) < 15 else k[:15]
    print("{} : {} / {}".format(display_key, len(np.unique(al[k])), 5))
    tmp = len(np.unique(al[k])) / 5

    # print(k, np.unique(al[k]))

# a = [290.8187998124707, 274.4796728971963, 435.7949976624591, 290.52094547156565, 275.24031190926274, 267.2970691676436, 270.45420118343196, 272.29602803738317, 273.4394859813084]
# print(np.mean(a))