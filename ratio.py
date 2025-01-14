import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME")
import traci
import sumolib

from shutil import copyfile
from signal_config import *
import xml.etree.ElementTree as elemTree
from tqdm import tqdm
import numpy as np

import pprint
from signal_config import signal_configs
from copy import deepcopy

import matplotlib.pyplot as plt
import sys
import argparse


if __name__ == "__main__":
    absolute_ratio = True

    print(absolute_ratio)

    ###################################################################################

    no_action = dict()
    target = dict()
    ratio = dict()

    ratio_with_key = list()

    no_action_lost = list()
    target_lost = list()

    not_arrived = 0
    can_arrived = 0
    ####################################################################################

    # NO ACTION
    tree = elemTree.parse(f'.\\logs_rule_based\\NoAction-tr0-ingolstadt21-21-no-no\\tripinfo_9.xml')    #intas21
    tree = elemTree.parse(f'.\\logs_rule_based\\NoAction-tr0-arterial4x4-0-no-no\\tripinfo_9.xml')      #art 44
    for trip in tree.findall('tripinfo'):
        if float(trip.get('arrival')) < 0:
            no_action_lost.append(trip.get('id'))
        else:
            no_action[trip.get('id')] = float(trip.get('arrival')) - float(trip.get('depart'))

    # MODEL
    # tree = elemTree.parse(f'.\\log_bm\\MPLight-tr0-ingolstadt21-21-mplight-pressure\\tripinfo_60.xml')
    # tree = elemTree.parse(f'.\\log_bm\\CoLight-tr0-ingolstadt21-21-colight_state-wait_norm\\tripinfo_90.xml')
    tree = elemTree.parse(f'.\\logs_0728\\co_art44\\CoLight0-tr0-arterial4x4-0-colight_state-wait_norm\\tripinfo_70.xml')
    tree = elemTree.parse(f'.\\logs_0728\\mp_art44\\MPLight0-tr0-arterial4x4-0-mplight-pressure\\tripinfo_40.xml')

    trips = []
    for trip in tree.findall('tripinfo'):
        # if float(trip.get('arrival')) > 0:
        if True:
            trips.append(float(trip.get('duration')))
    print(np.average(trips))

    for trip in tree.findall('tripinfo'):
        if float(trip.get('arrival')) < 0:
            target_lost.append(trip.get('id'))
        else:
            target[trip.get('id')] = float(trip.get('arrival')) - float(trip.get('depart'))

    # RATIO
    for t in list(no_action.keys()):
        if t in list(target.keys()):
            if not absolute_ratio:
                ratio[t] = target[t] / no_action[t]
                ratio_with_key.append([t, target[t] / no_action[t]])
            else:
                ratio[t] = target[t] - no_action[t]
                ratio_with_key.append([t, target[t] - no_action[t]])

        else: 
            # print(t)
            not_arrived += 1

    for n in no_action_lost:
        if n in list(target.keys()):
            can_arrived += 1

    ratio_values = list(ratio.values())
    ratio_values.sort()
    ratio_with_key.sort(key = lambda x : x[1])

    # print(list(no_action.values()))
    # print(list(target.values()))
    # print(ratio)
    # print()

    if not absolute_ratio:
        under_basis = len(np.where(np.array(ratio_values) < 1)[0]) / len(ratio_values)
        over_ground_avg = np.average(ratio_values[np.where(np.array(ratio_values) >= 0)[0][0]:])
    else:
        under_basis = len(np.where(np.array(ratio_values) < 0)[0]) / len(ratio_values)
        over_ground_avg = np.average(ratio_values[np.where(np.array(ratio_values) >= 1)[0][0]:])

    avg = np.average(ratio_values)
    print(f"도달 가능했으나 도달X = {not_arrived} / 도달 불능했으나 도달O = {can_arrived} / ground 미만 비율 = {under_basis} / 비율 평균 = {avg} / ground 이상 평균 = {over_ground_avg}")

    print(ratio_with_key[-10:])

    # plt.title("CoLight-Geo SingleGAT {:.3f} - {:.3f}".format(under_basis, avg))
    plt.title("MPLight {:.3f} - {:.3f}".format(under_basis, avg))
    plt.scatter(range(len(ratio_values)), ratio_values, s=15)

    if not absolute_ratio:
        plt.plot(range(len(ratio_values)), [1,] * len(ratio_values), color='red')
    else:
        plt.plot(range(len(ratio_values)), [0,] * len(ratio_values), color='red')

    plt.show()