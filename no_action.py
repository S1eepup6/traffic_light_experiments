import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME")
import traci
import sumolib
import pathlib
import argparse

from shutil import copyfile
from signal_config import *
from tqdm import tqdm

from map_config import map_configs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--tr", type=int, default=0)  
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--map", type=str, default='ingolstadt21',
                    choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                                'cologne1', 'cologne3', 'cologne8',
                                ])

    args = ap.parse_args()

    GUI = args.gui
    env = args.map
    tr = args.tr

    net = map_configs[env]['net']
    route = map_configs[env]['route']

        
    log_dir = str(pathlib.Path().absolute())+os.sep+'logs_rule_based'+os.sep
    log_dir = str(pathlib.Path().absolute())+os.sep+'logs'+os.sep

    total_log_path = log_dir + f"NoAction-tr{tr}-{env}-0-no-no"

    if not os.path.exists(total_log_path):
        os.mkdir(total_log_path)

    print(args)

    for i in range(1, args.eps+1):    
        print(env)
        print(net)
        print(route + '_' + str(i) + '.rou.xml')
        print(total_log_path)

        if GUI:
            sumo_cmd = [sumolib.checkBinary('sumo-gui')]
            sumo_cmd.append('--start')
        else:
            sumo_cmd = [sumolib.checkBinary('sumo')]
            
        if route is not None:
            sumo_cmd += ['-n', net, '-r', route + '_' + str(i) + '.rou.xml']
        else:
            sumo_cmd += ['-c', net]
            
        sumo_cmd += ['--no-warnings', 'True']
        sumo_cmd += ['--random', '--time-to-teleport', '-1', '--tripinfo-output',
                    total_log_path + os.sep + 'tripinfo_' + str(i) + '.xml',
                    '--tripinfo-output.write-unfinished',
                    '--no-step-log', 'True',]

        print(sumo_cmd)

        traci.start(sumo_cmd)
        sumo = traci.getConnection()

        for _ in tqdm(range(3600)):
        # for _ in tqdm(range(7200)):
            sumo.simulationStep()

            # for signal_id in self.signals:
            #     signal = self.signals[signal_id]
            #     queue_length, max_queue = 0, 0
            #     for lane in signal.lanes:
            #         queue = signal.full_observation[lane]['queue']
            #         if queue > max_queue: max_queue = queue
            #         queue_length += queue
            #     queue_lengths[signal_id] = queue_length
            #     max_queues[signal_id] = max_queue
            # self.metrics.append({
            #     'step': self.sumo.simulation.getTime(),
            #     'reward': rewards,
            #     'max_queues': max_queues,
            #     'queue_lengths': queue_lengths
            # })
        traci.close()

        