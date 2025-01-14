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

n_intas = [1, 7, 21]
n_intas = [21]

for n in n_intas:
    net = "./environments/InTAS-master/scenario/intas_gen.sumocfg"
    sumo_cmd = [sumolib.checkBinary('sumo'), '-c', net, '--no-warnings', 'True']

    intas_name = f"ingolstadt{n}"

    intersections = list(signal_configs[intas_name].keys())[2:]

    new_route_path = f'./environments/new_intas/intas{n}/new_route.xml'
    # new_route_path = 'tmp.xml'
    copyfile('./environments/intas_head.txt', new_route_path)
    new_route_file = open(new_route_path, 'a')
    new_route_file.write('\n')

    # lane_set = list()

    # for i in intersections:
    #     lanes = list(signal_configs[intas_name][i]['lane_sets'].values())
    #     for l in lanes:
    #         for id in l:
    #             lane_set.append(id)
    # lane_set = list(set(lane_set))

    
    # list_from = list()
    # list_to = list()

    # tree = elemTree.parse(f'./environments/ingolstadt{n}/ingolstadt{n}.rou.xml')
    # for trip in tree.findall('trip'):
    #     list_from.append(trip.get('from'))
    #     list_to.append(trip.get('to'))

    # del tree
    # list_from = list(set(list_from))
    # list_to = list(set(list_to))
    # lane_set = list(set(list_to + list_from))

    
    lane_set = list()
    tree = elemTree.parse(f'./environments/ingolstadt{n}/ingolstadt{n}.net.xml')
    for edge in tree.findall('edge'):
        lanes_in_edge = list(edge)
        for lane in lanes_in_edge:
            lane_set.append(lane.get('id'))
    
    print(lane_set)

    traci.start(sumo_cmd)
    sumo = traci.getConnection()

    targets = dict()

    TOTAL_STEP = 86400 * 10
    # TOTAL_STEP = (61200 - 57600) * 10

    for i in range(TOTAL_STEP):
        sumo.simulationStep()

        # 레인안에 들어온 차량 인식
        for l in lane_set:
            try:
                vehicle_in_lane = sumo.lane.getLastStepVehicleIDs(l)
            except:
                lane_set.remove(l)
                vehicle_in_lane = []
            if len(vehicle_in_lane) > 0:
                for v in vehicle_in_lane:
                    vtype = sumo.vehicle.getTypeID(v)
                    lane_id = l[:-2]
                    if v not in list(targets.keys()):
                        depart_step = (i * 0.1)
                        targets[v] = [depart_step, vtype, lane_id, lane_id]
                    else:
                        targets[v][3] = lane_id

        deleted = []
        for v in targets.keys():
            try:
                sumo.vehicle.getStopState(v)
            except:
                # print(f'\t<trip id="{v}" type="{targets[v][1]}" depart="{targets[v][0]}" from="{targets[v][2]}" to="{targets[v][3]}"/>\n')
                new_route_file.write(f'\t<trip id="{v}" type="{targets[v][1]}" depart="{targets[v][0]}" from="{targets[v][2]}" to="{targets[v][3]}"/>\n')
                deleted.append(v)

        for del_v in deleted:
            del targets[del_v]


    for vid in list(targets.keys()):
        new_route_file.write(f'\t<trip id="{vid}" type="{targets[vid][1]}" depart="{targets[vid][0]}" from="{targets[vid][2]}" to="{targets[vid][3]}"/>\n')


    new_route_file.write(f'</routes>\n')
    new_route_file.close()

    del targets