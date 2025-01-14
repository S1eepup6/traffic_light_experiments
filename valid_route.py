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

# n_intas = [1, 7, 21]
n_intas = [21]

for n in n_intas:
    ################################ 루트 depart로 정렬 ################################


    
    sorted_route_path = f'./environments/new_intas/intas{n}/new_route_sorted.xml'
    # sorted_route_path = 'tmp.xml'
    copyfile('./environments/intas_head.txt', sorted_route_path)
    sorted_route_file = open(sorted_route_path, 'a')
    sorted_route_file.write('\n')

    #############################################

    lane_set = list()

    list_from = list()
    list_to = list()

    tree = elemTree.parse(f'./environments/ingolstadt{n}/ingolstadt{n}.rou.xml')
    for trip in tree.findall('trip'):
        list_from.append(trip.get('from'))
        list_to.append(trip.get('to'))

    del tree
    list_from = list(set(list_from))
    list_to = list(set(list_to))
    lane_set = list(set(list_to + list_from))

    #############################################

    trips = list()
    tree = elemTree.parse(f'./environments/new_intas/intas{n}/new_route.xml')
    for trip in tree.findall('trip'):
        if n == 1:
            trips.append([trip.get('id'), trip.get('type'), trip.get('depart'), trip.get('from'), trip.get('to')])
        else:
            if (trip.get('from') in lane_set) and (trip.get('to') in lane_set):
                trips.append([trip.get('id'), trip.get('type'), trip.get('depart'), trip.get('from'), trip.get('to')])

    print(len(trips))
    trips = sorted(trips, key=lambda trips: float(trips[2]))
    for t in trips:
        sorted_route_file.write(f'\t<trip id="{t[0]}" type="{t[1]}" depart="{t[2]}" from="{t[3]}" to="{t[4]}"/>\n')

    sorted_route_file.write(f'</routes>\n')
    sorted_route_file.close()


    ################################# 불가능한 루트 제거 #################################
    net = f"./environments/ingolstadt{n}/ingolstadt{n}.sumocfg"
    sumo_cmd = [sumolib.checkBinary('sumo'), '-c', net, '--no-warnings', 'True']

    intas_name = f"ingolstadt{n}"

    intersections = list(signal_configs[intas_name].keys())[2:]

    src_route_path = sorted_route_path
    target_route_path = f"./environments/ingolstadt{n}/simul_route_valid.xml"

    copyfile('./environments/intas_head.txt', target_route_path)
    target_route_file = open(target_route_path, 'a')
    target_route_file.write('\n')

    traci.start(sumo_cmd)
    sumo = traci.getConnection()

    targets = dict()

    lane_set = list()
    tree = elemTree.parse(src_route_path)
    for trip in tqdm(tree.findall('trip')):
        s_edge = trip.get('from')
        e_edge = trip.get('to')

        if len(sumo.simulation.findRoute(s_edge, e_edge).__dict__["edges"]) > 0:
            vid = trip.get('id')
            vtype = trip.get('type')
            vdepart = trip.get('depart')
            target_route_file.write(f'\t<trip id="{vid}" type="{vtype}" depart="{vdepart}" from="{s_edge}" to="{e_edge}"/>\n')

    
    target_route_file.write(f'</routes>\n')
    target_route_file.close()

    del targets