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

import sys

n = 21
n_k = int(sys.argv[1])
from_google_maps = False
signal_configs_geo = deepcopy(signal_configs)

intersections = ['2330725114', "243749571", "cluster_2302665030_2337351369", "cluster_1863241547_1863241548_1976170214", "1863241632" ,"89173763",
                "89173808", "243351999", "243641585", "cluster_1840209209_268417350", "cluster_1427494838_273472399", "89127267", "30503246",
                "cluster_1757124350_1757124352", "cluster_1041665625_cluster_1387938793_1387938796_cluster_1757124361_1757124367_32564126",
                "cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190",
                "32564122", "cluster_1833965782_1833965806_371781950_cluster_32564118_371775504", "cluster_371462086_469470779_98101387_cluster_371462067_371775459_371775468",
                "cluster_274083968_cluster_1200364014_1200364088", "30624898"]

gne = {"cluster_274083968_cluster_1200364014_1200364088" : "gneJ207",
        "cluster_1041665625_cluster_1387938793_1387938796_cluster_1757124361_1757124367_32564126" : "gneJ143",
        "cluster_1833965782_1833965806_371781950_cluster_32564118_371775504" : "gneJ255",
        "cluster_371462086_469470779_98101387_cluster_371462067_371775459_371775468" : "gneJ210",
        "cluster_1840209209_268417350" : "gneJ257",
        "cluster_2302665030_2337351369" : "gneJ208"}

intersections = list(set(intersections))
distance = dict()
junctions = dict()


####################################################################################

if from_google_maps:
    junctions = {
        '2330725114' : [48.77762484735596, 11.394781265435022],
        '1863241632' : [48.77496513230185, 11.39629855185818],
        'cluster_1863241547_1863241548_1976170214' : [48.77350522880159, 11.396632354871274],
        '89173763' : [48.76970528083727, 11.39462953650187],
        '89173808' : [48.76410483344589, 11.394750919415724],
        'cluster_1757124350_1757124352' : [48.763754784653074, 11.412275577874315],
        'gneJ143' : [48.76484492837978, 11.411896255997409],
        'gneJ207' : [48.76630508369529, 11.411441070070463],
        'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190' : [48.7672851640784, 11.411729354747193],
        '243351999' : [48.76867486784927, 11.408083008818931],
        '32564122' : [48.76999492935441, 11.414614820589792],
        'gneJ255' : [48.7716360385052, 11.417231154075381],
        'gneJ210' : [48.77196901062218, 11.419775312687364],
        'gneJ208' : [48.77843856615253, 11.398007386963338],
        '243749571' : [48.7799010371926, 11.401354438282663],
        'cluster_1427494838_273472399' : [48.78337874780824, 11.412707177407393],
        '89127267' : [48.77991731226935, 11.415391087550928],
        '243641585' : [48.77520058227119, 11.409699002034564],
        'gneJ257' : [48.774981397393816, 11.409859934562387],
        '30624898' : [48.77516819102035, 11.417138792195896],
        '30503246' : [48.77494320956896, 11.41711603289955]
    }
else:
    tree = elemTree.parse(f'./environments/ingolstadt{n}/ingolstadt{n}.net.xml')
    for junc in tree.findall('junction'):
        id = junc.get('id')
        if id in intersections:
            if id in list(gne.keys()):
                id = gne[id]
            junctions[id] = [float(junc.get('x')), float(junc.get('y'))]

for i in list(junctions.keys()):
    distance[i] = list()
    for j in list(junctions.keys()):
        dist = (((junctions[i][0] - junctions[j][0]) ** 2) + ((junctions[i][1] - junctions[j][1]) ** 2)) ** 0.5

        distance[i].append([j, dist])

    distance[i].sort(key=lambda x : x[1])

for k in list(distance.keys()):
    print(k)
    top_k = list()
    for i in distance[k][1:1+n_k]:
        top_k.append(i[0])
    signal_configs_geo['ingolstadt21'][k]['geo_neighbor'] = top_k

with open('signal_configs_geo.py', 'w') as f:
    f.write("signal_configs_geo = ")
    pprint.pprint(signal_configs_geo, stream=f)
####################################################################################

# net = f"./environments/ingolstadt{n}/ingolstadt{n}.sumocfg"
# sumo_cmd = [sumolib.checkBinary('sumo'), '-c', net, '--no-warnings', 'True']
# traci.start(sumo_cmd)
# sumo = traci.getConnection()
# print('\n')

# for junc in intersections:
#     junctions[junc] = list(sumo.junction.getPosition(junc))

# for i in list(junctions.keys()):
#     if i in list(gne.keys()):
#         src_tl_id = gne[i]
#     else:
#         src_tl_id = i
#     distance[src_tl_id] = dict()
#     for j in list(junctions.keys()):
#         dist = sumo.simulation.getDistance2D(junctions[i][0], junctions[i][1], junctions[j][0], junctions[j][1], isGeo=True)
#         if j in list(gne.keys()):
#             dest_tl_id = gne[j]
#         else:
#             dest_tl_id = j
#         distance[src_tl_id][dest_tl_id] = dist

# for k in list(distance.keys()):
#     for d in list(distance[k].keys()):
#         if distance[k][d] < 0 and distance[d][k] > 0:
#             distance[k][d] = distance[d][k]

# dist_tmp = dict()
# for k in list(distance.keys()):
#     dist_tmp[k] = list()
#     for d in list(distance[k].keys()):
#         dist_tmp[k].append([d, distance[k][d]])
#     dist_tmp[k].sort(key=lambda x : x[1])
# distance = dist_tmp

# for k in list(distance.keys()):
#     print(k)
#     pprint.pprint(distance[k])
#     print()
# print(len(distance))


# traci.close()