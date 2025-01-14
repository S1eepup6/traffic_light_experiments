import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as elemTree
import numpy as np
from shutil import copyfile
from tqdm import tqdm

np.random.seed(42)

n_intersection = 1
list_from = list()
list_to = list()

tree = elemTree.parse(f'./environments/ingolstadt{n_intersection}/ingolstadt{n_intersection}.rou.xml')
for trip in tree.findall('trip'):
    list_from.append(trip.get('from'))
    list_to.append(trip.get('to'))

del tree
list_from = list(set(list_from))
list_to = list(set(list_to))

print(list_from)
print(list_to)

# new_route_path = f'tmp.xml'
# copyfile('./environments/intas_head.txt', new_route_path)
# new_route_file = open(new_route_path, 'a')
# new_route_file.write('\n')

tree = elemTree.parse(f'./environments/InTAS.simulation.tripinfo.xml')
for trip in tree.findall('tripinfo'):
    if trip.get('departLane')[:-2] in list_from:
        print(trip.get('id'))
# new_route_file.write(f'</routes>\n')
# new_route_file.close()