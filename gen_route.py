import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as elemTree
import numpy as np
from shutil import copyfile
from tqdm import tqdm

np.random.seed(42)

[1, 7, 21]
for n_intersection in [7]:
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

    new_route_path = f'./environments/new_intas/intas{n_intersection}/new_route.xml'
    copyfile('./environments/intas_head.txt', new_route_path)
    new_route_file = open(new_route_path, 'a')
    new_route_file.write('\n')

    for i in range(1, 22+1):
        path_num = str(i)
        if i < 10:
            path_num = '0' + path_num


        tree = elemTree.parse(f'./environments/original_intas/InTAS_0{path_num}.rou.xml')
        for vehicle in tree.findall('vehicle'):
            vid = vehicle.get('id')
            vtype = vehicle.get('type')
            vdepart = vehicle.get('depart')

            routes = list(list(vehicle)[0])
            prob = []
            for route in routes:
                prob.append(float(route.get('probability')))
            prob = np.array(prob)
            prob = prob / np.sum(prob)
            sampled = np.random.choice(len(prob), 1, p=prob)

            selected_route = routes[sampled[0]]
            edges = selected_route.get('edges').split()

            avail_from = (-1, None)
            avail_to = (-1, None)

            for e in enumerate(edges):
                if e[1] in list_from:
                    avail_from = e

            for e in enumerate(edges[avail_from[0]:]):
                if e[1] in list_to:
                    avail_to = e

            if avail_from[0] > -1 and avail_to[0] > -1:
                avail_from = avail_from[1]
                avail_to = avail_to[1]

                new_route_file.write(f'\t<trip id="{vid}" type="{vtype}" depart="{vdepart}" from="{avail_from}" to="{avail_to}"/>\n')

    new_route_file.write(f'</routes>\n')
    new_route_file.close()