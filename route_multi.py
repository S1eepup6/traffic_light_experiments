import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as elemTree
import numpy as np
from shutil import copyfile
from tqdm import tqdm

i = 1331
mult = 5


for i in range(1, 101):
    tree = elemTree.parse(f'./environments/grid4x4/grid4x4_{i}.rou.xml')

    new_route_path = f'./environments/grid4x4/mult{mult}/grid4x4_{i}.rou.xml'
    copyfile('./environments/grid4x4_head.txt', new_route_path)
    new_route_file = open(new_route_path, 'a')
    new_route_file.write('\n')

    vid = 0
    for vehicle in tree.findall('vehicle'):
        vdepart = vehicle.get('depart')

        for m in range(mult):

            new_route_file.write(f"\t<vehicle id=\"{vid}\" depart=\"{vdepart}\">\n")

            for route in vehicle.findall('route'):
                r = route.get('edges')
                new_route_file.write(f"\t\t<route edges=\"{r}\"/>\n")
            new_route_file.write("\t</vehicle>\n")

            vid += 1

    new_route_file.write("\t</routes>")
    new_route_file.close()