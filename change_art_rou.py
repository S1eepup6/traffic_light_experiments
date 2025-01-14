import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as elemTree
import numpy as np
from shutil import copyfile
from tqdm import tqdm


shape = '3x1'

edges = []

tree = elemTree.parse(f'./environments/arterial_others/arterial{shape}/arterial{shape}.net.xml')

for e in tree.findall('edge'):
    edges.append(e.get('id'))


print(edges)

for i in tqdm(range(1, 1401)):
    tree = elemTree.parse(f'./environments/arterial4x4/arterial4x4_{i}.rou.xml')
    new_rou_path = f'./environments/arterial_others/arterial{shape}/arterial{shape}_{i}.rou.xml'
    copyfile('./environments/arterial4x4_rou_head.txt', new_rou_path)
    new_rou_file = open(new_rou_path, 'a')
    new_rou_file.write('\n')

    for veh in tree.findall('vehicle'):
        valid_route = []
        veh_route = veh.findall('route')[0].get('edges').split()

        for r in veh_route:
            if r in edges:
                valid_route.append(r)

        if len(valid_route) >= 1:
            valid_edges = ' '.join(valid_route)
            v_attrib = veh.attrib
            new_rou_file.write("\t" + f"<vehicle id=\"{v_attrib['id']}\" type=\"{v_attrib['type']}\" depart=\"{v_attrib['depart']}\" departPos=\"{v_attrib['departPos']}\">\n")
            new_rou_file.write("\t\t" + f"<route edges=\"{valid_edges}\"/>\n")
            new_rou_file.write("\t" + f"</vehicle>\n")

    new_rou_file.write("</routes>")
    new_rou_file.close()