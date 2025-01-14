import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as elemTree
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from dicttoxml import dicttoxml

shape = '3x1'
new_map_path = f'./environments/arterial_others/arterial{shape}/arterial{shape}.net.xml'
target_np = [10, 11, 12, 13,]
target_nt = [13, 14, 15]
nt_to_np = [9, 10, 11, 16]

def add_suffix(l, mode='np'):
    r = []
    for n in l:
        r.append(mode + str(n))
    return r

# def edge_xml(d):
#     if 'length' in list(d.keys()):
#         return f"<edge id=\"{d['id']}\" from=\"{d['from']}\" to=\"{d['to']}\" priority=\"{d['priority']}\" type=\"{d['type']}\" length=\"{d['length']}\">"
#     elif 'from' in list(d.keys()):
#         return f"<edge id=\"{d['id']}\" from=\"{d['from']}\" to=\"{d['to']}\" priority=\"{d['priority']}\" type=\"{d['type']}\">"
#     else:
#         return f"<edge id=\"{d['id']}\" function=\"{d['function']}\">"

# def edge_lane_xml(d):
#     return f"<lane id=\"{d['id']}\" index=\"{d['index']}\" speed=\"{d['speed']}\" length=\"{d['length']}\" shape=\"{d['shape']}\"/>"

# def tl_phase_xml(d):
#     return f"<phase duration=\"{d['duration']}\" state=\"{d['state']}\"/>"

# def junc_xml(d, nt_to_np=False):
#     if 'shape' in list(d.keys()):
#         if nt_to_np:
#             return f"<junction id=\"{d['id']}\" type=\"priority\" x=\"{d['x']}\" y=\"{d['y']}\" incLanes=\"{d['incLanes']}\" intLanes=\"{d['intLanes']}\" shape=\"{d['shape']}\">"
#         else:
#             return f"<junction id=\"{d['id']}\" type=\"{d['type']}\" x=\"{d['x']}\" y=\"{d['y']}\" incLanes=\"{d['incLanes']}\" intLanes=\"{d['intLanes']}\" shape=\"{d['shape']}\">"
#     else:
#         return f"<junction id=\"{d['id']}\" type=\"internal\" x=\"{d['x']}\" y=\"{d['y']}\" incLanes=\"{d['incLanes']}\" intLanes=\"{d['intLanes']}\" />"


# def junc_req_xml(d):
#     return f"<request index=\"{d['index']}\" response=\"{d['response']}\" foes=\"{d['foes']}\" cont=\"{d['cont']}\"/>"

def dict_to_xml(d, title="connection", one_line=True):
    result = f"<{title} "
    for k in list(d.keys()):
        result += f"{k}=\"{d[k]}\" "
    if one_line:
        result += "/>"
    else:
        result += ">"
    return result

tree = elemTree.parse(f'./environments/arterial4x4/arterial4x4.net.xml')
copyfile('./environments/arterial4x4_head.txt', new_map_path)
new_map_file = open(new_map_path, 'a')
new_map_file.write('\n')

target_np = add_suffix(target_np)
target_nt = add_suffix(target_nt, 'nt')
nt_to_np = add_suffix(nt_to_np, 'nt')

nec = target_np + target_nt + nt_to_np

edges = []
lanes = []

for edge in tree.findall('edge'):
    edge_j = edge.get('id').split('_')
    j_from = edge_j[0]
    j_to = edge_j[1]

    if j_to in nt_to_np and j_from in nt_to_np:
        pass
    elif j_to in nec and j_from in nec:
        edges.append(edge.get('id'))

        new_map_file.write("\t" + dict_to_xml(edge.attrib, "edge", False) + "\n")
        for l in edge.findall('lane'):
            lanes.append(l.get('id'))
            new_map_file.write("\t" + "\t" + dict_to_xml(l.attrib, "lane") + "\n")
        new_map_file.write("\t" + "</edge>" + "\n")
        new_map_file.write("\n")
    elif j_from[0] != "n" and (j_from[1:] in nec or j_from in nec):
        edges.append(edge.get('id'))

        new_map_file.write("\t" + dict_to_xml(edge.attrib, "edge", False) + "\n")
        for l in edge.findall('lane'):
            lanes.append(l.get('id'))
            new_map_file.write("\t" + "\t" + dict_to_xml(l.attrib, "lane") + "\n")
        new_map_file.write("\t" + "</edge>" + "\n")
        new_map_file.write("\n")


for tl in tree.findall('tlLogic'):
    for junc in target_nt:
        if junc == tl.get('id'):
            tmp_d = tl.attrib
            new_map_file.write("\t" + f"<tlLogic id=\"{tmp_d['id']}\" type=\"{tmp_d['type']}\" programID=\"{tmp_d['programID']}\" offset=\"{tmp_d['offset']}\">" + "\n")
            for p in tl.findall('phase'):
                new_map_file.write("\t" + "\t" + dict_to_xml(p.attrib, "phase", one_line=True) + "\n")
            new_map_file.write("\t" + "</tlLogic>" + "\n")
            new_map_file.write("\n")

for j in tree.findall('junction'):
    j_attrib = j.attrib
    inclane = []
    intlane = []

    for l in j_attrib['incLanes'].split():
        if l in lanes:
            inclane.append(l)
    j_attrib['incLanes'] = " ".join(inclane)

    for l in j_attrib['intLanes'].split():
        if l in lanes:
            intlane.append(l)
    j_attrib['intLanes'] = " ".join(intlane)

    for junc in target_nt + target_np:
        if junc == j.get('id'):
            new_map_file.write("\t" + dict_to_xml(j_attrib, "junction", False) + "\n")
            for req in j.findall('request'):
                new_map_file.write("\t" + "\t" + dict_to_xml(req.attrib, "request", True) + "\n")
            new_map_file.write("\t" + "</junction>" + "\n")

    for junc in nt_to_np:
        if junc == j.get('id'):
            j_attrib['type'] = "priority"
            new_map_file.write("\t" + dict_to_xml(j_attrib, "junction", False) + "\n")
    
            new_map_file.write("\t" + "\t<request index=\"0\" response=\"0\" foes=\"0\" cont=\"0\"/>" + "\n")
    
            new_map_file.write("\t" + "</junction>" + "\n")
    
    for junc in nec:
        if junc == j.get('id').split("_")[0][1:]:
            new_map_file.write("\t" + dict_to_xml(j_attrib, "junction", True) + "\n")

new_map_file.write("\n")

check_attrib = edges + lanes
for c in tree.findall('connection'):
    c_attrib = c.attrib

    if 'tl' in list(c.attrib.keys()):
        if c.get('tl') not in target_nt:
            del c_attrib['tl']

    if c.get('from') in check_attrib and c.get('to') in check_attrib:
        if 'via' in list(c.attrib.keys()):
            if c.get('via') in check_attrib:
                new_map_file.write("\t" + dict_to_xml(c_attrib, "connection", True) + "\n")
        else:
            new_map_file.write("\t" + dict_to_xml(c_attrib, "connection", True) + "\n")

new_map_file.write("</net>")

new_map_file.close()

print(edges)
print(lanes)