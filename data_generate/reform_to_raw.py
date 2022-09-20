"""This file is used to export the relationships in generated family trees and reformalized to standard form."""
import os
from tqdm import tqdm
path = 'out'
outputfile = './facts_raw.txt'
file_list = os.listdir(path)
file_list = list(filter(lambda x: '.relations.data' in x and '.pred' not in x, file_list))
with open(outputfile, 'w'):
    pass
for file_name in tqdm(file_list):
    with open(os.path.join(path, file_name), 'r') as f:
        lines = f.readlines()
    lines = list(filter(lambda x: x[0]=='+', lines))
    if '.inf' in file_name:
        index = file_name[:-19]
    else:
        index = file_name[:-15]
    with open(outputfile, 'a') as wf:
        for line in lines:
            head = line.split()[1]
            relation = line.split()[2]
            tail = line.split()[3]
            wf.write("{}.{} {} {}.{}\n".format(index, head, relation, index, tail))