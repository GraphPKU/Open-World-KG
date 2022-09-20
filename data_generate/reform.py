"""This file is used to reformalize the data to fit Q2B and BetaE."""
import pickle
import argparse
from collections import defaultdict
import os
import shutil

def q2bque(path, savepath, name):
    with open(os.path.join(path, name), 'rb') as f:
        q = pickle.load(f)
        t = list(q)[0]
        assert isinstance(t[0], int)
        q = set([(t[0], (t[1],)) for t in q])
        output = defaultdict(set)
        output[('e', ('r',))] = q
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open(os.path.join(savepath, name), 'wb') as f:
        pickle.dump(output, f)

def q2bans(path, savepath, name):
    with open(os.path.join(path, name), 'rb') as f:
        answers = pickle.load(f)
        output = defaultdict(set)
        keys = answers.keys()
        for k in keys:
            output[(k[0], (k[1],))] = answers[k]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open(os.path.join(savepath, name), 'wb') as f:
        pickle.dump(output, f)
    
def q2b(path, savepath):
        q2bque(path, savepath, 'train-queries.pkl')
        q2bque(path, savepath, 'test-queries.pkl')
        q2bans(path, savepath, 'train-answers.pkl')
        q2bans(path, savepath, 'test-easy-answers.pkl')
        q2bans(path, savepath, 'test-full-hard-answers.pkl')
        q2bans(path, savepath, 'test-sparse-hard-answers.pkl')
        shutil.copy(os.path.join(path, '../../stats.txt'), savepath)
        
def transe(path, savepath, nentity, nrelation):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open(os.path.join(savepath, 'train.txt'), 'w'):
        pass
    with open(os.path.join(path, 'graph_sparse_train.pkl'), 'rb') as f:
        graph = pickle.load(f)
        for k in graph.keys():
            head = k[0]
            relation = k[1]
            with open(os.path.join(savepath, 'train.txt'), 'a') as wf:
                for tail in graph[k]:
                    wf.write('{} {} {}\n'.format(head, relation, tail))
    with open(os.path.join(savepath, 'entities.dict'), 'w') as wf:
        for i in range(nentity):
            wf.write('{} {}\n'.format(i, i))
    with open(os.path.join(savepath, 'relations.dict'), 'w') as wf:
        for i in range(nrelation):
            wf.write('{} {}\n'.format(i, i))
    for filename in ['test-queries.pkl', 'test-easy-answers.pkl', 'test-sparse-hard-answers.pkl', 'test-full-hard-answers.pkl']:
        shutil.copy(os.path.join(path, filename), savepath)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Data reform',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--sparse', nargs='+', type=int, help='a list whose element should be an int from 0 to 100')
    parser.add_argument('--type', type=str, choices=['q2b', 'transe'], required=True, help='reform type')
    parser.add_argument('--path', type=str, default='./raw', help='data path')
    parser.add_argument('--savepath', type=str, default='.', help='saving path')
    return parser.parse_args(args)    
        
if __name__ == '__main__':
    args = parse_args()
    type_savepath = os.path.join(args.savepath, args.type)
    with open(os.path.join(args.path, '../stats.txt'), 'r') as f:
        lines = f.readlines()
        nentity = int(lines[0].split()[-1])
        nrelation = int(lines[1].split()[-1])
    for sparse in args.sparse:
        path = os.path.join(args.path, str(sparse))
        savepath = os.path.join(type_savepath, str(sparse))
        if args.type == 'q2b':
            q2b(path, savepath)
        elif args.type == 'transe':
            transe(path, savepath, nentity, nrelation)


