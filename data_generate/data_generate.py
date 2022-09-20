from collections import defaultdict
import os
import pickle
import random
from tqdm import tqdm
import argparse
import numpy as np

def re_id(readfile, writepath, writefile):
    with open(os.path.join(writepath, writefile), 'w'):
        pass
    ent_to_id = dict()
    next_ent = 0
    rel_to_id = dict()
    next_rel = 0
    with open(readfile, 'r') as rf:
        while True:
            line = rf.readline()
            if not line:
                break
            if line.split()[0] in ent_to_id.keys():
                headid = ent_to_id[line.split()[0]]
            else:    
                headid = next_ent
                ent_to_id[line.split()[0]] = headid
                next_ent += 1
            
            if line.split()[1] in rel_to_id.keys():
                relid = rel_to_id[line.split()[1]]
                # inv_relid = rel_to_id['inv_'+line.split()[1]]
            else:
                relid = next_rel
                # inv_relid = next_rel + 1
                # next_rel += 2
                next_rel += 1
                rel_to_id[line.split()[1]] = relid
                # rel_to_id['inv_'+line.split()[1]] = inv_relid
            
            if line.split()[2] in ent_to_id.keys():
                tailid = ent_to_id[line.split()[2]]
            else:
                tailid = next_ent
                ent_to_id[line.split()[2]] = tailid
                next_ent += 1
                
            with open(os.path.join(writepath, writefile), 'a') as wf:
                wf.write('{} {} {}\n'.format(headid, relid, tailid))
                # wf.write('{} {} {}\n'.format(tailid, inv_relid, headid))
    with open(os.path.join(writepath, 'ent2id.pkl'), 'wb') as f:
        pickle.dump(ent_to_id, f)
    with open(os.path.join(writepath, 'rel2id.pkl'), 'wb') as f:
        pickle.dump(rel_to_id, f)
    print('Re_index finished! Num of entities: {}. Num of relations: {}'.format(next_ent, next_rel))
    with open(os.path.join(writepath, 'stats.txt'), 'w') as wf:
        wf.write('numentity: {}\nnumrelation: {}'.format(next_ent, next_rel))
    return next_ent, next_rel
    
def generate_graph(path, prefix, sparse, training_rate, seed, factfile, mode, nentity):
    if mode == 'node':
        node_prob = np.random.normal(np.sqrt(sparse/100), 0.2, nentity)
    graph = []
    graph_full = []
    save_path = os.path.join(path, prefix, str(sparse))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    graph_dict = defaultdict(set)
    graph_full_dict = defaultdict(set)
    graph_train = defaultdict(set)
    graph_test = defaultdict(set)
    graph_full_test = defaultdict(set)
    random.seed(seed+20222205)
    np.random.seed(seed+20222205)
    with open(os.path.join(path, factfile), 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            graph_full.append((int(line[0]), int(line[1]), int(line[2])))
            graph_full_dict[(int(line[0]), int(line[1]))].add(int(line[2]))
            if mode == 'edge':
                prob = sparse / 100
            elif mode == 'node':
                prob = node_prob[int(line[0])] * node_prob[int(line[2])]
            else:
                raise ValueError('mode not included!')
            if random.uniform(0,1) < prob: # add to sparse graph
                graph.append((int(line[0]), int(line[1]), int(line[2])))
                graph_dict[(int(line[0]), int(line[1]))].add(int(line[2]))
                # training set or test set
                if random.uniform(0, 1) < training_rate:
                    graph_train[(int(line[0]), int(line[1]))].add(int(line[2]))
                else:
                    graph_test[(int(line[0]), int(line[1]))].add(int(line[2]))
                    graph_full_test[(int(line[0]), int(line[1]))].add(int(line[2]))
            else:
                graph_full_test[(int(line[0]), int(line[1]))].add(int(line[2])) # graph_full_test is (graph - graph_train).
    with open(os.path.join(save_path, 'graph_sparse.pkl') ,'wb') as f:
        pickle.dump(graph, f)
    with open(os.path.join(save_path, 'graph_sparse_dict.pkl'), 'wb') as f:
        pickle.dump(graph_dict, f)
    with open(os.path.join(save_path, 'graph_sparse_train.pkl'), 'wb') as f:
        pickle.dump(graph_train, f)
    with open(os.path.join(save_path, 'graph_sparse_test.pkl'), 'wb') as f:
        pickle.dump(graph_test, f)
    with open(os.path.join(save_path, 'graph_full.pkl') ,'wb') as f:
        pickle.dump(graph_full, f)
    with open(os.path.join(save_path, 'graph_full_dict.pkl'), 'wb') as f:
        pickle.dump(graph_full_dict, f)
    with open(os.path.join(save_path, 'graph_full_test.pkl'), 'wb') as f:
        pickle.dump(graph_full_test, f)
    return graph_train, graph_test, graph_full_test

def generate_queries(N, thre, nentity, nrelation, graph_sparse, graph_full, seed, train):
    n = 0
    random.seed(seed)
    np.random.seed(seed)
    output = defaultdict(set)
    output_full = defaultdict(set)
    output_sparse = defaultdict(set)
    entity_times_rel = [(i,j) for j in range(nrelation) for i in range(nentity)]
    num_train = 0
    if N is None:
        for head_rel in tqdm(entity_times_rel):
            head = head_rel[0]
            rel = head_rel[1]
            if train:
                answers = graph_sparse[(head, rel)]
                if len(answers) >= thre:
                    output[(head, rel)] = answers
                    num_train += 1
            else:
                assert graph_full is not None
                answers_sparse = graph_sparse[(head, rel)]
                answers_full = graph_full[(head, rel)]
                if len(answers_full) >= thre and len(answers_sparse) >= 1:
                    output_full[(head, rel)] = answers_full
                    output_sparse[(head, rel)] = answers_sparse
                    num_train += 1
        if train:
            print('Train queries: {}'.format(num_train))
        else:
            print('Test queries: {}'.format(num_train))
                
    else:
        with tqdm(total=N) as bar:
            while n < N:
                l = len(entity_times_rel)
                if not l:
                    raise ValueError('There are no remaining queries. Please lower the threshold or reduce the number of queries.')
                ind = random.randint(0,l-1)
                head_rel = entity_times_rel.pop(ind)
                head = head_rel[0]
                rel = head_rel[1]
                if train:
                    answers = graph_sparse[(head, rel)]
                    if len(answers) >= thre:
                        output[(head, rel)] = answers
                        n += 1
                        bar.update(1)
                else:
                    assert graph_full is not None
                    answers_sparse = graph_sparse[(head, rel)]
                    answers_full = graph_full[(head, rel)]
                    if len(answers_full) >= thre and len(answers_sparse) >= 1:
                        output_full[(head, rel)] = answers_full
                        output_sparse[(head, rel)] = answers_sparse
                        n += 1
                        bar.update(1)
    if train:
        return output
    else:
        return output_full, output_sparse

def generate_data(graph_sparse_train, graph_apsrse_test, graph_full_test, nentity, nrelation, path, thre, seed):
    train = generate_queries(N=args.train_queries, thre=1, nentity=nentity, nrelation=nrelation, graph_sparse=graph_sparse_train, graph_full=None, seed = seed+512, train=True)
    test_full, test_sparse = generate_queries(N=args.test_queries, thre=thre, nentity=nentity, nrelation=nrelation, graph_sparse=graph_sparse_test, graph_full=graph_full_test, seed = seed+1024, train=False)
    with open(os.path.join(path, 'train-queries.pkl'), 'wb') as f:
        pickle.dump(list(train.keys()), f)
    with open(os.path.join(path, 'test-queries.pkl'), 'wb') as f:
        pickle.dump(list(test_full.keys()), f)
        
        
    with open(os.path.join(path, 'test-easy-answers.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(path, 'test-sparse-hard-answers.pkl'), 'wb') as f:
        pickle.dump(test_sparse, f)
    with open(os.path.join(path, 'test-full-hard-answers.pkl'), 'wb') as f:
        pickle.dump(test_full, f)
    with open(os.path.join(path, 'train-answers.pkl'), 'wb') as f:
        pickle.dump(train, f)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generate Trainging and Test Graph, Queries and Answers',
        usage='data_generate.py [<args>] [-h | --help]'
    )
    parser.add_argument('--sparse', nargs='+', type=int, help='a list whose element should be an int from 0 to 100')
    # with default value
    parser.add_argument('--train_queries', type=int, default=None, help='num of training queries, default: all')
    parser.add_argument('--test_queries', type=int, default=None, help='num of test queries, default: all')
    parser.add_argument('--training_rate', type=float, default=0.7, help='the rate of training graph')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--path', type=str, default='.', help='data path')
    parser.add_argument('--loadname', type=str, default='facts_raw.txt', help='the file of raw data')
    parser.add_argument('--prefix', type=str, default='raw', help='the prefix of saving path')
    parser.add_argument('--thre', type=int, default=10, help='the lower bound of the num of answers for each query in sparse test set')
    parser.add_argument('--mode', type=str, choices=['node', 'edge'], default='edge', help='how to sample in the full graph')

    return parser.parse_args(args)
        
if __name__ == '__main__':
    args = parse_args()
    nentities, nrelations = re_id(os.path.join(args.path,args.loadname), args.path, 'facts.txt')
    sparse_list = args.sparse
    for sparse in sparse_list:
        print('Generating sparse {} data'.format(sparse))
        graph_sparse_train, graph_sparse_test, graph_full_test = generate_graph(args.path, args.prefix, sparse, args.training_rate, args.seed, 'facts.txt', mode=args.mode, nentity=nentities)
        generate_data(graph_sparse_train, graph_sparse_test, graph_full_test, nentities, nrelations, os.path.join(args.path, args.prefix, str(sparse)), args.thre, args.seed)