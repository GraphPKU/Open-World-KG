import argparse
import torch
import pickle
import os
import requests, re
import chardet
import urllib.request as request
import numpy as np
import scipy.stats
import collections
from dataloader import TestDataset
from torch.utils.data import DataLoader
from main import load_data
import pdb
import json
from tqdm import tqdm

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--model_path', type=str, default=None, help="model path")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('--output_file', default='report/output.txt', type=str, help='path of output report')
    parser.add_argument('--tasks', default='1p', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--detail', action='store_true', help='generate detail report')
    return parser.parse_args(args)

def ent2name(ent, ind):
    # given ind in FB, get name in wikidata.
    entity_name = 'Not Find!'
    fbid = ent[ind]
    entity_name = fbid
    assert fbid[2] == '/'
    fbid = '<http://rdf.freebase.com/ns'+fbid[0:2]+'.'+fbid[3:]+'>'
    with open('fb2w.nt','r') as f:
        l = f.readline()
        while l:
            l = f.readline()
            if fbid not in l:
                continue
            wikiurl = str(l.split()[2][1:-1])
            wikiurl = wikiurl[0:24]+'wiki'+wikiurl[30:]
            try:
                entity_name = get_title(wikiurl)[:-11]
            except:
                pass
            return entity_name
        return entity_name
    
def rel2name(rel, ind):
    return rel[ind]

def get_title(url):
    # give url, get title.
    s = requests.session()
    response = request.urlopen(url)
    html = response.read()
    charset = chardet.detect(html)['encoding']
    result = s.get(url)
    if (charset == "GB2312" or charset is None):
        result.encoding = 'gbk'
    else:
        result.encoding = 'utf-8'
    content = result.text
    title = re.findall('<title>(.*)</title>', content)[0]
    return title

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, graph, graph_valid, detail=False, metric=None, ent=None, rel=None):
    
    """
    detail: if True, report the detailed report on each test query.
    """
    # if metric is None:
    #     assert not detail 
    model.eval()
    logs = collections.defaultdict(list)
    step = 0

    total_steps = len(test_dataloader) # num of queries
    if detail:
        items = 0
        task = ('e', ('r',))
        # mrr = metric[task]['MRR']
        # h1 = metric[task]['HITS1']
        # h3 = metric[task]['HITS3']
        # h10 = metric[task]['HITS10']
        # nrr_nm = metric[task]['MRR_nonmasked']
        # h1_nm = metric[task]['HITS1_nonmasked']
        # h3_nm = metric[task]['HITS3_nonmasked']
        # h10_nm = metric[task]['HITS10_nonmasked']
        with open(args.output_file,'w') as f:
            f.write('data path: {}\nmodel path: {}\n'.format(args.data_path, args.model_path))
            # f.write('MRR: {:.4f}, H1:{:.4f}, H3:{:.4f}, H10:{:.4f}\n'.format(mrr, h1, h3, h10))
            # f.write('Non-maksed MRR: {:.4f}, H1:{:.4f}, H3:{:.4f}, H10:{:.4f}\n\n'.format(nrr_nm, h1_nm, h3_nm, h10_nm))
            f.write('wr: not in answer, ie: included in easy answer, ih: included in hard answer, ??:others.\n\n')
    with torch.no_grad():
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader):
            # negative_sample is a tensor [0,1,...,nentity-1], which means all entities.
            batch_queries_dict = collections.defaultdict(list)
            batch_idxs_dict = collections.defaultdict(list)
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                if args.cuda:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                else:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
            if args.cuda:
                negative_sample = negative_sample.cuda()

            _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [query_structures[i] for i in idxs]
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            # print(step)
            ranking = argsort.clone().to(torch.float)
            if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
            else: # otherwise, create a new torch Tensor for batch_entity_range
                if args.cuda:
                    ranking = ranking.scatter_(1, 
                                               argsort, 
                                               torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                  1).cuda()
                                               ) # achieve the ranking of all entities
                else:
                    ranking = ranking.scatter_(1, 
                                               argsort, 
                                               torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                  1)
                                               ) # achieve the ranking of all entities
            for idx, (_, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                hard_answer = hard_answers[query]
                easy_answer = easy_answers[query]
                num_hard = len(hard_answer)
                num_easy = len(easy_answer)
                assert len(hard_answer.intersection(easy_answer)) == 0
                cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)] # orders of all known answers
                cur_ranking, indices = torch.sort(cur_ranking) # cur_ranking: orders of all known answers (from small to large), indices: the i-th answer
                masks = indices >= num_easy
                if args.cuda:
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                else:
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                cur_ranking_r = cur_ranking - answer_list + 1 # filtered setting
                cur_ranking_r_nonmasked = cur_ranking_r
                cur_ranking_r = cur_ranking_r[masks] # only take indices that belong to the hard answers                
            
                mrr = torch.mean(1./cur_ranking_r).item()
                h1 = torch.mean((cur_ranking_r <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking_r <= 3).to(torch.float)).item()
                h10 = torch.mean((cur_ranking_r <= 10).to(torch.float)).item()
                
                mrr_nm = torch.mean(1./cur_ranking_r_nonmasked).item() # nm means non-masekd
                h1_nm = torch.mean((cur_ranking_r_nonmasked <= 1).to(torch.float)).item()
                h3_nm = torch.mean((cur_ranking_r_nonmasked <= 3).to(torch.float)).item()
                h10_nm = torch.mean((cur_ranking_r_nonmasked <= 10).to(torch.float)).item()
                
                if detail:
                    items += 1
                    cur_ranking = cur_ranking.cpu()
                    indices = indices.cpu()
                    cur_ranking_r = cur_ranking_r.cpu()
                    query_pos = (query[0], query[1][0])
                    easy_answer_order = [int(cur_ranking[masks.logical_not()][i]) for i in range(len(cur_ranking[masks.logical_not()]))]
                    # prediction output
                    show_num = int(max(min(200, cur_ranking[masks][min(4,len(cur_ranking[masks])-1)].item()), 30))
                    show_list = argsort[idx, 0:show_num].tolist()
                    show_cur = [0,0,0,0]
                    for i, ent_ind in enumerate(show_list):
                        if ent_ind in easy_answer:
                            show_list[i] = (i, ent_ind, 'ie')
                            show_cur[0] += 1
                        elif ent_ind in hard_answer:
                            show_list[i] = (i, ent_ind, 'ih')
                            show_cur[1] += 1
                        elif ent_ind not in graph[(query_pos[0],query_pos[1])]:
                            show_list[i] = (i, ent_ind, 'wr')
                            show_cur[2] += 1
                        else:
                            show_list[i] = (i, ent_ind, '?? '+ent2name(ent, ent_ind))
                            show_cur[3] += 1
                    # hard answer order
                    hard_answer_order = [(int(cur_ranking_r[i]), int(cur_ranking[masks][i]),
                              (list(easy_answer)+list(hard_answer))[indices[masks][i]]) for i in range(len(cur_ranking[masks]))]
                    total_answer_order = [(int(cur_ranking[i]), (list(easy_answer)+list(hard_answer))[indices[i]]) for i in range(len(cur_ranking))]
                            
                    with open(args.output_file, 'a') as f:
                        f.write('Query {}: {}\n'.format(items, query))
                        f.write('({}, {}, {})\n'.format(ent2name(ent, query_pos[0]), rel2name(rel, query_pos[1]),
                               len(graph[tuple(query_pos)])
                               ))
                        f.write('Non-masked MRR:{:.2%} H1:{:.2%} H3:{:.2%} H10:{:.2%}\n\n'.format(mrr_nm,h1_nm,h3_nm,h10_nm))
                        f.write('MRR:{:.2%} H1:{:.2%} H3:{:.2%} H10:{:.2%}\n\n'.format(mrr,h1,h3,h10))
                        f.write('Orders of all answers (total: {}, ave: {:.2f}, har:{:.2f}): {}\n\n'.format(len(cur_ranking), np.average(cur_ranking),
                                                                                                            scipy.stats.mstats.hmean(cur_ranking), total_answer_order))
                        f.write('Orders of hard answers (total: {}, ave: {:.2f}, har:{:.2f}): {}\n\n'.format(len(cur_ranking[masks]),
                                                                                                             np.average(cur_ranking[masks]),
                                                                                                             scipy.stats.mstats.hmean(cur_ranking[masks]),
                                                                                                             hard_answer_order))
                        f.write('Easy answer order (total: {}, ave: {:.2f}, har:{:.2f}): {}\n\n'.format(len(easy_answer_order),np.average(easy_answer_order),
                                                                                                        scipy.stats.mstats.hmean(easy_answer_order),
                                                                                                        easy_answer_order))
                        f.write('The first {} prediction: {}\n\n'.format(show_num, show_list))
                        f.write('ie: {}, ih: {}, wr:{}, ??:{}\n\n'.format(show_cur[0],show_cur[1],show_cur[2],show_cur[3]))
                        f.write('#'*90+'\n\n')
                

                logs[query_structure].append({
                    'MRR': mrr,
                    'HITS1': h1,
                    'HITS3': h3,
                    'HITS10': h10,
                    'MRR_nonmasked': mrr_nm,
                    'HITS1_nonmasked': h1_nm,
                    'HITS3_nonmasked': h3_nm,
                    'HITS10_nonmasked': h10_nm,
                    'num_hard_answer': num_hard,
                })
                
            step += 1

            
    metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    for query_structure in logs:
        for metric in logs[query_structure][0].keys():
            if metric in ['num_hard_answer']:
                continue
            metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
        metrics[query_structure]['num_queries'] = len(logs[query_structure])

    return metrics


def main(args):
    with open(os.path.join(args.data_path, 'stats.txt')) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1]) 
        print('nentity: {} nrelation {}'.format(nentity, nrelation))
        
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        json_args = json.load(f)
    args.do_valid = False
    args.do_train = False
    for name in json_args:
        if not hasattr(args, name):
            setattr(args, name, json_args[name])
            
    
    
    from models import KGReasoning
    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma, # margin of loss
        geo=args.geo, # query2box
        args=args,
        use_cuda = args.cuda,
        box_mode=eval_tuple(args.box_mode), # eval_tuple: str2tuple
        beta_mode = eval_tuple(args.beta_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict = query_name_dict
    )
    print('Loading model...')
    checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if args.cuda:
        model = model.cuda()
    if args.detail:
        id2ent_file = os.path.join(args.data_path, 'id2ent.pkl')
        id2rel_file = os.path.join(args.data_path, 'id2rel.pkl')
        with open(id2ent_file, 'rb') as f:
            ent = pickle.load(f)
        with open(id2rel_file, 'rb') as f:
            rel = pickle.load(f)
    
    
    print('Loading data...')
    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data(args, args.tasks)
    test_queries = flatten_query(test_queries) # from a dict to a list

    test_dataloader = DataLoader(
        TestDataset(
            test_queries, 
            nentity, 
            nrelation), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn
    )


    with open(os.path.join(args.data_path, 'graph.pkl'),'rb') as f:
        graph = pickle.load(f)
    with open(os.path.join(args.data_path, 'graph(train+valid).pkl'),'rb') as f:
        graph_valid = pickle.load(f)
    with open(os.path.join(args.data_path, 'graph(train).pkl'),'rb') as f:
        graph_train = pickle.load(f)
    print('Start testing!')
    # metric = test_step(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, graph, graph_valid, detail=False, metric=None, ent=None, rel=None)
    if not args.detail:
        print(metric[('e', ('r',))])
    if args.detail:
        print('Start generating report!')
        # test_step(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, graph, graph_valid, detail=True, metric=metric, ent=ent, rel=rel)
        test_step(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, graph, graph_valid, detail=True, metric=None, ent=ent, rel=rel)
        
if __name__ == '__main__':
    main(parse_args())