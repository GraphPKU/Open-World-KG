import numpy as np
import random
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import pickle

def seed(seed):
    random.seed(seed)
    np.random.seed(seed) 

def sample_generate(answer_rate=0.01, density=0.9, nentity=14505):
    tp_fn = int(nentity * answer_rate)
    tn = nentity - tp_fn # true negative
    tp = np.random.binomial(tp_fn, density)
    fn = tp_fn - tp
    sample_tp=sorted(random.sample(range(nentity), k=tp))
    sample_fn=sorted(random.sample(set(range(nentity)).difference(set(sample_tp)), k=fn))
    sample_tn=sorted(list(set(range(nentity)).difference(sample_tp).difference(sample_fn)))
    return sample_tp, sample_fn, sample_tn

def test(nentity, sample_tp, sample_fn, sample_tn, strength, cor, alpha, method='mrr'):
    result = []
    l_missing = strength + np.sqrt(np.maximum(strength * (1-strength) * alpha / (1-alpha), 0)) * cor 
            # print(l_missing)
    l_test = strength - np.sqrt(np.maximum(strength * (1-strength) * (1-alpha) / alpha, 0)) * cor
    for i in range(nentity):
        if i in sample_tp:
            if random.uniform(0, 1) < l_test:
                result.insert(0, i)
            else:
                result.append(i)
        elif i in sample_fn:
            if random.uniform(0, 1) < l_missing:
                result.insert(0, i)
            else:
                result.append(i)
        else:
            result.append(i)
    mask = [i in sample_tp for i in result]
    mask = np.insert(np.cumsum(mask), 0, 0)
    sort = np.argsort(result)
    sort = sort - mask[sort]
    sort = sort[sample_tp] + 1
    if method == 'mrr':
        return np.average(1/sort)
    elif method == 'ndcg':
        return np.average(1/np.log(1+sort))*np.log(2)
    elif method == '-1/2':
        return np.average(1/np.sqrt(sort))
    elif method == 'ndcg@5':
        mask = sort > 5
        ndcg = 1/np.log(1+sort)*np.log(2)
        ndcg[mask] = 0
        return np.average(ndcg)
    elif method == 'hit@5':
        mask = sort <= 5
        return np.average(mask)
    
if __name__ == '__main__':
    seed(20222199)
    nentity = 14505
    answer_rate = 0.01 * 0.3
    A = int(nentity * answer_rate)
    times = 500
    learning_success_rate_p_list = list(np.arange(0.3,1.01,.01))
    density_list = [0.8, 0.65, 0.5, 0.35]
    cor_list = [-0.3,-0.2,-0.1,0,0.1,0.2]
    Eular = 0.577215665
    draw = []
    savedict = dict()
    for cor in tqdm(cor_list):
        plt.figure()
        savedict[cor] = dict()
        for alpha in tqdm(density_list):
            MRR = []
            MRR_std = []
            for learning_success_rate_p in tqdm(learning_success_rate_p_list):
                c = []
                for _ in range(times):
                    stp, sfn, stn = sample_generate(answer_rate, alpha, nentity)
                    c.append(test(nentity, stp, sfn, stn, learning_success_rate_p, cor, alpha, method='mrr'))
                MRR.append(np.mean(c))
                MRR_std.append(np.std(c)/np.sqrt(times))
            savedict[cor][alpha] = {'MRR': MRR, 'MRR_std': MRR_std}
            draw.append(plt.plot(learning_success_rate_p_list, MRR, label=str(alpha))[0]) # expectation
            # draw.append(plt.plot(learning_success_rate_p_list, np.array(MRR_std)**2, label=str(alpha))[0]) # plot variance
            # plt.fill_between(learning_success_rate_p_list, np.array(MRR) - 2* np.array(MRR_std), np.array(MRR) + 2 * np.array(MRR_std), alpha=0.3) # fill standard deviation
            # N = int(answer_rate * alpha * nentity)
            # M = int(answer_rate * nentity) - N
            l = np.array(learning_success_rate_p_list)
            l_missing = l + np.sqrt(np.maximum(l * (1-l) * alpha / (1-alpha), 0)) * cor 
            # print(l_missing)
            l_test = l - np.sqrt(np.maximum(l * (1-l) * (1-alpha) / alpha, 0)) * cor
            analy = l_test / l_missing * (np.log(np.maximum(l_missing * (1-alpha) * (A+2), 0)) + Eular)/((A+1)*(1-alpha))
            draw.append(plt.plot(learning_success_rate_p_list, analy, label=str(alpha)+'_')[0]) # analytic expecation
            plt.fill_between(learning_success_rate_p_list, analy - 2* np.array(MRR_std), analy + 2 * np.array(MRR_std), alpha=0.3) # fill standard deviation
            # draw.append(plt.plot(learning_success_rate_p_list, (1-l)/(l*N)*((np.log(l*(M+2))+Eular)/(M+1))**2+3/(8*(M+1)*N-19/(16*l*(M+1)*(M+2))), label=str(alpha)+'_')[0]) # analytic lower bound of variance
        density_list_ = [str(i)+'a' for i in density_list]
        density_list__ = [str(i)+'n' for i in density_list]
        from itertools import chain
        density_list___=list(chain.from_iterable(zip(density_list, density_list_)))
        plt.legend(draw, density_list___ ,title="density alpha", fontsize=10, title_fontsize=11, labelspacing=0.2)
        # plt.title('Approximation and Simulation of $E(MRR)$ \nwith Correlation Coefficient $r={}$'.format(cor))
        plt.title('Approximation and Simulation of $E(MRR)$ \nwith Correlation Coefficient $r={}$'.format(cor))
        plt.xlabel('strength $l$')
        plt.ylim([0.05,0.5])
        plt.ylabel('MRR')
        # plt.ylabel('variance of metric')
        f = plt.gcf()
        f.savefig('./other_experiments/approx_and_simu_cor_{}.pdf'.format(cor), bbox_inches='tight')
    with open('./other_experiments/simu.pkl', 'wb') as f:
        pickle.dump(savedict, f)