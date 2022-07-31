import os
import json
import gzip
import numpy as np
from tqdm import tqdm


def average_precision(truth, pred):
    """
        Computes the average precision.
        
        This function computes the average precision at k between two lists of items.

        Parameters
        ----------
        truth: set
                    A set of ground-truth elements (order doesn't matter)
        pred: list
                    A list of predicted elements (order does matter)
        Returns
        -------
        score: double
                    The average precision over the input lists 
    """
    if not truth:
        return 0.0
    
    score, hits_num = 0.0, 0
    for idx, doc in enumerate(pred):
        if doc in truth and doc not in pred[:idx]:
            hits_num += 1.0
            score += hits_num / (idx + 1.0)
    return score / max(1.0, len(truth))


def recall(truth, pred):
    if not truth:
        return 0.0
    
    score, hits_num = 0.0, 0
    for idx, doc in enumerate(pred):
        if doc in truth:
            return 1.0
    return 0.0


def NDCG(truth, pred, use_graded_scores=False):
    score = 0.0
    for rank, item in enumerate(pred):
        if item in truth:
            if use_graded_scores:
                grade = 1.0 / (truth.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)
    
    norm = 0.0
    for rank in range(len(truth)):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)


def metrics(truth, pred, metrics_map):
    """
        Return a numpy array containing metrics specified by metrics_map.
        truth: set
                    A set of ground-truth elements (order doesn't matter)
        pred: list
                    A list of predicted elements (order does matter)
    """
    out = np.zeros((len(metrics_map),), np.float32)

    if "MAP@20" in metrics_map:
        avg_precision = average_precision(truth, pred[:20])
        out[metrics_map.index('MAP@20')] = avg_precision
    
    if "P@1" in metrics_map:
        intersec = len(truth & set(pred[:1]))
        out[metrics_map.index('P@1')] = intersec / max(1., float(len(pred[:1])))

    if "P@10" in metrics_map:
        intersec = len(truth & set(pred[:10]))
        out[metrics_map.index('P@10')] = intersec / max(1., float(len(pred[:10])))
    
    if "P@20" in metrics_map:
        intersec = len(truth & set(pred[:20]))
        out[metrics_map.index('P@20')] = intersec / max(1., float(len(pred[:20])))

    if "P@100" in metrics_map:
        intersec = len(truth & set(pred[:100]))
        out[metrics_map.index('P@100')] = intersec / max(1., float(len(pred[:100])))

    if "R@1" in metrics_map:
        res = recall(truth, pred[:1])
        out[metrics_map.index('R@1')] = res
    
    if "R@10" in metrics_map:
        res = recall(truth, pred[:10])
        out[metrics_map.index('R@10')] = res
    
    if "R@100" in metrics_map:
        res = recall(truth, pred[:100])
        out[metrics_map.index('R@100')] = res

    if "R@1000" in metrics_map:
        res = recall(truth, pred[:1000])
        out[metrics_map.index('R@1000')] = res
    
    if "MRR" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR')] = score
        
    if "MRR@10" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:10]):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR@10')] = score
   
    if "MRR@100" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:100]):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR@100')] = score
    
    if "NDCG@10" in metrics_map:
         out[metrics_map.index('NDCG@10')] = NDCG(truth, pred[:10])

    if "NDCG@20" in metrics_map:
         out[metrics_map.index('NDCG@20')] = NDCG(truth, pred[:20])

    if "NDCG@100" in metrics_map:
         out[metrics_map.index('NDCG@100')] = NDCG(truth, pred[:100])
    
    return out
class evaluator:
    def __init__(self):
        self.METRICS_MAP = ['MRR@10', 'MRR', 'NDCG@10', 'NDCG@20', 'NDCG@100', 'MAP@20', 'P@1', 'P@10', 'P@20', 'P@100', 'R@1', 'R@10', 'R@100', 'R@1000']
    
    def evaluate_ranking(self, docid_truth, all_doc_probs, doc_idxs=None, query_ids=None, match_scores=None):
        map_list = []
        mrr_list, mrr_10_list = [], []
        ndcg_10_list, ndcg_20_list, ndcg_100_list = [], [], []
        p_1_list, p_10_list, p_20_list ,p_100_list = [], [], [], []
        r_1_list, r_10_list, r_100_list, r_1000_list = [], [], [], []

        for docid, probability in tqdm(zip(docid_truth, all_doc_probs)):
            click_doc = set(docid)
            sorted_docs = probability
            _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r10, _r100, _r1000 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)
                
            mrr_10_list.append(_mrr10)
            mrr_list.append(_mrr)

            ndcg_10_list.append(_ndcg10)
            ndcg_20_list.append(_ndcg20)
            ndcg_100_list.append(_ndcg100)
                
            p_1_list.append(_p1)
            p_10_list.append(_p10)
            p_20_list.append(_p20)
            p_100_list.append(_p100)

            r_1_list.append(_r1)
            r_10_list.append(_r10)
            r_100_list.append(_r100)
            r_1000_list.append(_r1000)

            map_list.append(_map20)

        return [np.mean(mrr_10_list), np.mean(mrr_list), np.mean(ndcg_10_list), np.mean(ndcg_20_list), np.mean(ndcg_100_list), np.mean(map_list), np.mean(p_1_list), np.mean(p_10_list), np.mean(p_20_list), np.mean(p_100_list), np.mean(r_1_list), np.mean(r_10_list), np.mean(r_100_list), np.mean(r_1000_list)]