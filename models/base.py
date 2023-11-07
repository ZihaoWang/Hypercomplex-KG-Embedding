"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod
import os
import torch as T
import numpy as np
from torch import nn
from datasets.kg_dataset import KGDataset
from collections import defaultdict


class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        gamma: T.nn.Parameter for margin in ranking-based loss
        data_type: T.dtype for machine precision (single or double)
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: T.nn.Embedding with entity embeddings
        rel: T.nn.Embedding with relation embeddings
        bh: T.nn.Embedding with head entity bias embeddings
        bt: T.nn.Embedding with tail entity bias embeddings
    """

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size, args):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        if data_type == 'double':
            self.data_type = T.double
        else:
            self.data_type = T.float
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(T.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)
        self.rel = nn.Embedding(sizes[1], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = T.zeros((sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = T.zeros((sizes[0], 1), dtype=self.data_type)
        device = "cuda:"+str(args.cuda_n) if args.cuda_n >= 0 else "cpu"
        self.cuda1 = T.device(device)


        self.storageTriple = T.tensor([]).to(self.cuda1).to(T.long)
        self.storageEntId = T.tensor([]).to(self.cuda1).to(T.long)
        self.storageallscores = T.tensor([]).to(self.cuda1)
        self.storagecurrentcores = T.tensor([]).to(self.cuda1)
        self.storagerank = T.tensor([])
        self.lm = args.lm
        if self.lm == True:
            self.entity_lm = nn.Embedding(self.sizes[0], rank).requires_grad_(False) #entity embedding
            self.rel_lm = nn.Embedding(self.sizes[1], rank).requires_grad_(False) #relation embedding
            
        #self.data_type = args.data_type

    @abstractmethod
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: T.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: T.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: T.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities.

        Args:
            queries: T.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
             rhs_e: T.Tensor with targets' embeddings
                    if eval_mode=False returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: T.Tensor with targets' biases
                         if eval_mode=False returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: T.Tensor with queries' embeddings
            rhs_e: T.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: T.Tensor with similarity scores of queries against targets
        """
        pass

    def score(self, lhs, rhs, eval_mode):
        """Scores queries against targets

        Args:
            lhs: Tuple[T.Tensor, T.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[T.Tensor, T.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: T.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: T.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[T.Tensor, T.Tensor, T.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: T.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: T.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        # get factors for regularization
        factors = self.get_factors(queries)
        #print(len(factors))
        #exit()
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: T.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: T.Tensor with ranks or correct entities
        """
        ranks = T.ones(len(queries), dtype=self.data_type)
        with T.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)
            j = 0
            while b_begin < len(queries):
                j = j + 1
                these_queries = queries[b_begin:b_begin + batch_size].to(self.cuda1)

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)
                
                for i, query in enumerate(these_queries):                   
                    if query.size()[0] == 5:
                       filter_out = filters[(query[0].item(), query[1].item(), query[3].item(), query[4].item())]
                    elif query.size()[0] == 4:
                       filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                    else:
                       filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    if query[0].item() != query[2].item():
                       filter_out += [queries[b_begin + i, 0].item()]
                    scores[i, T.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += T.sum(
                    (scores >= targets).type(self.data_type), dim=1
                ).cpu()
                b_begin += batch_size

                if j < 0:
                   self.storageTriple = T.cat((self.storageTriple, these_queries), 0)
                   self.storageallscores = T.cat((self.storageallscores, scores), 0)
                   self.storagecurrentcores = T.cat((self.storagecurrentcores,targets),0)
                   self.storagerank = T.cat((self.storagerank,ranks),0)

        #np.savetxt('my_file.txt', self.storageallscores.cpu().numpy())
        ##i=0

        ##a, indx = T.topk(self.storageallscores, 10, dim = 1)
        #oo = these_queries[indx]
        ##indx = indx.cpu().tolist()
        ##a = a.cpu().tolist()
        #print(indx.size())
        #print(a.size())
        #exit()
        ##b = self.storageTriple.cpu().tolist()
        ##c = self.storagecurrentcores.cpu().tolist()
        ##d = self.storagerank.cpu().tolist()
        ##with open('score.txt', 'w') as filehandle:
        ##    for listitem in a:
        ##        ids = indx[i]
        ##        listtriple = b[i]
        ##        listitems = set(listitem)
        ##        listcurrent = c[i]
        ##        listrank = d[i]
        ##        num = len(listitem) - len(listitems)
        ##        ##print(num)
        ##        i = i + 1
        ##        ##print(i)
        ##        ##filehandle.write('%s\n'% num)
        ##        ##filehandle.write('%s\n\n\n'% listtriple)
        ##        ##filehandle.write('%s\n'% listcurrent)
        ##        ##filehandle.write('rank is: \t %s\n\n\n'% listrank)
        ##        ##filehandle.write('TopK Scores:\t %s\n\n\n\t'% listitem)
                ##filehandle.write('TopK index:\t %s\n\n\n'% ids)
                #filehandle.write('%s\n\n\n'% listitems)
        #exit()
        return ranks

    def compute_metrics(self, examples, filters, batch_size=500, remove_rel = False):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: T.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}
        #print(examples.size()[1])
        #exit()

        class RelTuple(object):
            def __init__(self):
                self.rhs_tuple = []
                self.lhs_tuple = []

            def make_tensor(self):
                if len(self.rhs_tuple) == 1:
                    self.rhs_tuple = self.rhs_tuple[0].unsqueeze(0)
                else:
                    self.rhs_tuple = T.stack(self.rhs_tuple, 0)
                if len(self.lhs_tuple) == 1:
                    self.lhs_tuple = self.lhs_tuple[0].unsqueeze(0)
                else:
                    self.lhs_tuple = T.stack(self.lhs_tuple, 0)

                return self.rhs_tuple, self.lhs_tuple

        if remove_rel:
            HITS_FOR_TRIMMING = 10
            TRIMMING_PERCENT = 0.36
            q_rel = defaultdict(RelTuple)
            q_hits = []
            final_l_tuple = []
            final_r_tuple = []

            for i in range(examples.shape[0]):
                rel = examples[i, 1].item()
                r_tuple = T.clone(examples[i])
                l_tuple = T.clone(examples[i])
                l_tuple[0] = r_tuple[2]
                l_tuple[1] += self.sizes[1] // 2
                l_tuple[2] = r_tuple[0]
                q_rel[rel].rhs_tuple.append(r_tuple)
                q_rel[rel].lhs_tuple.append(l_tuple)

            for k in q_rel:
                rhs_tuple, lhs_tuple = q_rel[k].make_tensor()

                try:
                    r_ranks = self.get_ranking(rhs_tuple, filters["rhs"], batch_size=batch_size)
                    l_ranks = self.get_ranking(lhs_tuple, filters["lhs"], batch_size=batch_size)
                except:
                    print(q_rel[k])
                    print(m)
                    raise
                r_hits = T.mean((r_ranks <= HITS_FOR_TRIMMING).float()).item()
                l_hits = T.mean((l_ranks <= HITS_FOR_TRIMMING).float()).item()
                hits = (r_hits + l_hits) * 0.5
                q_hits.append((k, round(hits, 3), rhs_tuple.shape[0] + lhs_tuple.shape[0]))

                if hits > TRIMMING_PERCENT:
                    v = q_rel[k]
                    final_l_tuple.append(v.lhs_tuple)
                    final_r_tuple.append(v.rhs_tuple)

            q_hits = sorted(q_hits, key = lambda x: x[1], reverse = True)
            for i in range(len(q_hits)):
                if q_hits[i][1] < TRIMMING_PERCENT:
                    q_hits = q_hits[:i]
                    break

            final_l_tuple = T.cat(final_l_tuple, 0)
            final_r_tuple = T.cat(final_r_tuple, 0)
            print("before trimming #rel = {}, after #rel = {}, hits_for_trimming = {}, trimming_precent = {}".format(len(q_rel) / 2, len(final_l_tuple), len(final_r_tuple), HITS_FOR_TRIMMING, TRIMMING_PERCENT))
            print("q_hits@{} = {}".format(HITS_FOR_TRIMMING, q_hits))

            for m, q in zip(("rhs", "lhs"), (final_r_tuple, final_l_tuple)):
                ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
                mean_rank[m] = T.mean(ranks).item()
                mean_reciprocal_rank[m] = T.mean(1. / ranks).item()
                hits_at[m] = T.FloatTensor((list(map(lambda x: T.mean((ranks <= x).float()).item(), (1, 3, 10)))))
        else:
            for m in ["rhs", "lhs"]:
                q = examples.clone()            
                if m == "lhs":
                    tmp = T.clone(q[:, 0])
                    q[:, 0] = q[:, 2]
                    q[:, 2] = tmp
                    q[:, 1] += self.sizes[1] // 2
                ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
                mean_rank[m] = T.mean(ranks).item()
                mean_reciprocal_rank[m] = T.mean(1. / ranks).item()
                hits_at[m] = T.FloatTensor((list(map(lambda x: T.mean((ranks <= x).float()).item(), (1, 3, 10)))))

        return mean_rank, mean_reciprocal_rank, hits_at
