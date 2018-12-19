import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy
import copy
import os
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from utils import Str2id
from collections import Counter
from utils import NodeClassifier, get_word_embeddings
from tqdm import tqdm_notebook as tqdm
from tqdm import tnrange
from datasets import NormalEdgeDataset, NodeTextEdgeDataset, MultipleLableDataset


class SemiSkipGram(nn.Module):
    def __init__(self, emb_dim, label_dim, use_cuda):
        super(SemiSkipGram, self).__init__()
        self.use_cuda = use_cuda
        self.ranking = nn.Sequential(nn.Linear(emb_dim * 2, label_dim))
        self.criteria = nn.BCEWithLogitsLoss(size_average=False)

    def stucture_loss(self, emb_u, emb_v, neg_emb_v):
        losses = []
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze().view(-1)
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def edge_label_loss(self, emb_u, emb_v, labels):
        embeddings = torch.cat((emb_u, emb_v), dim=1)
        ranking = self.ranking(embeddings)
        loss = self.criteria(ranking, labels)
        return loss

    def predicate_edge_label(self, emb_u, emb_v):
        embeddings = torch.cat((emb_u, emb_v), dim=1)
        ranking = self.ranking(embeddings)
        return ranking


class NodeRepresentation(nn.Module):
    def __init__(self, emb_size, emb_dim, use_cuda):
        super(NodeRepresentation, self).__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.u_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.init_emb()
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def init_node_embedding(self, node_embs):
        if self.use_cuda:
            self.v_embeddings.weight.data = torch.from_numpy(
                node_embs[:, self.emb_dim:]
            ).cuda()
        else:
            self.u_embeddings.weight.data = torch.from_numpy(
                node_embs[:, :self.emb_dim]
            )
            self.v_embeddings.weight.data = torch.from_numpy(
                node_embs[:, self.emb_dim:]
            )

    def init_emb(self):
        initrange = 0.5 / self.emb_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, nids, is_start, directed):
        losses = []
        if self.use_cuda:
            nids = Variable(torch.cuda.LongTensor(nids))
        else:
            nids = Variable(torch.LongTensor(nids))
        if directed:
            if is_start:
                return self.u_embeddings(nids)
            else:
                return self.v_embeddings(nids)
        else:
            return torch.cat(
                (self.u_embeddings(nids), self.v_embeddings(nids)), dim=1
            )

    def save_embedding(self, node_map, file_name, use_cuda):
        embeddings = self.u_embeddings.weight
        embeddings = embeddings.cpu().data.numpy()
        fout = open(file_name, 'w')
        fout.write("%d %d\n" % (self.emb_size, self.emb_dim))
        for wid in range(self.emb_size):
            e = embeddings[wid]
            w = node_map.id2str(wid)
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
