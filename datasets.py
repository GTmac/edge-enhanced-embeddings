import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy
import copy
import matplotlib
import os
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn.functional as F
sns.set(style="ticks")
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from utils import Str2id
from collections import Counter
from utils import NodeClassifier
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm
from tqdm import tnrange


class NormalEdgeDataset(Dataset):
    """Edge dataset."""

    def __init__(
        self,
        node_map1,
        node_map2,
        filename,
        neg_sampling_size,
        from_random_walk=False,
        window_size=None
    ):
        """
        Args:
            edge_list (list): A list containts all edges. Each edge is a tuple
        """
        if from_random_walk:
            assert window_size is not None
            self.random_walk_to_edge_list(
                node_map1, node_map2, filename, neg_sampling_size, window_size
            )
        else:
            self.init_from_edge_list_file(
                node_map1, node_map2, filename, neg_sampling_size
            )

    def random_walk_to_edge_list(
        self, node_map1, node_map2, in_file_path, neg_sampling_size, window_size
    ):
        self.edge_list = []
        self.node_list = []
        self.graph = []
        for line in tqdm(open(in_file_path)):
            line = line.strip().split(' ')
            n = len(line)
            for i in range(len(line)):
                left = max(0, i - window_size)
                right = min(i + window_size, n)
                for j in range(left, right):
                    if j == i:
                        continue
                    u = node_map1.str2id(line[i])
                    v = node_map2.str2id(line[j])
                    if u >= len(self.graph):
                        for _ in range(u - len(self.graph) + 1):
                            self.graph.append(set())
                    self.graph[u].add(v)
                    self.edge_list.append([u, v, []])
                    self.node_list.append(v)
        self.init_sample_table()
        self.neg_sampling_size = neg_sampling_size

    def init_from_edge_list_file(
        self, node_map1, node_map2, edge_list_file, neg_sampling_size
    ):
        self.edge_list = []
        self.node_list = []
        self.graph = []
        for line in tqdm(open(edge_list_file)):
            line = line.strip().split(' ')
            line[0] = node_map1.str2id(line[0])
            line[1] = node_map2.str2id(line[1])
            if line[0] >= len(self.graph):
                for _ in range(line[0] - len(self.graph) + 1):
                    self.graph.append(set())
            self.graph[line[0]].add(line[1])
            self.edge_list.append([line[0], line[1], []])
            self.node_list.append(line[1])
        self.init_sample_table()
        self.neg_sampling_size = neg_sampling_size

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        node_frequency = list(Counter(self.node_list).items())
        nids = [f[0] for f in node_frequency]
        values = [f[1] for f in node_frequency]
        pow_frequency = numpy.array(values)**0.75
        nodes_pow = numpy.sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = numpy.round(ratio * sample_table_size)
        for index, c in enumerate(count):
            self.sample_table += [nids[index]] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    def negative_sampling1(self):
        #numpy.random.seed(41)
        neg = numpy.random.choice(
            self.sample_table,
            size=(len(self.edge_list), self.neg_sampling_size)
        )
        #         neg=numpy.random.choice(list(set(self.node_list)),size=(len(self.edge_list),self.neg_sampling_size))
        for i in range(len(self.edge_list)):
            self.edge_list[i][2] = list(neg[i])

    def negative_sampling2(self):
        #numpy.random.seed(41)
        for i in range(len(self.edge_list)):
            neg = numpy.random.choice(
                self.sample_table, size=(self.neg_sampling_size)
            )
            self.edge_list[i][2] = list(neg)

    def negative_sampling(self):
        #numpy.random.seed(41)
        negs = numpy.random.choice(
            self.sample_table,
            size=(len(self.edge_list), self.neg_sampling_size)
        )
        for i in tqdm(range(len(self.edge_list))):
            u = self.edge_list[i][0]
            v = self.edge_list[i][1]
            neg = list(filter(lambda x: x not in self.graph[u], negs[i]))
            while len(neg) < self.neg_sampling_size:
                nid = numpy.random.choice(self.sample_table)
                if nid in self.graph[u]:
                    continue
                neg.append(nid)
            self.edge_list[i][2] = neg

    def __len__(self):
        return len(self.edge_list)

    def __getitem__(self, idx):
        return self.edge_list[idx]


class NodeTextEdgeDataset(Dataset):
    """Edge dataset."""

    def __init__(
        self, node_map1, node_map2, word_map, edge_list_file, text_file,
        neg_sampling_size
    ):
        """
        Args:
            edge_list (list): A list containts all edges. Each edge is a tuple
        """
        self.edge_list = []
        self.node_list = set()
        self.text_list = dict()
        for line in open(edge_list_file):
            line = line.strip().split(' ')
            line[0] = node_map1.str2id(line[0])
            line[1] = node_map2.str2id(line[1])
            self.edge_list.append([line[0], line[1], []])
            self.node_list.add(line[1])
        self.node_list = list(self.node_list)
        for line in open(text_file):
            line = line.replace('||||', ' ').strip().split(' ')
            self.text_list[node_map2.str2id(line[0])] = [
                word_map.str2id(w) for w in line[1:]
            ]
        self.neg_sampling_size = neg_sampling_size
        self.negative_sampling()

    def negative_sampling(self):
        numpy.random.seed(41)
        neg = numpy.random.choice(
            list(self.node_list),
            size=(len(self.edge_list), self.neg_sampling_size)
        )
        for i in range(len(self.edge_list)):
            self.edge_list[i][2] = list(neg[i])

    def __len__(self):
        return len(self.edge_list)

    def __getitem__(self, idx):
        node, text, neg_text = self.edge_list[idx]
        text = self.text_list[text]
        neg_text = [self.text_list[t] for t in neg_text]
        return node, text, neg_text


class MultipleLableDataset(Dataset):
    """
    Multiple label dataset
    Input is a file, where each line contains:
        source_node, target_node and some labels
    """

    def __init__(
        self,
        u_map,
        v_map,
        label_filename,
        total_label_count,
        ratio=1.0,
        topk=-1,
        use_all_zeros=False
    ):
        """
        Args:
            edge_list (list): A list containts all edges. Each edge is a tuple
        """
        print('Use %0.2f labels' % ratio)
        self.labels = []
        fin = open(label_filename)
        fin.readline()
        not_count_topk = 0
        for idx, line in enumerate(fin):
            if numpy.random.rand() > ratio:
                continue
            line = line.strip().split(' ')
            line[0] = u_map.str2id(line[0])
            line[1] = v_map.str2id(line[1])
            label = [float(x) for x in line[2:]]
            label = numpy.array(label)
            if (label == 0).all():
                if not use_all_zeros:
                    continue
            if topk > 0:
                if len(set(label) - {0, 1}) > 0:
                    s = numpy.argsort(label)
                    label[s[-topk:]] = 1.0
                    label[s[:-topk]] = 0.0
                else:
                    not_count_topk += 1
            self.labels.append([line[0], line[1], label])
        print('not count topk', not_count_topk)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]


if __name__ == '__main__':
    edge_list_path = "/mnt/store1/plus1lab/multilabel-data/v1_aminer_10k_1/original-edgelist.txt"
    edge_label_path = "/mnt/store1/plus1lab/multilabel-model/aegcn/v1_aminer_10k_1/prop-mat.txt"
    node_map = Str2id()
    label_map = Str2id()
    edges = []
    NormalEdgeDataset(
        node_map,
        node_map,
        edge_list_path,
        10,
        from_random_walk=True,
        window_size=10
    )
    node_map.freeze = True
    d = MultipleLableDataset(node_map, node_map, edge_label_path, 100, 1.0, 5)
    print(d[999])
