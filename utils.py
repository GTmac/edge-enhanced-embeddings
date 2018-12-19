from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import torch
import numpy


class Str2id:
    def __init__(self):
        self.str2id_dict = dict()
        self.id2str_dict = dict()
        self.freeze = False

    def str2id(self, s):
        try:
            return self.str2id_dict[s]
        except:
            if self.freeze:
                raise Exception('Already Freezed')
            else:
                sid = len(self.str2id_dict)
                self.str2id_dict[s] = sid
                self.id2str_dict[sid] = s
                return self.str2id_dict[s]

    def id2str(self, sid):
        return self.id2str_dict[sid]


class NodeClassifier:
    def __init__(self, node_map, label_filename, used_label):
        self.Xid = []
        self.Y = []
        used_label = set(used_label)
        for line in open(label_filename):
            line = line.strip().split(' ')
            if int(line[1]) in used_label:
                self.Xid.append(node_map.str2id(line[0]))
                self.Y.append(int(line[1]))
        self.split(0.3, 41)

    def split(self, test_size, random_state):
        self.Xid_train, self.Xid_test, self.Y_train, self.Y_test = train_test_split(
            self.Xid, self.Y, test_size=test_size, random_state=random_state
        )

    def evaluate(self, model, use_cuda):
        u = model.forward(self.Xid_train, is_start=True, directed=True)
        v = model.forward(self.Xid_train, is_start=False, directed=True)
        if use_cuda:
            X_train = torch.cat((u, v), dim=1).cpu().data.numpy()
        else:
            X_train = torch.cat((u, v), dim=1).data.numpy()
        u = model.forward(self.Xid_test, is_start=True, directed=True)
        v = model.forward(self.Xid_test, is_start=False, directed=True)
        if use_cuda:
            X_test = torch.cat((u, v), dim=1).cpu().data.numpy()
        else:
            X_test = torch.cat((u, v), dim=1).data.numpy()
        y_train = [y for y in self.Y_train]
        y_test = [y for y in self.Y_test]
        clf = LogisticRegression(C=1e5, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return f1_score(y_test, y_pred, pos_label=None, average='macro')


def get_word_embeddings(word_embedding_filename, word_map):
    f = open(word_embedding_filename)
    first = f.readline()
    first = first.strip().split(' ')
    word_emb_dim = int(first[1])
    numpy.random.seed(71)
    word_embs = numpy.random.uniform(
        low=-0.5 / word_emb_dim,
        high=0.5 / word_emb_dim,
        size=(len(word_map.str2id_dict) + 1, word_emb_dim)
    )
    for line in f:
        line = line.strip().split(' ')
        if line[0] in word_map.str2id_dict:
            word_embs[word_map.str2id(line[0])] = [float(e) for e in line[1:]]
    return word_embs
