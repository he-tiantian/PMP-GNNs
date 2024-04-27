import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl import backend as F
import random


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# def load_data(path="./data/cora/", dataset="cora"):
def load_data(dataset, r1, r2, v1, v2, t1, t2, random_split):
    print('Loading {} dataset...'.format(dataset))
    path = "./../data/" + dataset + "/"

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # weights of normalized laplacian
    sim_matrix = similarity_matrix(adj)
    lap = sp.eye(adj.shape[0]) - normalize_adj(sim_matrix)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    edges = adj.nonzero()

    e0 = np.array(edges[0])
    e1 = np.array(edges[1])

    features = normalize_features(features)

    lapdata = np.array(lap[e0, e1])

    g = dgl.graph((e0, e1))

    g.ndata['feat'] = F.tensor(features.todense(), dtype=F.data_type_dict['float32'])

    g.edata['lapw'] = F.tensor(lapdata.reshape(e0.shape), dtype=F.data_type_dict['float32'])

    if random_split:
        idx_train, idx_val, idx_test = shuffle_nodes(r1, r2, v1, v2, t1, t2, adj.shape[0])
    else:
        if r1 == 0 and r2 == 0:
            idx_train = list()
            f = open(path + dataset + '.trainids')
            lines = f.readlines()
            for line in lines:
                idx_train.append(int(line))

            f.close()
            idx_train.sort()
        else:
            idx_train = range(r1, r2)

        idx_train = torch.LongTensor(idx_train)

        if v1 == 0 and v2 == 0:
            idx_val = list()
            f = open(path + dataset + '.validids')
            lines = f.readlines()
            for line in lines:
                idx_val.append(int(line))

            f.close()
            idx_val.sort()
        else:
            idx_val = range(v1, v2)

        idx_val = torch.LongTensor(idx_val)

        if t1 == 0 and t2 == 0:
            idx_test = list()
            f = open(path + dataset + '.testids')
            lines = f.readlines()
            for line in lines:
                idx_test.append(int(line))

            f.close()
            idx_test.sort()
        else:
            idx_test = range(t1, t2)

        idx_test = torch.LongTensor(idx_test)

    labels = torch.LongTensor(np.where(labels)[1])

    return g, labels, idx_train, idx_val, idx_test


def shuffle_nodes(r1, r2, v1, v2, t1, t2, number_of_nodes):
    node_ids = [node for node in range(number_of_nodes)]
    random.shuffle(node_ids)
    idx_train = node_ids[r1:r2]
    idx_val = node_ids[v1:v2]
    idx_test = node_ids[t1:t2]
    idx_train.sort()
    idx_val.sort()
    idx_test.sort()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum + 1e-9, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# early stopping control
def training_performance(val_loss, best_val, acc, best_acc, bad_counter, epoch, best_epoch, early):

    if early:
        if val_loss < best_val:
            best_acc = acc
            best_epoch = epoch
            best_val = val_loss
            bad_counter = 0
        else:
            bad_counter += 1

        return best_val, best_acc, best_epoch, bad_counter
    else:
        if val_loss < best_val:
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_val = val_loss

        return best_val, best_acc, best_epoch


def similarity_matrix(adj):
    """node-node similarity based on edge-Cosine function"""

    edges = adj.nonzero()
    e0 = np.array(edges[0])
    e1 = np.array(edges[1])

    rowsum = np.array(adj.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 1e-9
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    con_currence = adj.transpose().dot(adj)

    con_currence = con_currence + adj

    data = con_currence[e0, e1]

    data = np.array(data)

    s_matrix = sp.coo_matrix((data.reshape(e0.shape), (e0, e1)), shape=(adj.shape[0], adj.shape[0]), dtype=np.float32)

    return s_matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)