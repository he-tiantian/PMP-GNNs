import dgl
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import accuracy, training_performance
from model import PMPGNN


class Training:
    def __init__(self, dataset, r1, r2, v1, v2, t1, t2, random_split, graph, labels, idx_train,
                 idx_val, idx_test):
        self.fastmode = False
        self.random_split = random_split
        self.log_every = 50
        self.WL = True
        self.early = False
        self.seed = 36
        self.epochs = 1000
        self.lr = 0.01
        self.weight_decay = 5e-4

        if self.random_split:
            self.hidden = 64
        else:
            self.hidden = 64

        self.dropout = 0.5
        self.patience = 200
        self.data = dataset
        self.k = 5
        self.alpha = 0.1
        self.r1 = r1
        self.r2 = r2
        self.v1 = v1
        self.v2 = v2
        self.t1 = t1
        self.t2 = t2

        self.features = graph.ndata['feat']
        self.lapweights = graph.edata['lapw']

        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        g1 = dgl.remove_self_loop(graph)

        adj = g1.adjacency_matrix()

        self.adj = adj

        self.model = PMPGNN(nsample=self.adj.shape[0],
                            nfeat=self.features.shape[1],
                            nhid=self.hidden,
                            nclass=int(self.labels.max()) + 1,
                            alpha=self.alpha,
                            dropout=self.dropout,
                            k=self.k,
                            WL=self.WL,
                            random_split=random_split)

        if torch.cuda.is_available():
            device = torch.device("cuda:%d" % 0)
            self.model.cuda()
            self.features = self.features.cuda()
            self.lapweights = self.lapweights.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
            self.graph = graph.to(device)

        self.features, self.adj, self.labels = Variable(self.features), Variable(self.adj), Variable(self.labels)

        self.lapweights = Variable(self.lapweights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        print("Parameters per layer: ")
        print([np.prod(p.size()) for p in self.model.parameters() if p.requires_grad])
        print("Total parameters: " + str(sum([np.prod(p.size()) for p in self.model.parameters() if p.requires_grad])))

    def run(self):

        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = 0
        best_val = 1e9
        best_epoch = 0

        print("Network Fitting...")

        for epoch in range(self.epochs):
            loss_values.append(self.train(epoch))

            loss_t, acc_t = self.compute_test(epoch)

            if self.early:
                best_val, best, best_epoch = training_performance(loss_values[-1],
                                                                  best_val, acc_t,
                                                                  best,
                                                                  bad_counter,
                                                                  epoch,
                                                                  best_epoch,
                                                                  self.early)
                if bad_counter == self.patience:
                    break
            else:
                best_val, best, best_epoch = training_performance(loss_values[-1],
                                                                  best_val, acc_t,
                                                                  best,
                                                                  bad_counter,
                                                                  epoch,
                                                                  best_epoch,
                                                                  self.early)

        total_time = time.time() - t_total
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(total_time))

        print("Test accuracy is: {:.4f}".format(best))

        return best

    def train(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        predict, lp_loss = self.model(self.graph, self.features, self.lapweights)

        node_predict_loss = F.nll_loss(predict[self.idx_train], self.labels[self.idx_train])

        loss_train = node_predict_loss + lp_loss

        acc_train = accuracy(predict[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

        if not self.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            predict, _ = self.model(self.graph, self.features, self.lapweights)

        loss_val = F.nll_loss(predict[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(predict[self.idx_val], self.labels[self.idx_val])

        if (epoch + 1) % self.log_every == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(node_predict_loss.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def compute_test(self, epoch):
        self.model.eval()

        predict, _ = self.model(self.graph, self.features, self.lapweights)

        loss_test = F.nll_loss(predict[self.idx_val], self.labels[self.idx_val])
        acc_test = accuracy(predict[self.idx_test], self.labels[self.idx_test])

        if (epoch + 1) % self.log_every == 0:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.data),
                  "accuracy= {:.4f}".format(acc_test.data))

        return loss_test, acc_test
