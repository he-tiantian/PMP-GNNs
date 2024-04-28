from network_training import Training
from utils import load_data, shuffle_nodes
import numpy as np
import torch

# this py file can also be used to train with flickr dataset:
# NO. of classes * 20 (train), 500 nodes for validation, all nodes for clustering

# blog/flickr
data = 'blog'

# Times to train the network to obtain average performances.
# One can run this py file multiple times to obtain steady performances
times = 5

r1 = 0
r2 = 120
v1 = 120
v2 = 620

# t1 t2 does not use for test split
t1 = 620
t2 = 700

# True for randomly split data into train, validation and test sets.
random_split = True

graph, labels, idx_train, idx_val, _ = load_data(data, r1, r2, v1, v2, t1, t2, random_split)

# clustering all nodes in the dataset
idx_test = torch.LongTensor(range(0, labels.shape[0]))

accs = []

for i in range(times):
    training = Training(dataset=data, r1=r1, r2=r2, v1=v1, v2=v2,
                        t1=t1, t2=t2, random_split=random_split,
                        graph=graph, labels=labels, idx_train=idx_train,
                        idx_val=idx_val, idx_test=idx_test)
    a = training.run()
    accs.append(a.cpu().numpy())

print(accs)
print(f"Average accuracy for classification: {np.mean(accs)} ± {np.std(accs)}")
