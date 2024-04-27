from network_training import Training
from utils import load_data, shuffle_nodes
import numpy as np

# change according to the name of the dataset: bbcsport/bbc/guardian2013/irishtimes2013/wikihigh
data = 'bbc'

# Times to train the network to obtain average performances.
# One can run this py file multiple times to obtain steady performances
times = 5

# True for randomly split data into train, validation and test sets.
random_split = True

# set it to make load_data work, train/validation/test will be generated by shuffle_nodes
r1 = 0
r2 = 2
v1 = 4
v2 = 6
t1 = 8
t2 = 10

graph, labels, _, _, _ = load_data(data, r1, r2, v1, v2, t1, t2, random_split)

# Ratio for train/validation/test: 6:2:2.
n = graph.number_of_nodes()
r1 = 0
r2 = int(n * 0.6)
v1 = r2
v2 = v1 + int(n * 0.2)
t1 = v2
t2 = n

accs = []

idx_train, idx_val, idx_test = shuffle_nodes(r1, r2, v1, v2, t1, t2, graph.number_of_nodes())

for i in range(times):
    training = Training(dataset=data, r1=r1, r2=r2, v1=v1, v2=v2,
                        t1=t1, t2=t2, random_split=random_split,
                        graph=graph, labels=labels, idx_train=idx_train,
                        idx_val=idx_val, idx_test=idx_test)
    a = training.run()
    accs.append(a.cpu().numpy())

print(accs)
print(f"Average accuracy for classification: {np.mean(accs)} ± {np.std(accs)}")
