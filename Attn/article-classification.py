from network_training import Training
from utils import load_data
import numpy as np

# change according to the name of the dataset: cora/cite/pubmed
data = 'cora'

# Times to train the network to obtain average performances.
times = 5

# False for using split established by previous studies.
random_split = False

# Reading fixed splits
r1 = 0
r2 = 0
v1 = 0
v2 = 0
t1 = 0
t2 = 0


graph, labels, idx_train, idx_val, idx_test = load_data(data, r1, r2, v1, v2, t1, t2, random_split)

accs = []

for i in range(times):
    training = Training(dataset=data, r1=r1, r2=r2, v1=v1, v2=v2, t1=t1, t2=t2, random_split=random_split,
                        graph=graph, labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    a = training.run()
    accs.append(a.cpu().numpy())

print(accs)
print(f"Average accuracy for classification: {np.mean(accs)} ± {np.std(accs)}")






