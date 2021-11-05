import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA

class demo_net(nn.Module):
    def __init__(self):
        super(demo_net, self).__init__()

        self.fc1 = torch.nn.Linear(2, 2)

        self.eigen_vectors = []
        for i in range(2):
            tmp_matrix = np.random.rand(9).reshape(3, 3)
            tmp_matrix = np.triu(tmp_matrix)
            tmp_matrix = tmp_matrix + tmp_matrix.T - np.diag(tmp_matrix.diagonal())
            # print(tmp_matrix)
            eigen_vector, _ = LA.eigh(tmp_matrix)
            eigen_vector = torch.from_numpy(eigen_vector).float()
            # eigen_vector.requires_grad = True
            # print('eigen_vector:', eigen_vector.shape)
            print(eigen_vector)
            self.eigen_vectors.append(eigen_vector)
        self.eigen_vectors = torch.stack(self.eigen_vectors, 0)
        self.eigen_vectors = nn.Parameter(self.eigen_vectors)
        print('eigen_vector:', self.eigen_vectors.shape)

net = demo_net()
d = net.state_dict()
# d.update({"custom_stuff": 123})
torch.save(d, "save_test.pth")

print("##################")
net1 = demo_net()

print("##################")
# x = torch.load("save_test.pth")
net1.load_state_dict(torch.load("save_test.pth"))
print(net1.eigen_vectors)