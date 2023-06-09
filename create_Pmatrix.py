import numpy as np
# import configs
import os

P_matrix = []
M = 20
for i in range(M):
    matrixSize = 256 #512
    A = np.random.rand(matrixSize, matrixSize)
    B = np.dot(A, A.transpose())
    C = (B + B.T) / 2
    eigenvalue, featurevector = np.linalg.eig(C)

    list_a = eigenvalue.tolist()

    list_a_min_list = min(list_a)
    min_index = list_a.index(min(list_a))
    # featurevector_new = np.delete(featurevector.T, min_index, axis=0).T

    # P_matrix.append(featurevector_new)
    P_matrix.append(featurevector)

P_matrix= np.array(P_matrix)
print(P_matrix.shape)
# save_dir = configs.save_dir
# checkpoint_dir = '%s/checkpoints' % save_dir
# if not os.path.isdir(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
file_name = 'P' + str(M) + '_matrix_' +str(matrixSize) + '.npy'
# save_file = os.path.join(checkpoint_dir, 'P_matrix.npy')
np.save(file_name, P_matrix)
