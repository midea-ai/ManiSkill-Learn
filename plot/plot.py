import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/quan/maniskill_model/11-03/20-27-13/files/ppo_file.npz'
all_file = np.load(file_path)
reward = all_file['reward']
plt.plot(reward)
plt.savefig('reward.jpg')

losses = all_file['loss']
sum_total_loss = losses[:, 0]
sum_action_loss = losses[:, 1]
sum_value_loss = losses[:, 2]
sum_ent_loss = losses[:, 3]

plt.clf()
plt.plot(sum_total_loss)
plt.savefig('total_loss.jpg')

plt.clf()
plt.plot(sum_action_loss)
plt.savefig('action_loss.jpg')

plt.clf()
plt.plot(sum_value_loss)
plt.savefig('value_loss.jpg')

plt.clf()
plt.plot(sum_ent_loss)
plt.savefig('entropy_loss.jpg')