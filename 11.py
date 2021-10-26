import os

example_path = os.path.join('.', 'example_mani_skill_data')
for file_name in os.listdir(example_path):
    if file_name.find("README") >= 0:
        continue

    tar_name = file_name.split('=')[-1]
    cur_path = os.path.join(example_path, file_name)
    tar_path = os.path.join(example_path, tar_name)
    os.rename(cur_path, tar_path)