
import torch
import torchvision


import sys
sys.path.append('model')
from model.dataset import load_data
from model.dataset import data_split
from model.server_client import Server
from model.server_client import Client
from model.cnn_model import CNN
from model.model_method import check_performance

descript = True  # descript setting

# Setting federated parameters
# -client setting
num_clients = 100
num_abnormal = 20  # 1 to this number of clients will be the abnormal clients

# -server setting
num_pos_flags = 5
num_neg_flags = 5
batch_size = 10

# -training setting
rounds = 10
epoch = 1

if descript:
    print(f"The number of    total clients: {num_clients:>5}")
    print(f"The number of   normal clients: {num_clients - num_abnormal:>5}")
    print(f"The number of abnormal clients: {num_abnormal:>5}")
    print(f"positive flags: {num_pos_flags},  negative flags: {num_neg_flags}")
    print("-"*40)
    print(f"positive client numbers : {1:3} to {num_abnormal:3}")
    print(f"negative client numbers : {num_abnormal+1:3} to {num_clients:3}")
    print("-"*40)

# Download data
root = "./data"
data_train, test_data = load_data(root)
data_train = data_split(data_train, num_clients + num_neg_flags + num_neg_flags, batch_size=batch_size)  # separate data
train_data_client, train_data_global = data_train[:100], data_train[100:]

# build CNN Model
cnn = CNN()

# make Global server and clients
global_server = Server(cnn, train_data_global, test_data, num_pos_flags, num_neg_flags, num_clients)
clients = [Client(idx, CNN(), train_data_client[idx], abnormal=idx <= num_abnormal) for idx in range(num_clients)]

# make connections
for idx, client in enumerate(clients):
    client.server_url = global_server
    global_server.model_child[idx] = client

""" training session """

for i in range(rounds):
    # local_model
    for client in clients:
        client.get_model()
        client.train_model(epoch=epoch)
        client.post_model()

    # acc = check_performance(client.model, test_data)
    # print(acc, "local")

    # global_server
    global_server.train_flags(epoch=epoch)
    global_server.valid_model(cases=1, k=3)
    global_server.update_model()


    # Description for the result.
    filtered_list = global_server.get_filtered_list()
    print(f"the number of the filtered clients : {len(filtered_list)}")


    acc = global_server.check_performance_model()
    print(acc, "global")


# Training
print("fin")