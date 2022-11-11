import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(root):
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    transform=transforms.ToTensor(),
                                                    train=True,
                                                    download=True)

    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   transform=transforms.ToTensor(),
                                                   train=False,
                                                   download=True)

    return mnist_train, mnist_test


def mk_batch_set(dataset, batch_size):
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)
    return data_loader


"""
split data
"""
def data_split(data, num_clients, IID=True, data_num_list="auto", batch_size=False):
    assert IID, "None-IID is not exist."
    num_data = len(data)

    if data_num_list == "auto":
        data_num_list = [num_data//num_clients] * num_clients + [num_data%num_clients]

    if IID:
        res = torch.utils.data.random_split(data, data_num_list)

    if batch_size:
        for idx, data_client in enumerate(res):
            res[idx] = mk_batch_set(data_client, batch_size)

    return res[:-1]

