from tqdm import tqdm
import sys
import time
import argparse

sys.path.append('model')
from model.dataset import load_data
from model.dataset import data_split
from model.server_client import Server
from model.server_client import Client
from model.cnn_model import CNN
from model.F1_score import F1score


def main(args):
    """
    Abnormal Clustering In Federated learning Model
    1. pre-training the global model
    2. copying the global model and send client models.
    3. training client models with own dataset.
    4. sending back trained client models to the global server.
    5. generating validation vectors by inputting the random noise data.
    6. implement KNN- Clustering.
    7. remove

    :param args:
        .clients: the number of all local clients
        .abnormal: the number of abnormal clients/sibyls in all clients.
        .noise: the ratio of the noise in the abnormal clients' data
        .pos_flags: the number of positive flags used to generate validation vectors
        .neg_flags: the number of negative flags used to generate validation vectors.
        .batch: the size of batches.
        .round: training round that communication among the server and clients.
        .epoch: training epoch during each round.
        .k: the number of flags used to classify the vectors in the KNN algorithm.
        .case: the number of KNN cases implemented to classify.

    :return:
        None
    """
    assert args.k < args.neg_flags + args.pos_flags, "Invalid k, k must be less than the total number of flags."
    assert args.abnormal <= args.clients, "invalid abnormal, it must be less than the total number of clients."

    # -client setting
    num_clients = args.clients
    num_abnormal = args.abnormal  # 1 to this number of clients will be the abnormal clients
    noise_rate = args.noise

    # -server setting
    num_pos_flags = args.pos_flags
    num_neg_flags = args.neg_flags
    batch_size = args.batch

    # -training setting
    rounds = args.round
    epoch = args.epoch

    if args.description:
        print(f"The number of    total clients: {num_clients:>5}")
        print(f"The number of   normal clients: {num_clients - num_abnormal:>5}")
        print(f"The number of abnormal clients: {num_abnormal:>5}")
        print(f"positive flags: {num_pos_flags},  negative flags: {num_neg_flags}")
        print("-" * 40)
        print(f"positive client numbers : {0:3} to {num_abnormal - 1:3}")
        print(f"negative client numbers : {num_abnormal:3} to {num_clients-1:3}")
        print("-" * 40)

    # Download data
    root = "./data"
    data_train, test_data = load_data(root)
    data_train = data_split(data_train, num_clients + num_neg_flags + num_neg_flags,
                            batch_size=batch_size)  # separate data
    train_data_client, train_data_global = data_train[:100], data_train[100:]

    # build CNN Model
    cnn = CNN()

    # make Global server and clients
    global_server = Server(cnn,
                           train_data_global,
                           test_data,
                           num_pos_flags,
                           num_neg_flags,
                           num_clients,
                           pretrain=True)
    clients = [Client(idx,
                      CNN(),
                      train_data_client[idx],
                      abnormal=idx < num_abnormal,
                      noise_rate=noise_rate)
               for idx in range(num_clients)]

    # make connections
    for idx, client in enumerate(clients):
        client.server_url = global_server
        global_server.model_child[idx] = client

    """ training session """
    f1score = F1score(num_clients, num_abnormal)
    for i in range(rounds):
        # local_model
        for client in tqdm(clients, total=num_clients,
                           desc=f"Round[{i+1:03}/{args.round:03}] : ", leave=False):
            client.get_model()
            client.train_model(epoch=epoch)
            client.post_model()
        time.sleep(0.1)

        # global_server
        global_server.train_flags(epoch=epoch)
        global_server.valid_model(cases=1, k=3)
        global_server.update_model()

        # Description for the result.
        filtered_list = global_server.get_filtered_list()
        print(f"the number of the filtered clients : {len(filtered_list)}", filtered_list)

        acc = global_server.check_performance_model()
        print(f"Global Test Acc : {acc:.2%}")

        # compute F1 Score
        f1score.add(filtered_list)
        time.sleep(0.1)

    precision, recall, f1_score = f1score.get_f1()
    print(f"{precision = :.3}, {recall = :.3}, {f1_score = :.3}")

    # Training
    print("fin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "-clients", dest="clients", type=int, default=100)
    parser.add_argument("-ab", "-abnormal", dest="abnormal", type=int, default=20)
    parser.add_argument("-n", "-noise", dest="noise", type=float, default=0.6)
    parser.add_argument("-b", "-batch", dest="batch", type=int, default=10)
    parser.add_argument("-nf", "-neg_flags", dest="neg_flags", type=int, default=5)
    parser.add_argument("-pf", "-pos_flags", dest="pos_flags", type=int, default=5)
    parser.add_argument("-r", "-round", dest="round", type=int, default=5)
    parser.add_argument("-e", "-epoch", dest="epoch", type=int, default=1)
    parser.add_argument("-k", "-k", dest="k", type=int, default=3)
    parser.add_argument("-ca", "-case", dest="case", type=int, default=1)
    parser.add_argument("-d", "-description", dest="description", type=bool, default=True)
    args = parser.parse_args()
    main(args)
