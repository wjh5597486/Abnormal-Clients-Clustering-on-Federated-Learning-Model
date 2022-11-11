import torch
import model_method
from copy import deepcopy
from clustering import KNN_clustering
from collections import OrderedDict

device = "cuda" if torch.cuda.is_available() else "cpu"


class Server:
    def __init__(self, model, train_data, test_data,
                 num_pos_flags, num_neg_flags, num_clients,
                 noise_rate=0.8):

        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data

        # for flags
        self.num_pos_flags = num_pos_flags
        self.num_neg_flags = num_neg_flags
        self.flags_class = [1] * num_pos_flags + [0] * num_neg_flags
        self.noise_rate_list = [0] * num_pos_flags + [noise_rate] * num_neg_flags

        self.num_clients = num_clients
        self.num_models = num_pos_flags + num_neg_flags + num_clients
        self.model_child = [None] * self.num_models
        self.model_buffer = [None] * self.num_models

        self.valid_model_result = None

        self.filtered_list = None

    def asked_model(self):
        return self.model.state_dict()

    def take_model(self, idx, model):
        try:
            self.model_buffer[idx] = model
            return "success"

        except RuntimeError:
            return "fail"

    def train_flags(self, epoch):

        for i in range(self.num_pos_flags + self.num_neg_flags):
            # copy parameter
            self.model.load_state_dict(deepcopy(self.model.state_dict()))

            # train model
            model_method.train_model(model=self.model,
                                     train_data=self.train_data[i],
                                     epoch=epoch,
                                     noise_rate=self.noise_rate_list[i],
                                     device=device)

            # save parameter
            self.model_buffer[self.num_clients + i] = deepcopy(self.model.state_dict())

    def valid_model(self, k=3, cases=3, num_class=2):
        test_data = torch.randn_like(self.test_data[0][0])
        result_data = [None] * self.num_models
        for idx, param in enumerate(self.model_buffer):
            self.model.load_state_dict(param)
            result_data[idx] = self.model(test_data.to(device))

        result_list = torch.Tensor([[0] * num_class for i in range(self.num_clients)])
        for case in range(cases):
            res = KNN_clustering(k=k,
                                 flags=result_data[self.num_clients:],
                                 flags_class=self.flags_class,
                                 test_cases=result_data[:self.num_clients],
                                 num_class=num_class)
            result_list += torch.Tensor(res)

        self.valid_model_result = [0] * self.num_clients
        for i in range(self.num_clients):
            self.valid_model_result[i] = torch.argmax(result_list[i])


    def update_model(self, filter=True):
        cnt = 0
        updated_list = []
        filtered_list = []

        # compute new parameter
        self.model_parameter = OrderedDict()
        for idx, [normal, param] in enumerate(zip(self.valid_model_result, self.model_buffer)):
            if normal:
                cnt += 1
                updated_list.append(idx)
                for key in param.keys():
                    if key in self.model_parameter.keys():
                        self.model_parameter[key] += param[key]
                    else:
                        self.model_parameter[key] = param[key]
            else:
                filtered_list.append(idx)

        for key in self.model_parameter.keys():
            self.model_parameter[key] /= cnt

        self.filtered_list = filtered_list

    def check_performance_model(self):
        self.model.load_state_dict(deepcopy(self.model_parameter))
        return model_method.check_performance(self.model, self.test_data)

    def get_filtered_list(self):
        return self.filtered_list


class Client:
    def __init__(self, client_id, model, train_data, abnormal=False, noise_rate=0.7):
        self.id = client_id
        self.model = model.to(device)
        self.train_data = train_data
        self.noise_rate = noise_rate if abnormal else 0

        self.server_url = None

    def get_model(self):
        model = self.server_url.asked_model()
        self.model.load_state_dict(model)

    def post_model(self):
        model = deepcopy(self.model.state_dict())
        self.server_url.take_model(self.id, model)

    def train_model(self, epoch=1):
        model_method.train_model(model=self.model,
                                 train_data=self.train_data,
                                 epoch=epoch,
                                 noise_rate=self.noise_rate,
                                 device=device)
