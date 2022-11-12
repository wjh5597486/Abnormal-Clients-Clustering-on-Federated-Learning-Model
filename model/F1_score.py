class F1score:
    def __init__(self, num_clients: int, num_abnormal: int):
        self.num_clients = list(range(num_clients))
        self.num_abnormal = set(range(num_abnormal))

        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

    def add(self, filtered_list):
        filtered_list = set(filtered_list)
        for idx in self.num_clients:
            if idx not in self.num_abnormal and idx not in filtered_list:
                self.true_positive += 1

            elif idx in self.num_abnormal and idx in filtered_list:
                self.true_negative += 1

            elif idx not in self.num_abnormal and idx in filtered_list:
                self.false_negative += 1

            elif idx in self.num_abnormal and idx not in filtered_list:
                self.false_positive += 1

    def get_f1(self):
        precision = self.true_positive / (self.true_positive + self.false_positive)
        recall = self.true_positive / (self.true_positive + self.false_negative)
        f1_score = 2 * precision * recall / (precision + recall)
        return precision, recall, f1_score
