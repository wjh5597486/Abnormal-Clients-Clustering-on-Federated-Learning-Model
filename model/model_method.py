import torch


def train_model(model, train_data,
                epoch=1,
                noise_rate=False,
                device="cpu",
                loss_f=torch.nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam,
                lr=0.001):
    optimizer=optimizer(model.parameters(), lr=lr)
    for i in range(epoch):
        for x, y in train_data:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            if noise_rate:
                x = x * (1 - noise_rate) + noise_rate * torch.randn_like(x)
            y_predict = model(x)
            loss = loss_f(y_predict, y)
            loss.backward()
            optimizer.step()

def check_performance(model, test_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    cnt = 0
    for x, y in test_data:
        x = x.to(device)
        y_predict = model(x.reshape(1, 1, 28, 28))
        y_predict = torch.argmax(y_predict)
        if y_predict == y:
            cnt += 1
    return cnt / len(test_data)