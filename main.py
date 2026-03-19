from tqdm import tqdm
from dataset import load_dataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision.models import resnet50
import torch
import os

weight_decay = 0.01
n_epoch = 300
batch_size = 128
dro_step = 0.01

criterion = torch.nn.CrossEntropyLoss(reduction='none')


class Logger():
    def __init__(self):
        self.file = open('logs/log.txt', 'w')

    def log(self, str):
        print(str)
        self.file.write(str)

    def flush(self):
        self.file.flush()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.file.close()

def train(data_loader, model, optimizer, q, device):
    total_loss_per_group = torch.zeros(4).to(device)
    total_correct_per_group = torch.zeros(4).to(device)
    total_count_per_group = torch.zeros(4).to(device)

    for x, y, group in tqdm(data_loader):
        x, y, group = x.to(device), y.to(device), group.to(device)

        optimizer.zero_grad()
        output = model(x)

        loss = criterion(output, y)
        predictions = output.argmax(dim=1)

        loss_per_group = torch.zeros(4).to(device)
        for group_index in range(4):
            mask = (group == group_index)
            if mask.sum() > 0:
                loss_per_group[group_index] = loss[mask].mean()
                total_loss_per_group[group_index] += loss[mask].sum()
                total_correct_per_group[group_index] += (predictions[mask] == y[mask]).sum()
                total_count_per_group[group_index] += mask.sum()

        q = q.detach() * torch.exp(dro_step * loss_per_group.detach())
        q = q / q.sum()

        loss_final = (q * loss_per_group).sum()
        l2 = sum(p.pow(2).sum() for p in model.parameters())
        loss_final = loss_final + weight_decay * l2

        loss_final.backward()
        optimizer.step()

    avg_loss_per_group = (total_loss_per_group / total_count_per_group.clamp(min=1)).tolist()
    avg_acc_per_group = (total_correct_per_group / total_count_per_group.clamp(min=1)).tolist()

    return q, avg_loss_per_group, avg_acc_per_group


def eval(data_loader, model, device):
    model.eval()
    total_loss_per_group = torch.zeros(4).to(device)
    total_correct_per_group = torch.zeros(4).to(device)
    total_count_per_group = torch.zeros(4).to(device)

    with torch.no_grad():
        for x, y, group in tqdm(data_loader):
            x, y, group = x.to(device), y.to(device), group.to(device)

            output = model(x)
            loss = criterion(output, y)
            predictions = output.argmax(dim=1)

            for group_index in range(4):
                mask = (group == group_index)
                if mask.sum() > 0:
                    total_loss_per_group[group_index] += loss[mask].sum()
                    total_correct_per_group[group_index] += (predictions[mask] == y[mask]).sum()
                    total_count_per_group[group_index] += mask.sum()

    model.train()

    avg_loss_per_group = (total_loss_per_group / total_count_per_group.clamp(min=1)).tolist()
    avg_acc_per_group = (total_correct_per_group / total_count_per_group.clamp(min=1)).tolist()

    return avg_loss_per_group, avg_acc_per_group


def log(label, n, avg_loss_per_group, avg_acc_per_group, logger):
    logger.log(f"{label} [{n}]")
    logger.log(f"  loss     | " + " | ".join([f"g{i}: {avg_loss_per_group[i]:.4f}" for i in range(4)]) + f" | avg: {sum(avg_loss_per_group)/4:.4f} | worst: {max(avg_loss_per_group):.4f}")
    logger.log(f"  accuracy | " + " | ".join([f"g{i}: {avg_acc_per_group[i]:.4f}" for i in range(4)]) + f" | avg: {sum(avg_acc_per_group)/4:.4f} | worst: {min(avg_acc_per_group):.4f}")


def main():
    os.makedirs('./logs', exist_ok=True)

    logger = Logger()

    train_dataset, val_dataset, test_dataset = load_dataset()
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet50(weights=None)
    state_dict = torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    )
    model.load_state_dict(state_dict)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    q = (torch.ones(4) / 4).to(device)

    for n in range(n_epoch):
        q, avg_loss_per_group, avg_acc_per_group = train(train_data_loader, model, optimizer, q, device)
        log("Train", n, avg_loss_per_group, avg_acc_per_group, logger)
        logger.log(f"  q        | " + " | ".join([f"g{i}: {q[i]:.4f}" for i in range(4)]))

        torch.cuda.empty_cache()

        avg_loss_per_group, avg_acc_per_group = eval(val_data_loader, model, device)
        log("Val", n, avg_loss_per_group, avg_acc_per_group, logger)

        torch.cuda.empty_cache()

        avg_loss_per_group, avg_acc_per_group = eval(test_data_loader, model, device)
        log("Test", n, avg_loss_per_group, avg_acc_per_group, logger)

        torch.cuda.empty_cache()

        if (n + 1) % 10 == 0:
            logger.flush()
            torch.save(model.state_dict(), f'./logs/model_epoch_{n + 1}.pth')
            print(f"  modèle sauvegardé : ./logs/model_epoch_{n + 1}.pth")


if __name__ == '__main__':
    main()