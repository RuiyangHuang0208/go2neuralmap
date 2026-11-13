import torch
from torch import nn
import torch.optim as optim
from dataloader import split_set, CostMapDataset
import os
import argparse
from datetime import datetime
from cost_predict_nn_sigle_latent import CostPredNNSingleLatent, Config
from dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


def logger_builder(save_dir):
    os.makedirs(name=save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    def foo(info_dict: dict):
        for key, val in info_dict.items():
            writer.add_scalar(key, *val)
            print("added item")
        pass

    def foo1():
        writer.close()
    return foo, foo1


now = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default=f"cost_pred-{now.strftime('%Y-%m-%d_%H-%M-%S')}")
parser.add_argument("--dataset", type=str, required=True)
args_cli = parser.parse_args()

def train():
    cfg = Config()
    cfg.num_epochs = 5000
    device = cfg.device
    save_dir = os.path.join(cfg.save_dir, args_cli.run_name)
    logger, logger_close = logger_builder(f"{save_dir}/stats")

    if not os.path.exists(save_dir):
        os.makedirs(name=save_dir, exist_ok=True)

    model: CostPredNNSingleLatent = CostPredNNSingleLatent(cfg).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    num_epochs = cfg.num_epochs

    # dataset_origin = torch.load("abc.pt")
    dataset_origin = torch.load(args_cli.dataset)
    train_idx, test_idx = split_set(dataset_origin["input"].shape[0])

    training_set = CostMapDataset(dataset_origin["input"][train_idx], dataset_origin["label"][train_idx])
    test_set = CostMapDataset(dataset_origin["input"][test_idx], dataset_origin["label"][test_idx])

    train_loader = DataLoader(training_set, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(num_epochs):
        info_dict = {}
        model.train()
        total_loss = 0.0

        print("")
        print("")

        print(f"------------ epoch {epoch}/{num_epochs} -----------")

        for xb, yb in train_loader:
            xb: torch.Tensor = xb.to(device)
            yb: torch.Tensor = yb.to(device)

            pred = model.forward(xb)
            loss: torch.Tensor = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"totla loss: {total_loss}")
        info_dict["Loss/train"] = (total_loss, epoch)

        if epoch % 500 == 0:
            torch.save(
                {
                    "model": model.state_dict()
                },
                f"{save_dir}/model_{epoch}.pt")
        idx = torch.randint(0, len(test_set), (1,))
        xb, yb = test_set[idx]
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model.inference(xb)
        loss = criterion(pred, yb)
        print(f"eval loss: {loss.mean(dim=-1).item()}")
        info_dict["Loss/eval"] = (loss, epoch)
        logger(info_dict.copy())
    logger_close()


if __name__ == "__main__":
    train()
