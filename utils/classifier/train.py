import torch
from torch import nn
import torch.optim as optim
from dataloader import generate_dataset
import os
import argparse
from datetime import datetime
from classifier import Classifier, Config
from dataloader import DataLoader


now = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default=f"{now.strftime('%Y-%m-%d_%H-%M-%S')}")
args_cli = parser.parse_args()



def train():
    cfg = Config()
    device = cfg.device
    save_dir = os.path.join(cfg.save_dir, args_cli.run_name)

    if not os.path.exists(save_dir):
        os.makedirs(name=save_dir, exist_ok=True)

    model: Classifier = Classifier(cfg.calc_input_dim(), cfg.output_dim, cfg.hidden_dims).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    num_epochs = cfg.num_epochs

    train_set, test_set = generate_dataset(mode=cfg.mode, dataset_root=cfg.dataset_dir, unknown_label=cfg.output_dim - 1)

    # train_set = fake_data(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        print("")
        print("")

        print(f"------------ epoch {epoch}/{num_epochs} -----------")

        for xb, yb in train_loader:
            xb: torch.Tensor = xb.to(device)
            yb: torch.Tensor = yb.to(device)

            logits = model.forward(xb)
            loss: torch.Tensor = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"total loss: {total_loss}")

        if epoch % 50 == 0:
            torch.save(
                {
                    "model": model.state_dict()
                },
                f"{save_dir}/model_{epoch}.pt")

        # Evaluation
        idx = torch.randint(0, len(test_set), (1,))
        xb, yb = test_set[idx]
        xb = xb.to(device)
        yb = yb.to(device)

        logits, preds = model.inference(xb)
        loss: torch.Tensor = criterion(logits, yb)
        print(f"eval loss: {loss.mean(dim=-1).item()}")

    pass


if __name__ == "__main__":
    train()
    pass
