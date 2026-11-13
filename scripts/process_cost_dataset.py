import argparse
import os
import torch
from datetime import datetime
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Src dir")
    parser.add_argument("--tgt", help="Target dir")
    args_cli = parser.parse_args()

    root_dir = args_cli.src
    subs = os.listdir(root_dir)

    input = []
    label = []

    for terrain in tqdm(subs):
        terrain_dir = f"{root_dir}/{terrain}"
        cost_map = torch.load(f"{terrain_dir}/cost_map.pt", weights_only=True)
        long_map = torch.load(f"{terrain_dir}/long_map.pt", weights_only=True)
        short_map = torch.load(f"{terrain_dir}/short_map.pt", weights_only=True)

        # Only valid under 8m x 8m cell, but 12m x 12m perception
        valid_cost_map = cost_map[0, 4:-4, 4:-4, :]
        valid_long_map = long_map[0, 4:-4, 4:-4, :]
        valid_short_map = short_map[0, 4:-4, 4:-4, :]

        valid_latent_map = torch.cat([valid_long_map, valid_short_map], dim=-1)

        sub_input = valid_latent_map.flatten(start_dim=0, end_dim=1)
        sub_label = valid_cost_map.flatten(start_dim=0, end_dim=1)

        input.append(sub_input)
        label.append(sub_label)

    input_tensor = torch.cat(input, dim=0)
    label_tensor = torch.cat(label, dim=0)
    
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")

    target_path = args_cli.tgt
    os.makedirs(target_path, exist_ok=True)
    torch.save(
        {
            "input": input_tensor,
            "label": label_tensor,
        }, f"{target_path}/{now}.pt"
    )


if __name__ == "__main__":
    main()
