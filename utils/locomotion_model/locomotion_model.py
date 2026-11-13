from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg, GO2RoughCfgPPO
from rsl_rl.modules.former.transformer_in_use import RCTransformer  # , RGTransformer
import torch
from torch import nn
from torch.distributions import Normal
from .locomotion_base_model import LocoMotionBaseModel, LocoMotionBaseModelCfg, get_activation
from dataclasses import dataclass, field


@dataclass
class LocoMotionModelCfg(LocoMotionBaseModelCfg):
    model_path: str = "/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/locomotion_model/long_short_transformer/locomotion.pt"

    
class LocoMotionModel(LocoMotionBaseModel):
    def __init__(self,
                 cfg_local: LocoMotionModelCfg = LocoMotionModelCfg(),
                 **kwargs,
                 ):
        super().__init__()
        cfg = GO2RoughCfg()
        train_cfg = GO2RoughCfgPPO()
        self.cfg_local = cfg_local
        model_path = self.cfg_local.model_path

        self.long_encoder, self.short_proj, self.actor_backbone = self.prepare_models(cfg, train_cfg)

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        self.long_encoder.load_state_dict(state_dict["long_encoder"])
        print("Load long encoder successfuly")
        self.short_proj.load_state_dict(state_dict["short_proj"])
        print("Load short encoder successfuly")
        self.actor_backbone.load_state_dict(state_dict["actor_backbone"])
        print("Load actor actor_backbone successfuly")
        self.std = nn.Parameter(train_cfg.policy.init_noise_std * torch.ones(cfg.env.num_actions))

    def compute_latents(self, obs):
        """
        Slices always assume your original full observation layout so you don't
        need to change the env. We just compute what we need, based on flags.

        Returns dict with only the pieces the forward() may use.
        """
        B = obs.shape[0]
        P = self.proprio_obs_dim
        terrain_one_hot_dim = 1

        expected_obs_dim = 3 + P + terrain_one_hot_dim + 3 + \
            self.short_history_length * P + \
            self.long_history_length * P
        assert obs.shape[1] == expected_obs_dim, \
            f"Expected obs.shape[1]={expected_obs_dim}, got {obs.shape[1]}"

        command_obs = obs[:, :3]
        current_obs = obs[:, 3: 3 + P + 3]  # (B, P+3)

        flat_short = obs[:, 3 + P + terrain_one_hot_dim + 3:
                         3 + P + terrain_one_hot_dim + 3 + self.short_history_length * P]
        short_seq = flat_short.view(B, self.short_history_length, P)  # includes current step

        flat_long = obs[:, 3 + P + terrain_one_hot_dim + 3 + self.short_history_length * P:]
        long_seq = flat_long.view(B, self.long_history_length, P)

        out = {"command_obs": command_obs}

        out["current_obs"] = current_obs

        recent_6 = short_seq[:, -6:, :].contiguous().view(B, -1)  # (B, 6P)
        out["short_proj"] = self.short_proj(recent_6)

        out["latent_long"] = self.long_encoder(long_seq)

        self.out = out

    def update_distribution(self):
        x_parts = [self.out["command_obs"], self.out["current_obs"], self.out["short_proj"], self.out["latent_long"]]
        x = torch.cat(x_parts, dim=-1)
        mean = self.actor_backbone(x)
        print(f"[DEBUG][MODIFIED]: {mean}")
        # Actor.forward returns actions only
        mean = torch.clamp(mean, -100, 100)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print("[NaN/Inf Mean] in update_distribution, obs stats:")
            raise ValueError("Actor mean has NaNs or Infs!")
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self,
            command_obs=None,
            current_obs=None,
            short_proj=None,
            latent_long=None,
            ):
        if command_obs is None:
            command_obs = self.out["command_obs"]
        if current_obs is None:
            current_obs = self.out["current_obs"]
        if short_proj is None:
            short_proj = self.out["short_proj"]
        if latent_long is None:
            latent_long = self.out["latent_long"]
        x_parts = [command_obs, current_obs, short_proj, latent_long]
        x = torch.cat(x_parts, dim=-1)
        mean = self.actor_backbone(x)
        return mean

    def prepare_models(self, cfg: GO2RoughCfg, train_cfg: GO2RoughCfgPPO):

        activation = get_activation(train_cfg.policy.activation)
        long_history_length = cfg.env.num_long_obs_history_length
        num_proprio_obs = cfg.env.num_proprio_obs
        dim_long_history_latent = cfg.env.dim_long_history_latent

        self.proprio_obs_dim = num_proprio_obs
        self.short_history_length = cfg.env.num_short_obs_history_length
        self.long_history_length = long_history_length

        # Example config; adjust as needed
        long_encoder = RCTransformer(
            emb=32, heads=2, depth=1,
            seq_length=long_history_length,
            obs_act_dim=num_proprio_obs,
            output_dim=dim_long_history_latent,
            causal=False, dropout=0.0,
        )

        H_1, H_2, L = 128, 64, dim_long_history_latent
        short_encoder = nn.Sequential(
            nn.Linear(6 * num_proprio_obs, H_1),
            activation,
            nn.Linear(H_1, H_2),
            activation,
            nn.Linear(H_2, L)
        )

        actor_hidden_dims = train_cfg.policy.actor_hidden_dims

        input_dim = 3 + num_proprio_obs + 3 + dim_long_history_latent + dim_long_history_latent

        layers = [nn.Linear(input_dim, actor_hidden_dims[0]), activation]

        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                layers.append(nn.Linear(actor_hidden_dims[l], cfg.env.num_actions))
            else:
                layers += [nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]), activation]
        actor_backbone = nn.Sequential(*layers)

        return long_encoder, short_encoder, actor_backbone

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


def main():
    pass


if __name__ == "__main__":
    main()
