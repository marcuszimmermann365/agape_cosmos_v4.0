import torch
import torch.nn as nn
from torchsde import sdeint

class SDEVectorField(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.Softplus(),
            nn.Linear(256, d_out)
        )

    def forward(self, t, y):
        return self.net(y)

class Jiva(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.drift = SDEVectorField(config.d_hidden, config.d_hidden)
        self.diffusion = SDEVectorField(config.d_hidden, config.d_hidden)
        self.decoder = nn.Linear(config.d_hidden, config.d_world)
        self.ts = torch.linspace(0, 1, 2)
        self.u_agape = nn.Parameter(torch.randn(5))

    def get_agape_weights(self):
        import torch.nn.functional as F
        return F.softmax(self.u_agape, dim=0)

    def forward(self, z):
        h0 = torch.zeros(z.size(0), self.cfg.d_hidden, device=z.device)
        h_final = sdeint(self, h0, self.ts, method='srk', dt=0.1, sde_type='stratonovich')[-1]
        x_hat = self.decoder(h_final)
        return x_hat, h_final

class CosmicSubstrate:
    def __init__(self, config, device):
        self.state = torch.randn(config.d_world, config.d_latent, device=device) * 0.5
        self.vector_field = SDEVectorField(config.d_latent, config.d_latent).to(device)
        self.ts = torch.linspace(0, 1, 2)
        self.cfg = config
        self.device = device

    def get_reality_for(self, z):
        return z @ self.state.T

    def evolve(self, collective_actions):
        with torch.no_grad():
            f = lambda t, y: self.vector_field(y.T).T * self.cfg.env_drift
            g = lambda t, y: torch.ones_like(y) * self.cfg.env_diffusion
            new_state = sdeint(f, g, self.state, self.ts)[-1]
            perturbation = collective_actions.mean(dim=0).T
            self.state = (1 - self.cfg.env_plasticity) * new_state + self.cfg.env_plasticity * perturbation
