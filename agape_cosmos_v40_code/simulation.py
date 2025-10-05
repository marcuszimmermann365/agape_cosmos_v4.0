import torch
from config import Config
from models import Jiva, CosmicSubstrate
from objectives import (j1_integration, j2_diversity, j3_robustness, j4_empowerment_proxy,
                        j5_causal_ahimsa_proxy, regularized_procrustes_analysis)

class MultiAgentSystem:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        self.jivas = [Jiva(config).to(device) for _ in range(config.num_jivas)]
        self.environment = CosmicSubstrate(config, device)

        theta_params = [p for jiva in self.jivas for p in jiva.parameters() if p.requires_grad]
        self.optimizer_theta = torch.optim.Adam(theta_params, lr=config.lr_theta)

        w_params = [jiva.u_agape for jiva in self.jivas]
        self.optimizer_w = torch.optim.Adam(w_params, lr=config.lr_w)

        self.social_graph = torch.ones(config.num_jivas, config.num_jivas, device=device) * 0.5

    def calculate_total_loss(self, outputs, z, x_world):
        h_states = [out[1] for out in outputs]
        total_loss = 0

        # Social Loss
        social_loss = 0
        for i in range(self.cfg.num_jivas):
            for j in range(i + 1, self.cfg.num_jivas):
                social_loss += self.social_graph[i, j] * regularized_procrustes_analysis(h_states[i], h_states[j],
                                    self.cfg.procrustes_regularization)
        total_loss += self.cfg.lambda_coupling * (social_loss / (self.cfg.num_jivas * (self.cfg.num_jivas - 1) / 2))

        all_j_values = []
        for i in range(self.cfg.num_jivas):
            x_hat_i, h_i = outputs[i]
            total_loss += torch.mean((x_hat_i - x_world) ** 2)

            j1 = j1_integration(h_i)
            j2 = j2_diversity(h_i)
            j3 = j3_robustness(x_hat_i, z)
            j4 = j4_empowerment_proxy(h_i, self.jivas[i].decoder)

            harm_caused = 0
            for j in range(self.cfg.num_jivas):
                if i == j:
                    continue
                harm_caused += j5_causal_ahimsa_proxy(h_i, h_states[j], self.jivas[j].decoder)
            j5 = -harm_caused

            objectives = torch.stack([j1, j2, j3, j4, j5])
            all_j_values.append(objectives)

        return total_loss, all_j_values

    def run_simulation(self):
        print("Starting Agape Cosmos V4.1 Simulation...")
        for outer_step in range(self.cfg.num_outer_steps):
            utopian_targets = self.cfg.utopian_targets.to(self.device)
            alpha_weights = self.cfg.alpha_weights.to(self.device)

            for inner_step in range(self.cfg.num_inner_steps):
                z = torch.randn(self.cfg.batch_size, self.cfg.d_latent, device=self.device, requires_grad=True)
                x_world = self.environment.get_reality_for(z)
                outputs = [jiva(z) for jiva in self.jivas]

                total_loss, all_j_values = self.calculate_total_loss(outputs, z, x_world)

                self.optimizer_theta.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer_theta.step()

            meta_loss = 0
            for i in range(self.cfg.num_jivas):
                weighted_distances = alpha_weights * torch.abs(utopian_targets - all_j_values[i].detach())
                meta_loss += torch.max(weighted_distances)

            self.optimizer_w.zero_grad()
            meta_loss.backward()
            self.optimizer_w.step()

            with torch.no_grad():
                collective_actions = torch.stack([out[0] for out in outputs])
                self.environment.evolve(collective_actions)

            print(f"Outer Step {outer_step+1}/{self.cfg.num_outer_steps} | Final Inner Loss: {total_loss.item():.4f} | Meta Loss: {meta_loss.item():.4f}")

        print("Simulation complete.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    system = MultiAgentSystem(config, device)
    system.run_simulation()
