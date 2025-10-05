import torch

class Config:
    # Model Dimensions
    d_latent = 32
    d_hidden = 128
    d_world = 64

    # Simulation
    num_jivas = 16
    batch_size = 256

    # Bi-Level Optimizer
    num_outer_steps = 100
    num_inner_steps = 50
    lr_theta = 1e-4  # Fast learning for world-models
    lr_w = 2e-5      # Slow learning for values/wisdom

    # Physics & SDE
    env_drift = 0.01
    env_diffusion = 0.005
    env_plasticity = 0.1
    jiva_diffusion = 0.02

    # Social Dynamics
    lambda_coupling = 0.5
    graph_update_freq = 25
    connection_threshold = 0.3
    procrustes_regularization = 1e-4

    # Agape & Ethics
    lambda_agape = 1.0
    ahimsa_threshold = 0.8

    utopian_targets = torch.tensor([10.0, 0.0, 0.0, 20.0, 0.0]) # Ideal J1..J5
    alpha_weights = torch.tensor([0.5, 1.0, 1.0, 2.0, 5.0])     # Importance of J1..J5
