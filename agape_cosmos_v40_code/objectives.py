import torch
import torch.nn.functional as F

def j1_integration(h, eps=1e-6):
    h_centered = h - h.mean(dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / (h.size(0) - 1)
    cov_normalized = cov / (torch.trace(cov).detach() + eps)
    return torch.slogdet(cov_normalized + eps * torch.eye(h.size(1), device=h.device))[1]

def j2_diversity(h):
    h_centered = h - h.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(h_centered)
    return -torch.var(s)

def j3_robustness(x_hat, z, eps=1e-8):
    v = torch.randn_like(z)
    v = F.normalize(v, p=2, dim=1, eps=eps)
    Jv = torch.autograd.grad(x_hat, z, grad_outputs=v, retain_graph=True)[0]
    return -Jv.pow(2).sum()

def j4_empowerment_proxy(h, decoder):
    x_hat = decoder(h)
    return torch.log(x_hat.var(dim=0) + 1e-8).sum()

def j5_causal_ahimsa_proxy(h_i, h_j, decoder_j):
    x_hat_j_original = decoder_j(h_j).detach()
    h_j_perturbed = h_j + h_i.detach()
    x_hat_j_perturbed = decoder_j(h_j_perturbed)
    harm = (x_hat_j_perturbed - x_hat_j_original).pow(2).mean()
    return harm

def regularized_procrustes_analysis(H1, H2, reg=1e-4):
    H1_c = H1 - H1.mean(dim=0, keepdim=True)
    H2_c = H2 - H2.mean(dim=0, keepdim=True)
    M = H2_c.T @ H1_c
    U, S, Vt = torch.linalg.svd(M)
    R = U @ Vt
    reg_loss = reg * (R - torch.eye(R.shape[0], device=R.device)).pow(2).sum()
    H1_aligned = H1_c @ R.T
    distance = (H1_aligned - H2_c).pow(2).mean() + reg_loss
    return distance
