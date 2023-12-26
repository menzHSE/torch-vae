import torch.nn.functional as F



def mse_kl_loss(x, x_recon, kl_div, alpha_kl_div_loss=1.0):
    """The loss function for the VAE. It is a combination of the reconstruction loss
    and alpha_kl_div_loss * the KL divergence between the latent distribution and the standard normal distribution.
    """

    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # Total loss
    loss = recon_loss + alpha_kl_div_loss * kl_div
    #print(f"Loss: {loss.item():.2f} | Recon loss: {recon_loss.item():.2f} | KL div: {kl_div.item():.2f}", end="\n", flush=True)
    return loss

