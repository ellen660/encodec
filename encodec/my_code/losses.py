import torch
# from audio_to_mel import Audio2Mel
from torch.nn import functional as F

# https://github.com/ZhikangNiu/encodec-pytorch/blob/main/losses.py

def loss_fn_l1(input, output):
    # l1Loss = torch.nn.L1Loss(reduction='mean')
    # return l1Loss(input, output)
    return F.l1_loss(input, output, reduction='mean')

def loss_fn_l2(input, output):
    # l2Loss = torch.nn.MSELoss(reduction='mean')
    # return l2Loss(input, output)
    return F.mse_loss(input, output, reduction='mean')

# def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=24000):
def total_loss(input, output, sample_rate=10):
    """This function is used to compute the total loss of the encodec generator.
        Loss = \lambda_t * L_t + \lambda_f * L_f + \lambda_g * L_g + \lambda_feat * L_feat
        L_t: time domain loss | L_f: frequency domain loss | L_g: generator loss | L_feat: feature loss
        \lambda_t = 0.1       | \lambda_f = 1              | \lambda_g = 3       | \lambda_feat = 3
    Args:
        fmap_real (list): fmap_real is the output of the discriminator when the input is the real audio. 
            len(fmap_real) = len(fmap_fake) = disc.num_discriminators = 3
        logits_fake (_type_): logits_fake is the list of every sub discriminator output of the Multi discriminator 
            logits_fake, _ = disc_model(model(input_wav)[0].detach())
        fmap_fake (_type_): fmap_fake is the output of the discriminator when the input is the fake audio.
            fmap_fake = disc_model(model(input_wav)[0]) = disc_model(reconstructed_audio)
        input_wav (tensor): input_wav is the input audio of the generator (GT audio)
        output_wav (tensor): output_wav is the output of the generator (output = model(input_wav)[0])
        sample_rate (int, optional): Defaults to 24000.

    Returns:
        loss: total loss
    """
    l2Loss = torch.nn.MSELoss(reduction='mean')
    # Collect losses as defined in paper for use with balancer
    # l_g - adversarial loss for the generator
    # l_feat - relative feature matching loss for the generator
    # l_t = torch.tensor([0.0], device='cuda', requires_grad=True)
    # l_f = torch.tensor([0.0], device='cuda', requires_grad=True)
    # l_g = torch.tensor([0.0], device='cuda', requires_grad=True)
    # l_feat = torch.tensor([0.0], device='cuda', requires_grad=True)

    #time domain loss, output_wav is the output of the generator
    # l_t = l1Loss(input, output)
    l_t = l2Loss(input, output)

    #generator loss and feat loss, D_k(\hat x) = logits_fake[k], D_k^l(x) = fmap_real[k][l], D_k^l(\hat x) = fmap_fake[k][l]
    # l_g = \sum max(0, 1 - D_k(\hat x)) / K, K = disc.num_discriminators = len(fmap_real) = len(fmap_fake) = len(logits_fake) = 3
    # l_feat = \sum |D_k^l(x) - D_k^l(\hat x)| / |D_k^l(x)| / KL, KL = len(fmap_real[0])*len(fmap_real)=3 * 5
    # for tt1 in range(len(fmap_real)): # len(fmap_real) = 3
    #     l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
    #     for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
    #         # l_feat = l_feat + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach()))
    #         l_feat = l_feat + l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))

    # KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
    # l_feat /= KL_scale
    # K_scale = len(fmap_real) # len(fmap_real[0]) = len(fmap_fake[0]) == L
    # l_g /= K_scale

    return {
        'l_t': l_t,
        # 'l_f': l_f,
        # 'l_g': l_g,
        # 'l_feat': l_feat,
    }

# def disc_loss(logits_real, logits_fake):
#     """This function is used to compute the loss of the discriminator.
#         l_d = \sum max(0, 1 - D_k(x)) + max(0, 1 + D_k(\hat x)) / K, K = disc.num_discriminators = len(logits_real) = len(logits_fake) = 3
#     Args:
#         logits_real (List[torch.Tensor]): logits_real = disc_model(input_wav)[0]
#         logits_fake (List[torch.Tensor]): logits_fake = disc_model(model(input_wav)[0])[0]

#     Returns:
#         lossd: discriminator loss
#     """
#     relu = torch.nn.ReLU()
#     lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
#     for tt1 in range(len(logits_real)):
#         lossd = lossd + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1]))
#     lossd = lossd / len(logits_real)
#     return lossd