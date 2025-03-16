import torch
from torch.nn import functional as F
import sys

# https://github.com/ZhikangNiu/encodec-pytorch/blob/main/losses.py

def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=10):
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
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='none')
    l2Loss = torch.nn.MSELoss(reduction='none')

    l_g = 0
    l_feat = 0

    #time domain loss, output_wav is the output of the generator
    l_t = l1Loss(input_wav, output_wav).mean(dim=(1,2))
    l_t_2 = l2Loss(input_wav, output_wav).mean(dim=(1,2))
    l1 = torch.nn.L1Loss(reduction='mean')(input_wav, output_wav)
    l2 = torch.nn.MSELoss(reduction='mean')(input_wav, output_wav)

    #generator loss and feat loss, D_k(\hat x) = logits_fake[k], D_k^l(x) = fmap_real[k][l], D_k^l(\hat x) = fmap_fake[k][l]
    # l_g = \sum max(0, 1 - D_k(\hat x)) / K, K = disc.num_discriminators = len(fmap_real) = len(fmap_fake) = len(logits_fake) = 3
    # l_feat = \sum |D_k^l(x) - D_k^l(\hat x)| / |D_k^l(x)| / KL, KL = len(fmap_real[0])*len(fmap_real)=3 * 5

    if fmap_real is not None:
        for tt1 in range(len(fmap_real)): # len(fmap_real) = num discriminators 
            # l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) #/ len(logits_fake) 
            # l_g = l_g + torch.mean(relu(1-sigmoid(logits_fake[tt1]))) / len(logits_fake)
            l_g = l_g + torch.mean((1 - logits_fake[tt1]) ** 2)  #squared generator loss
            for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
                l_feat = l_feat + torch.nn.L1Loss(reduction='mean')(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach()))
                # l_feat = l_feat + l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))
                # l_feat = l_feat + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))

        KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
        l_feat /= KL_scale
        K_scale = len(fmap_real) # len(fmap_real[0]) = len(fmap_fake[0]) == L
        l_g /= K_scale
    else:
        l_g = torch.tensor([0.0], device='cuda', requires_grad=False)
        l_feat = torch.tensor([0.0], device='cuda', requires_grad=False)

    return {
        'l_t': l_t,
        'l_t_2': l_t_2,
        'l_1': l1,
        'l_2': l2,
        'l_g': l_g,
        'l_feat': l_feat,
    }

def disc_loss(logits_real, logits_fake):
    """This function is used to compute the loss of the discriminator.
        l_d = \sum max(0, 1 - D_k(x)) + max(0, 1 + D_k(\hat x)) / K, K = disc.num_discriminators = len(logits_real) = len(logits_fake) = 3
    Args:
        logits_real (List[torch.Tensor]): 
            B x 1 x T x filters*2
            logits_real = disc_model(input_wav)[0]  
        logits_fake (List[torch.Tensor]): logits_fake = disc_model(model(input_wav)[0])[0]

    Returns:
        lossd: discriminator loss
    """
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()
    loss_d = 0

    for tt1 in range(len(logits_real)): #[1,1, T, filters*2]
        # loss_d = loss_d + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1])) #Encodec
        # loss_d = loss_d + torch.mean(relu(1-sigmoid(logits_real[tt1]))) + torch.mean(relu(sigmoid(logits_fake[tt1]))) #our own?
        # loss_d = loss_d + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(logits_fake[tt1])) #without sigmoid and 1+
        loss_d = loss_d + torch.mean(logits_fake[tt1] ** 2) + torch.mean((1 - logits_real[tt1]) ** 2) #DAC squared loss
    loss_d = loss_d / len(logits_real)
    return loss_d

#Ok so previously the discriminator was making logits_fake -0.7 ish, logits_real -0.4 ish which works  in that it discriminator loss goes down BUT 
#it got stuck once the generator loss kicked it because generator loss wants logits_fake to be 1 (>0) at least, and so it does do that.
#but then, it gets stuck.
#I changed the losses to make more intuitive sense 
