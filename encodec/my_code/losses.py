import torch
# from audio_to_mel import Audio2Mel
from torch.nn import functional as F
import sys

# https://github.com/ZhikangNiu/encodec-pytorch/blob/main/losses.py

def loss_fn_l1(input, output):
    # l1Loss = torch.nn.L1Loss(reduction='mean')
    # return l1Loss(input, output)
    return F.l1_loss(input, output, reduction='mean')

def loss_fn_l2(input, output):
    # l2Loss = torch.nn.MSELoss(reduction='mean')
    # return l2Loss(input, output)
    return F.mse_loss(input, output, reduction='mean')

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
    # Collect losses as defined in paper for use with balancer
    # l_t - L1 distance between the target and compressed audio over the time domain
    # l_f - linear combination between the L1 and L2 losses over the mel-spectrogram using several time scales
    # l_g - adversarial loss for the generator
    # l_feat - relative feature matching loss for the generator
    # l_t = torch.tensor([0.0], device='cuda', requires_grad=True)
    # l_f = torch.tensor([0.0], device='cuda', requires_grad=True)
    # l_g = torch.tensor([0.0], device='cuda', requires_grad=True)
    # l_feat = torch.tensor([0.0], device='cuda', requires_grad=True)

    l_g = 0
    l_feat = 0

    #time domain loss, output_wav is the output of the generator
    l_t = l1Loss(input_wav, output_wav).mean(dim=(1,2))
    l_t_2 = l2Loss(input_wav, output_wav).mean(dim=(1,2))
    l1 = torch.nn.L1Loss(reduction='mean')(input_wav, output_wav)
    l2 = torch.nn.MSELoss(reduction='mean')(input_wav, output_wav)
    sigmoid = torch.nn.Sigmoid()

    #frequency domain loss, window length is 2^i, hop length is 2^i/4, i \in [5,11]. combine l1 and l2 loss
    # for i in range(5, 12): #e=5,...,11
    #     fft = Audio2Mel(n_fft=2 ** i,win_length=2 ** i, hop_length=(2 ** i) // 4, n_mel_channels=64, sampling_rate=sample_rate)
    #     l_f = l_f + l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))

    #generator loss and feat loss, D_k(\hat x) = logits_fake[k], D_k^l(x) = fmap_real[k][l], D_k^l(\hat x) = fmap_fake[k][l]
    # l_g = \sum max(0, 1 - D_k(\hat x)) / K, K = disc.num_discriminators = len(fmap_real) = len(fmap_fake) = len(logits_fake) = 3
    # l_feat = \sum |D_k^l(x) - D_k^l(\hat x)| / |D_k^l(x)| / KL, KL = len(fmap_real[0])*len(fmap_real)=3 * 5

    if fmap_real is not None:
        for tt1 in range(len(fmap_real)): # len(fmap_real) = num discriminators 
            # print(f'generator logits_fake: {torch.mean(logits_fake[tt1])}')
            # l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) #/ len(logits_fake) #min is 0 if logits_fake is 1
            #our own 
            # l_g = l_g + torch.mean(relu(1-sigmoid(logits_fake[tt1]))) / len(logits_fake)
                # 0 .2 means logits_fake is -0.8

            l_g = l_g + torch.mean((1 - logits_fake[tt1]) ** 2)  
                #min is 0 if logits_fake is 1
                #logits_fake 0.48 means l_g is 0.52
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

    # print(f"l_t: {l_t}, l_g: {l_g}, l_feat: {l_feat}")
    # TODO: l_t does not have gradients???

    return {
        'l_t': l_t,
        'l_t_2': l_t_2,
        'l_1': l1,
        'l_2': l2,
        # 'l_f': l_f,
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
    # lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    loss_d = 0
    for tt1 in range(len(logits_real)): #[1,1, T, filters*2]

        # print(f'real: {torch.mean(relu(1-logits_real[tt1]))}') #1.4 means logits_real is -0.4, 0 means logits_real is 1
        #     #min 0 if logits_real is 1
        # print(f'fake: {torch.mean(relu(1+logits_fake[tt1]))}') #0.3 means logits_fake is -0.7
            #min 0 if logits_fake is 0
        # loss_d = loss_d + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1]))
        # print(f'real: {torch.mean(logits_real[tt1])}') #0 means logits_real is 1
        # print(f'fake: {torch.mean(logits_fake[tt1])}') #0 means logits_fake is 0

        # loss_d = loss_d + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1])) #Encodec
            # real: 1.4854092597961426  #1.4 means logits_real is -0.4, 0 means logits_real is 1
            # fake: 0.3382878005504608  #0.3 means logits_fake is -0.7
            # real: 1.5711915493011475
            # fake: 0.2277427315711975
        # loss_d = loss_d + torch.mean(relu(1-sigmoid(logits_real[tt1]))) + torch.mean(relu(sigmoid(logits_fake[tt1]))) #our own?
            # real: 143.40602111816406  #0 and 1
            # fake: 142.12840270996094
            # real: 296.2453918457031
            # fake: 290.69921875
        
        # loss_d = loss_d + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(logits_fake[tt1])) #without sigmoid and 1+
            # real: 0.1342935413122177
            # fake: 0.10814659297466278
            # real: 0.22330671548843384
            # fake: 0.19009989500045776

            #after training for longer until l_d= 0.6
            # real: 0.11632872372865677
            # fake: 0.09444870799779892
            # real: 0.8395664095878601
            # fake: 0.059129368513822556

        loss_d = loss_d + torch.mean(logits_fake[tt1] ** 2) + torch.mean((1 - logits_real[tt1]) ** 2) #DAC
        #Cross Entropy value error
    
            # real: 0.5131673812866211, 0.25577738881111145
            # fake: 0.4625246226787567, 0.21392902731895447
            # real: 0.5093390941619873, 0.24603983759880066
            # fake: 0.46865516901016235, 0.2196376621723175
    loss_d = loss_d / len(logits_real)
    return loss_d

#Ok so previously the discriminator was making logits_fake -0.7 ish, logits_real -0.4 ish which works  in that it discriminator loss goes down BUT 
#it got stuck once the generator loss kicked it because generator loss wants logits_fake to be 1 (>0) at least, and so it does do that.
#but then, it gets stuck.
#I changed the losses to make more intuitive sense 

# def discriminator_loss(self, fake, real):
#         d_fake, d_real = self.forward(fake.clone().detach(), real)

#         loss_d = 0
#         for x_fake, x_real in zip(d_fake, d_real):
#             loss_d += torch.mean(x_fake[-1] ** 2)
#             loss_d += torch.mean((1 - x_real[-1]) ** 2)
#         return loss_d

#     def generator_loss(self, fake, real):
#         d_fake, d_real = self.forward(fake, real)

#         loss_g = 0
#         for x_fake in d_fake:
#             loss_g += torch.mean((1 - x_fake[-1]) ** 2)

#         loss_feature = 0

#         for i in range(len(d_fake)):
#             for j in range(len(d_fake[i]) - 1):
#                 loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
#         return loss_g, loss_feature