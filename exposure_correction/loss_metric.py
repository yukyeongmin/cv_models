import tensorflow as tf


class Grad_Loss_L1_MS(tf.Module):
    def __init__(self):
        super().__init__(self)
        self.grad_loss_l1 = SpatialGradientLoss(loss_func=l1, lam=0.01)
        self.grad_loss_ms_l1_g = Multi_Scale(loss_func=self.grad_loss_l1,
                                             num_levels_in_an_octave=num_levels_in_an_octave,
                                             starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                             enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)

    def forward(self, x, y):
        return self.grad_loss_ms_l1_g(x, y)


class L1_MS(tf.Module):
    def __init__(self):
        super().__init__(self)
        self.l1_ms_g = Multi_Scale(loss_func=l1, num_levels_in_an_octave=num_levels_in_an_octave,
                                   starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                   enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)

    def forward(self, x, y):
        return self.l1_ms_g(x, y)


class L2_MS(tf.Module):
    def __init__(self):
        super(L2_MS, self).__init__()
        self.l2_ms_g = Multi_Scale(loss_func=l2, num_levels_in_an_octave=num_levels_in_an_octave,
                                   starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                   enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)

    def forward(self, x, y):
        return self.l2_ms_g(x, y)


class DQ_MS(tf.Module):
    def __init__(self):
        super(DQ_MS, self).__init__()
        self.dq = dq(enhanced_penalty=False, multiple_moments=multiple_moments, alpha=alpha)
        self.dq_ms_g = Multi_Scale(loss_func=self.dq, num_levels_in_an_octave=num_levels_in_an_octave,
                                   starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                   enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)

    def forward(self, x, y):
        return self.dq_ms_g(x, y)


class DQ_MS_Gradient(tf.Module):
    def __init__(self):
        super(DQ_MS_Gradient, self).__init__()
        self.dq = dq(enhanced_penalty=False, multiple_moments=multiple_moments, alpha=alpha)
        self.grad_loss_dq = SpatialGradientLoss(loss_func=self.dq, lam=0.01)
        self.grad_loss_ms_dq_g = Multi_Scale(loss_func=self.grad_loss_dq,
                                             num_levels_in_an_octave=num_levels_in_an_octave,
                                             starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                             enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)

    def forward(self, x, y):
        return self.grad_loss_ms_dq_g(x, y)


class VGG_MSE(tf.Module):  # NOTE, THE IMPLEMENTATION IN THE ORIGINAL SR_RESNET WAS SLIGHTLY DIFFERENT. THEY FIRST CALLED LOSS.BACKWARD() ON THE MSE LOSS AND THEN CALLED LOSS.BACKWARD() ON THE VGG LOSS
    def __init__(self):
        super(VGG_MSE, self).__init__()
        self.mse = nn.MSELoss()
        self.vgg = VGGPerceptualLoss(loss_func=self.mse)
        self.lambda_tradeoff = 0.01  # lambda_tradeoff

    def forward(self, x, y):
        # print("gere")
        return self.lambda_tradeoff * self.vgg(x, y) + self.mse(x, y)


class VGG_only(tf.Module):  # NOTE, THE IMPLEMENTATION IN THE ORIGINAL SR_RESNET WAS SLIGHTLY DIFFERENT. THEY FIRST CALLED LOSS.BACKWARD() ON THE MSE LOSS AND THEN CALLED LOSS.BACKWARD() ON THE VGG LOSS
    def __init__(self):
        super(VGG_only, self).__init__()
        self.mse = nn.MSELoss()
        self.vgg = VGGPerceptualLoss(loss_func=self.mse)

    def forward(self, x, y):
        return self.vgg(x, y)


class LPIPS_MSE(tf.Module):
    def __init__(self):
        super(LPIPS_MSE, self).__init__()
        self.lpips = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        self.mse = nn.MSELoss()
        self.lambda_tradeoff = 1.0  # lambda_tradeoff

    def forward(self, x, y):
        return self.lambda_tradeoff * self.lpips(x, y) + self.mse(x, y)


class MSSSIM_L1(tf.Module):
    def __init__(self):
        super(MSSSIM_L1, self).__init__()
        self.msssim = MSSSIM()
        self.l1 = torch.nn.L1Loss().type(dtype)

    def forward(self, x, y):
        return 0.84 * self.msssim(x, y) + 0.16 * self.l1(x, y)


class L2_DQ_MS(tf.Module):
    def __init__(self):
        super(L2_DQ_MS, self).__init__()
        self.dq = dq(enhanced_penalty=False, multiple_moments=multiple_moments, alpha=alpha)
        self.l2_ms_g = Multi_Scale(loss_func=l2, num_levels_in_an_octave=num_levels_in_an_octave,
                                   starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                   enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)
        self.dq_ms_g = Multi_Scale(loss_func=self.dq, num_levels_in_an_octave=num_levels_in_an_octave,
                                   starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                   enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)

    def forward(self, x, y):
        return 0.25 * self.l2_ms_g(x, y) + 0.75 * self.dq_ms_g(x, y)


class DQ_MS_Gradient_DQ_MS(nn.Module):
    def __init__(self):
        super(DQ_MS_Gradient_DQ_MS, self).__init__()
        self.dq = dq(enhanced_penalty=False, multiple_moments=multiple_moments, alpha=alpha)
        self.grad_loss_dq = SpatialGradientLoss(loss_func=self.dq, lam=0.01)
        self.grad_loss_ms_dq_g = Multi_Scale(loss_func=self.grad_loss_dq,
                                             num_levels_in_an_octave=num_levels_in_an_octave,
                                             starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                             enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)
        self.dq_ms_g = Multi_Scale(loss_func=self.dq, num_levels_in_an_octave=num_levels_in_an_octave,
                                   starting_sigma=starting_sigma, pyramid_type='Gaussian',
                                   enhanced_penalty=enhanced_penalty, lam=lam, alpha=alpha)

    def forward(self, x, y):
        return 0.9 * self.dq_ms_g(x, y) + 0.1 * self.grad_loss_ms_dq_g(x, y)


class SuperLoss(tf.Module):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__()

        loss_object = None
        if loss_name == "ssim":
            loss_object = SSIM()
        elif loss_name == "ssim_luminance":
            loss_object = SSIM_split(component='luminance')
        elif loss_name == "msssim":
            loss_object = MSSSIM()
        elif loss_name == "dq":
            loss_object = dq(
                enhanced_penalty=False, multiple_moments=multiple_moments,
                alpha=alpha
            )
        elif loss_name == "lpips":
            loss_object = models.PerceptualLoss(
                model='net-lin', net='alex', use_gpu=True, gpu_ids=[0]
            )
        elif loss_name == "flip":
            loss_object = FLIPLoss()
        elif loss_name == "grad_loss_l1":
            loss_object = SpatialGradientLoss(loss_func=l1, lam=0.01)
        elif loss_name == "l1":
            loss_object = torch.nn.L1Loss().type(dtype)
        elif loss_name == "l2":
            loss_object = torch.nn.MSELoss().type(dtype)
        elif loss_name == "fsim":
            loss_object = FSIM()
        elif loss_name == "fsimc":
            loss_object = FSIMc()
        elif loss_name == "vgg_mse":
            loss_object = VGG_MSE()
        elif loss_name == "lpips_mse":
            loss_object = LPIPS_MSE(*args, **kwargs)
        elif loss_name == "vgg":
            loss_object = VGG_only()
        # Our losses
        elif loss_name == "singan":
            loss_object = SinGANLoss(*args, **kwargs)
        elif loss_name == 'msssim_l1':
            loss_object = MSSSIM_L1()
        else:
            "Loss is not defined yet"

        self.loss_object = loss_object

    def forward(self, *args, **kwargs):
        res = self.loss_object(*args, **kwargs)

        return res
