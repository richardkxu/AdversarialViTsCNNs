import torch 
import torch.nn as nn 
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class NoOpAttacker:
    def attack(self, image, label):
        return image, -torch.ones_like(label)


class PGDAttacker:
    def __init__(self, num_iter, epsilon, step_size, image_scale, prob_start_from_clean=0.0, loss_scaler=None, use_amp=False):
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * image_scale / 255.0
        self.step_size = step_size * image_scale / 255.0
        self.prob_start_from_clean = prob_start_from_clean
        self.loss_scaler = loss_scaler
        self.use_amp = use_amp

    def attack(self, image_clean, label, model, train=True, mixup=False):

        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1., max=1.)

        # rand restart
        init_start = torch.empty_like(image_clean).uniform_(-self.epsilon, self.epsilon)
        start_from_noise_index = torch.randn([], device=init_start.device) > self.prob_start_from_clean
        adv = image_clean + start_from_noise_index * init_start
        # adv = torch.where(adv > lower_bound, adv, lower_bound)
        # adv = torch.where(adv < upper_bound, adv, upper_bound).detach()
        
        for i in range(self.num_iter):
            adv.requires_grad = True
            pred_logits = model(adv)

            if train:
                if mixup:
                    criterion = SoftTargetCrossEntropy()
                    losses = criterion(pred_logits, label)
                else:
                    losses = F.cross_entropy(pred_logits, label)
            else:
                losses = F.cross_entropy(pred_logits, label)

            if self.use_amp:
                with torch.cuda.amp.autocast(enabled=False):
                    scaled_grad_adv = torch.autograd.grad(self.loss_scaler._scaler.scale(losses),
                                                          adv, retain_graph=False, create_graph=False)[0]
                    # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
                    # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
                    g = scaled_grad_adv / self.loss_scaler._scaler.get_scale()
            else:
                g = torch.autograd.grad(losses, adv, retain_graph=False, create_graph=False)[0]

            # projection
            adv = adv + torch.sign(g) * self.step_size
            adv = torch.where(adv > lower_bound, adv, lower_bound)
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()
        
        return adv, label
