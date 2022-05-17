import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from pl_bolts.models.self_supervised.evaluator import Flatten


def dice_loss(input, target):
    # this loss function need input in the range (0, 1), and target in (0, 1)
    smooth = 0.01

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def focal_loss(input, target, alpha, gamma, eps=1e-6):
    # this loss function need input in the range (0, 1), and target in (0, 1)
    input = input.view(-1, 1)
    input = torch.clamp(input, min=eps, max=1-eps)
    target = target.view(-1, 1)
    loss = -target * alpha * ((1 - input) ** gamma) * torch.log(input) - (1 - target) * (1-alpha) * (
                input ** gamma) * torch.log(1 - input)
    return loss.mean()

class Projection(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=48, output_dim=32):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))
    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

def nt_xent_loss(out_1, out_2, temperature):
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    neg[neg<1e-6] = 1.0

    loss = -torch.log(pos / neg).mean()
    return loss

class FocalLoss(nn.Module):
    # this loss function need input in the range (-1, 1), and target in (0, 1)
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        input = torch.unsqueeze(input, -1)
        input = torch.cat([0 - input, input], -1)
        input = input.contiguous().view(-1, 2)
        target = target.view(-1, 1).long()

        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()
