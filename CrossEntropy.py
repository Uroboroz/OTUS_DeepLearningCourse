import torch
import torch.nn.functional as F


def CrossEntropy(input, target):
    m = target.shape[0]
    max_val = (-input).clamp(min=0)
    loss = input - input * torch.Tensor([range(m), target]) + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    return loss.mean()


o = torch.Tensor([[200., -200.], [400., -400.]])

t = torch.Tensor([1, 1]).to(torch.long)

print(CrossEntropy(o, t), F.nll_loss(F.log_softmax(o, dim=1), t))
