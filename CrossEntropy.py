import torch
import torch.nn.functional as F


def CrossEntropy(input, target):
    exps = []
    for i in input:
        exps.append(i - i.max() - (((i - i.max()).exp()).sum() + 0.00001).log())
    print(torch.stack(exps))
    return F.nll_loss(torch.stack(exps), target)


o = torch.Tensor([[100., -100.], [400., -400.]])

t = torch.Tensor([1, 1]).to(torch.long)

print(CrossEntropy(o, t), F.nll_loss(F.log_softmax(o, dim=1), t))
