import torch
import torch.nn.functional as F

""" multi-class """

# logits = torch.randn(2, 3)
logits = torch.tensor(
    [[-1.4710, 0.5296, 0.3145], [1.6896, -0.5447, 0.4537]], dtype=torch.float32
)
sparse_labels = torch.tensor([2, 0])
raw_ce = F.cross_entropy(logits, sparse_labels, reduction="none")
raw_ce_ = F.nll_loss(F.log_softmax(logits, dim=-1), sparse_labels, reduction="none")

torch.testing.assert_close(raw_ce, raw_ce_)

labels = F.one_hot(sparse_labels, num_classes=logits.size(1))

log_probability = F.log_softmax(logits, dim=-1)
log_probability_ = torch.log(F.softmax(logits, dim=-1))

torch.testing.assert_close(log_probability, log_probability_)

raw_ce__ = (-labels * log_probability).sum(dim=-1)

torch.testing.assert_close(raw_ce, raw_ce__)
print(raw_ce)
