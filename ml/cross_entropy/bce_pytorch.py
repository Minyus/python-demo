import torch
import torch.nn.functional as F

""" multi-labels (binary) """

logits = torch.randn(2)
labels = torch.tensor([0.0, 1.0], dtype=torch.float32)

raw_bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
raw_bce_ = F.binary_cross_entropy(F.sigmoid(logits), labels, reduction="none")

torch.testing.assert_close(raw_bce, raw_bce_)

log_probability = F.logsigmoid(logits)
log_probability_ = torch.log(F.sigmoid(logits))
log_probability_neg = F.logsigmoid(-logits)
log_probability_neg_ = torch.log(1 - F.sigmoid(logits))

torch.testing.assert_close(log_probability, log_probability_)
torch.testing.assert_close(log_probability_neg, log_probability_neg_)

raw_bce__ = -labels * log_probability - (1 - labels) * log_probability_neg

torch.testing.assert_close(raw_bce, raw_bce__)
