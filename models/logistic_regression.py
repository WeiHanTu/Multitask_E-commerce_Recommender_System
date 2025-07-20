import torch
from torch import nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, task_num):
        super(LogisticRegressionModel, self).__init__()
        self.task_num = task_num
        self.linear = nn.Linear(input_dim, task_num)

    def forward(self, x):
        logits = self.linear(x)
        outputs = [torch.sigmoid(logits[:, i]) for i in range(self.task_num)]
        return outputs
