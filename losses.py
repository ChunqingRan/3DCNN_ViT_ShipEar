import torch
import torch.nn as nn


class JointLoss(nn.Module):
    """
    联合损失函数：分类 Loss + 域对抗 Loss
    """

    def __init__(self, lambda_domain=0.1):
        super(JointLoss, self).__init__()
        self.lambda_domain = lambda_domain

        # 主任务分类损失
        self.cls_criterion = nn.CrossEntropyLoss()

        # 域分类损失 (通常也是 CrossEntropy，或者是 BCE)
        # 这里假设 num_domains >= 2，用 CrossEntropy
        self.domain_criterion = nn.CrossEntropyLoss()

    def forward(self, cls_out, target_cls, domain_out=None, target_domain=None):
        """
        cls_out: (Batch, Num_Classes)
        target_cls: (Batch,)
        domain_out: (Batch, Num_Domains) [可选]
        target_domain: (Batch,) [可选]
        """
        # 1. 计算分类损失
        loss_cls = self.cls_criterion(cls_out, target_cls)

        # 2. 计算域损失 (仅在提供域标签且训练时)
        loss_domain = 0.0
        if domain_out is not None and target_domain is not None:
            loss_domain = self.domain_criterion(domain_out, target_domain)

        # 3. 总损失
        total_loss = loss_cls + self.lambda_domain * loss_domain

        return total_loss, loss_cls, loss_domain