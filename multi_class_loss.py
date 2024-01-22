import torch.nn as nn

bce_loss = nn.BCELoss(reduction='mean')


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, target, input):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        loss = (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

        loss = 1 - loss.sum()

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self, weight=None):
        super(MulticlassDiceLoss, self).__init__()
        self.__name__ = 'dice_loss + bce_loss'
        self.ce = nn.BCELoss(reduce=True, weight=None)
        self.dice = DiceLoss()
        self.weight = weight

    def forward(self, input, target, ):
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        ce = self.ce(input, target)
        C = input.shape[-1]
        input = input.contiguous().view(-1, C)
        target = target.contiguous().view(-1, C)

        dice = 0

        for i in range(C):
            dice_i = self.dice(input[:, i], target[:, i])
            dice += dice_i

        if self.weight is None:
            loss = ce + dice
        else:
            loss = ce + dice * self.weight
        return loss
