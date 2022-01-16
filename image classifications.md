# Training scheme
## Simple Augmentations to use during training
* Randomly crop a rectangular area with [3/4, 4/3] aspect ratio, then randomly sample with [8%, 100%], and finally resize to an `img_size x img_size` square, do this randomly on each batch.
* Add some probability of random horizontal flip
* Change randomly hue, saturation, brightness and choose coefficients proportional to [0:6, 1:4]
* Normalize the RGB channels by subtracting each channel by the mean value and dividing by it's std.
## During test
* Keep the same aspect ratio, resize the short side to `img_size` and then cut out the `img_size-1 x img_size-1` area in the center.
* Standardize as above.
## Weight initialization
* Init the convolutions and fully connected layers with the `Xavier weights`.
* Init biases to 0
* BatchNorms: init alpha and beta to 1 and 0
* lr scheduler: init it to 0.1, and divided by 10 every couple of epochs.

# Effective training
## Batch size
* If the batch size is too large, it will affect the convergence speed and reduce the accuracy. If the batch size is too small, it will never finish.
## LR scheduler
* Increasing the batch size will not affect the gradient itself, but it will reduce the variance, that is, reduce the noise. Use LR schedular to compensate for larger batch size.
* If baseline batch size is `256` and initial lr is `0.1`, so when increasing `b`, should change lr to `0.1 x b / 256`.
## LR warm-up
* Because the weights are random at the beginning, large learning rate will lead to unstable training on the first batches. Therefore, a warm-up phase is advised: use some data `m batches`, such as 5 epochs, to increase the learning rate from 0 to the initial learning rate. This simple tricks works well often.
## Fine-tunning
* After you are done training: train the last layer for couple more episode when all the other layers are frozen.
## Use Learning rate cosine decay
## Label smoothing
* The deal is simple: y_true = y_true * (1.0 - $\epsilon$) + 0.5 * $\epsilon$ [example: $\epsilon = 0.001$]
* Built-in tensorflow:
```
loss = BinaryCrossentropy(label_smoothing = label_smoothing)
```
* For pytorch:
```
from torch.nn.modules.loss import _WeightedLoss

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight = None, reduction = 'mean', smoothing = 0.0, pos_weight = None):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets, n_labels, smoothing = 0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad(): targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight, pos_weight = self.pos_weight)
        if  self.reduction == 'sum': loss = loss.sum()
        elif  self.reduction == 'mean': loss = loss.mean()
        return loss
```