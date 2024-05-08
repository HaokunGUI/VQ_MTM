import torch

def masked_mae_loss(y_pred, y_true, mask_val=0.):
    """
    Only compute loss on unmasked part
    """
    masks = (y_true != mask_val).float()
    masks /= masks.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_mse_loss(y_pred, y_true, mask_val=0.):
    """
    Only compute MSE loss on unmasked part
    """
    masks = (y_true != mask_val).float()
    masks /= masks.mean()
    loss = (y_pred - y_true).pow(2)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    loss = torch.sqrt(torch.mean(loss))
    return loss


def compute_regression_loss(
        y_true,
        y_predicted,
        standard_scaler=None,
        loss_fn='mae',
        mask_val=0.,
        is_tensor=True):
    """
    Compute masked MAE loss with inverse scaled y_true and y_predict
    Args:
        y_true: ground truth signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        y_predicted: predicted signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        standard_scaler: class StandardScaler object
        device: device
        mask: int, masked node ID
        loss_fn: 'mae' or 'mse'
        is_tensor: whether y_true and y_predicted are PyTorch tensor
    """

    if standard_scaler is not None:
        y_true = standard_scaler.inverse_transform(y_true,
                                                   is_tensor=is_tensor)

        y_predicted = standard_scaler.inverse_transform(y_predicted,
                                                        is_tensor=is_tensor)

    if loss_fn == 'mae':
        return masked_mae_loss(y_predicted, y_true, mask_val=mask_val)
    else:
        return masked_mse_loss(y_predicted, y_true, mask_val=mask_val)


class loss_fn():
    def __init__(self, standard_sclar, loss_fn='mae', is_tensor=True, mask_val=0.):
        self.standard_scaler = standard_sclar
        self.loss_fn = loss_fn
        self.is_tensor = is_tensor
        self.mask_val = mask_val

    def __call__(self, y_true, y_predicted):
        return compute_regression_loss(
            y_true=y_true,
            y_predicted=y_predicted,
            standard_scaler=self.standard_scaler,
            loss_fn=self.loss_fn,
            mask_val=self.mask_val,
            is_tensor=self.is_tensor
        )