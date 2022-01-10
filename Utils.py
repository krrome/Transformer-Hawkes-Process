import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    # calculate differences between time stamps
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    # Add dimension at the end (batch_size, seq_len) and then shorten the time difference by a uniform random
    # amount (by division) --> gives (batch_size, seq_len, num_samples)
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    # Now turn 'temp_time' into the MC-randomised "current" factor in Eq6. Why devision by tj + 1 and not tj is
    # unclear to me...
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    # fetch the history + base (Eq6) part starting form the 2nd time slot
    temp_hid = model.linear(data)[:, 1:, :]
    # keep only the model history and base elements of the intensity for the correct type!
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    # now calculate the intensity given the randomly sampled time stamps
    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    # calculate the average intensity across random samples
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence.
        Caveats and thoughts:
         - The loss function looks only at the time embeddings calculated for the given types, but ignores the values
            for all other types!
         - Unclear why compute_integral_unbiased adds +1 to the times (worried about a 0 timestamp?)
    """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    # create a binary tensor indicating the type of events with dimension (batch_size, sequence length, number of types)
    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    # the embedding to predictions of event types using a linear layer ("w" in Eq6):
    all_hid = model.linear(data)
    # calculate softplus likelyhood at tj ´sequence-length´ (ignore interpolations of t in Eq6)
    all_lambda = softplus(all_hid, model.beta)
    # for every timeslot in the series only keep the lambda of the true type, then aggregate over all types.
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood (1st half of Eq 8)
    # basically just log(type_lambda) and set unused time slots of the results of the series to 0
    event_ll = compute_event(type_lambda, non_pad_mask)
    # sum over all events
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing.
        Thoughts: ignore types that happen after the end of the series - it doesn't necessarily make sense to predict
            an end - LabelSmoothingLoss does this already
        LabelSmoothingLoss reduces the contrast of the 1-hot encoded positive and negative values.
    """
    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss.
        Thoughts: ignore types that happen after the end of the series - it doesn't necessarily make sense to predict
            an end
    """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
