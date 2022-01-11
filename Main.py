import argparse
from datetime import datetime
import json
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import tensorboard
import socket
import subprocess

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader, get_dataloader_h5
from transformer.Models import Transformer
from tqdm import tqdm

tb_writer = None


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(os.path.join(opt.data.rstrip("/"), 'train.pkl'), 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(os.path.join(opt.data.rstrip("/"), 'dev.pkl'), 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(os.path.join(opt.data.rstrip("/"), 'test.pkl'), 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def prepare_dataloader_h5(opt):
    """ Load data and prepare dataloader. """
    import h5py

    train_path = os.path.join(opt.data.rstrip("/"), 'train.h5')
    test_path = os.path.join(opt.data.rstrip("/"), 'test.h5')

    trainloader = get_dataloader_h5(train_path, opt.batch_size, shuffle=True)
    testloader = get_dataloader_h5(test_path, opt.batch_size, shuffle=False)

    with h5py.File(train_path, "r") as fh:
        num_types = int(fh["num_types"][()])

    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt, epoch_i):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    load_times = []
    train_times = []
    losses = {"event_loss": [], "pred_loss": [], "se_loss": []}
    pre_load_time = time.time()
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        pre_train_time = time.time()
        """ prepare data """
        event_time, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        # if isinstance(tb_writer, tensorboard.SummaryWriter):
        #     if epoch_i == 0:
        #         tb_writer.add_graph(model, (event_time, event_type))
        #         tb_writer.close()

        """ backward """
        # negative log-likelihood
        if opt.ll_loss_factor > 0:
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
        else:
            event_loss = None

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.get_time_loss_fn(opt.time_loss_fn)(prediction[1], event_time, event_type)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = opt.time_loss_scaler
        if opt.ll_loss_factor > 0:
            loss = event_loss * opt.ll_loss_factor + pred_loss + se / scale_time_loss
        else:
            loss = pred_loss + se / scale_time_loss
        if isinstance(tb_writer, tensorboard.SummaryWriter):
            if event_loss is not None:
                losses["event_loss"].append(event_loss.cpu().item())
            losses["pred_loss"].append(pred_loss.cpu().item())
            losses["se_loss"].append(se.cpu().item() / scale_time_loss)

        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        if event_loss is not None:
            total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
        post_train_time = time.time()
        load_times.append(pre_train_time - pre_load_time)
        train_times.append(post_train_time - pre_train_time)
        pre_load_time = time.time()

    avg_load_time = sum(load_times) / len(load_times)
    avg_train_time = sum(train_times) / len(train_times)
    print(f"Avg load time: {avg_load_time}, avg train time: {avg_train_time}")

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, avg_load_time, \
           avg_train_time, losses


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.get_time_loss_fn(opt.time_loss_fn)(prediction[1], event_time, event_type)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    best_event_ll = -99999
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time, avg_load_time, avg_train_time, losses = train_epoch(model, training_data,
                                                                                         optimizer, pred_loss_func, opt,
                                                                                         epoch_i)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))
        model_saved = False
        if valid_event > best_event_ll:
            best_event_ll = valid_event
            torch.save(model, os.path.join(opt.log_path, "best_model.pt"))
            print("--(Best model saved)---")
            model_saved = True

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        summary_dict = {"avg_batch_load_time": avg_load_time, "avg_batch_train_time": avg_train_time,
                        "valid_ll": valid_event, "valid_acc_type": valid_type, "valid_rmse_time": valid_time,
                        "train_ll": train_event, "train_acc_type": train_type, "train_rmse_time": train_time}

        log_time = datetime.now()
        if isinstance(tb_writer, tensorboard.SummaryWriter):
            [tb_writer.add_scalar(k, v, epoch_i, log_time.timestamp()) for k, v in summary_dict.items()]
            [tb_writer.add_histogram(k, np.array(v), epoch_i, walltime=log_time.timestamp()) for k, v in losses.items()]

            if epoch_i % 2 == 0:
                for tag, value in model.named_parameters():
                    grad = value.grad
                    if grad is not None:
                        tb_writer.add_histogram(tag + "/grad", grad.cpu(), epoch_i, walltime=log_time.timestamp())

        # logging
        with open(os.path.join(opt.log_path, 'log.txt'), 'a') as f:
            log_dict = {"epoch": epoch_i, "datetime": log_time.isoformat(), "model_saved": model_saved}
            log_dict.update(summary_dict)
            f.write(json.dumps(log_dict) + '\n')

        scheduler.step()

    # if isinstance(tb_writer, tensorboard.SummaryWriter):
    #     hparams = opt.__dict__
    #     metrics = {f"hpar_{k}": v for k, v in summary_dict.items()}
    #     tb_writer.add_hparams(hparams, metrics)
    #     tb_writer.close()

    torch.save(model, os.path.join(opt.log_path, "final_model.pt"))


def main():
    """ Main function. """
    global tb_writer

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-device', type=str, default='cuda')

    parser.add_argument('-ll_loss_factor', type=float, default=1, help="ll loss is multiplied by this")
    parser.add_argument('-time_loss_scaler', type=float, default=100, help="time loss is divided by this")
    parser.add_argument('-time_loss_fn', type=str, default="include_padding", choices=["include_padding",
                                                                                       "exclude_padding"])

    parser.add_argument('-log_path', type=str, default=os.getcwd())

    opt = parser.parse_args()

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    opt.log_path = os.path.join(opt.log_path.rstrip("/"), 'runs', current_time + '_' + socket.gethostname())

    os.makedirs(opt.log_path, exist_ok=True)

    # setup the log file
    with open(os.path.join(opt.log_path, 'log.txt'), 'w') as f:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        config = {"git_hash": git_hash}
        config.update(opt.__dict__)
        f.write(json.dumps(config) + '\n')


    tb_writer = tensorboard.SummaryWriter(log_dir=opt.log_path)

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader_h5(opt)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        device=opt.device
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()
