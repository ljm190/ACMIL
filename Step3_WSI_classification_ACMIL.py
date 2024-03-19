
# !/usr/bin/env python
import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed, Wandb_Writer
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import AttnMIL6 as AttnMIL
from architecture.transformer import TransformWrapper

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
import torchmetrics
from sklearn.metrics import roc_auc_score, auc, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/huaxi_medical_ssl_config.yml',
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help=""
    )
    parser.add_argument(
        "--wd", type=float, default=0.00001, help=""
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument(
        "--n_token", type=int, default=5, help="number of query token"
    )
    parser.add_argument(
        "--n_masked_patch", type=int, default=10, help="whether use adversarial mask"
    )
    parser.add_argument(
        "--mask_drop", type=float, default=0.5, help="number of query token"
    )
    parser.add_argument(
        "--level", type=str, default=1, help="just used for wandb"
    )
    parser.add_argument(
        "--model_type", type=str, default=1, help="just used for wandb"
    )
    parser.add_argument(
        "--split", type=int, default=0, help="split number"
    )
    parser.add_argument("--data_dir", type=str, default='', help="path to features")
    parser.add_argument("--arch", type=str, default='ga', help="choice of architecture type")
    args = parser.parse_args()
    return args

def main():
    # Load config file
    args = get_arguments()

    # get config
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    #conf.data_dir = args.data_dir
    #conf.name = args.data_dir.split("/")[-3]    

    conf.lr = args.lr
    conf.wd = args.wd

    group_name = 'ds_%s_%s_arch_%s_ntoken_%s_nmp_%s_mask_drop_%s_%sepochs' % (conf.dataset, conf.pretrain, conf.arch, conf.n_token, conf.n_masked_patch, conf.mask_drop, conf.train_epoch)
    name = f"Split: {args.split} - {args.model_type} - lr  {args.lr} - wd {args.wd}"
    log_writer = Wandb_Writer(project_name= "ACMIL-updated_metrics",entity='upc_gpi',name=name,args= args)
    conf.ckpt_dir = log_writer.wandb.dir[:-5] + 'saved_models'
    if conf.wandb_mode == 'disabled':
        conf.ckpt_dir = os.path.join(conf.ckpt_dir, group_name, str(args.seed))
    os.makedirs(conf.ckpt_dir, exist_ok=True)
    print("Used config:");
    pprint(vars(conf));

    # Prepare dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf,args.split)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # define network
    if conf.arch == 'ga':
        milnet = AttnMIL(conf)
    else:
        milnet = TransformWrapper(conf)
    milnet.to(device)

    criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, milnet.parameters()), lr=args.lr, weight_decay=conf.wd)

    #best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    best_state = {'best_epoch':-1, 'best_test_acc':0, 'best_test_auc':0, 'best_test_f1':0, 'best_test_precision':0,'best_test_recall':0,'best_val_acc':0, 'best_val_auc':0, 'best_val_f1':0, 'best_val_precision':0,'best_val_recall':0}
    train_epoch = conf.train_epoch
    if conf.arch == 'mha':
        train_epoch = 50
    for epoch in range(train_epoch):
        metrics = train_one_epoch(milnet, criterion, train_loader, optimizer0, device, epoch, conf, log_writer)


        val_auc, val_acc, val_f1, val_loss,val_precision,val_recall = evaluate(milnet, criterion, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss,test_precision,test_recall = evaluate(milnet, criterion, test_loader, device, conf, 'Test')

        if log_writer is not None:
            metrics.update({'perf/val_acc1': val_acc,'perf/val_auc': val_auc,
                            'perf/val_f1': val_f1,'perf/val_loss': val_loss,'perf/test_acc1': test_acc,
                            'perf/test_auc': test_auc,'perf/test_f1': test_f1, 'perf/test_loss': test_loss,
                            'perf/val_precision': val_precision,
                            'perf/val_recall': val_recall,'perf/test_precision': test_precision,
                            'perf/test_recall': test_recall })
            log_writer.wandb.log({**metrics})

            #log_writer.log('perf/val_acc1', val_acc, commit=False)
            #log_writer.log('perf/val_auc', val_auc, commit=False)
            #log_writer.log('perf/val_f1', val_f1, commit=False)
            #log_writer.log('perf/val_loss', val_loss, commit=False)
            #log_writer.log('perf/test_acc1', test_acc, commit=False)
            #log_writer.log('perf/test_auc', test_auc, commit=False)
            #log_writer.log('perf/test_f1', test_f1, commit=False)
            #log_writer.log('perf/test_loss', test_loss, commit=False)

        #if val_f1 + val_auc > best_state['best_val_f1'] + best_state['best_val_auc']:
        if val_f1 > best_state['best_val_f1'] :
            best_state['best_epoch'] = epoch
            best_state['best_val_auc'] = val_auc
            best_state['best_val_acc'] = val_acc
            best_state['best_val_f1'] = val_f1
            best_state['best_test_auc'] = test_auc
            best_state['best_test_acc'] = test_acc
            best_state['best_test_f1'] = test_f1
            best_state['best_val_precision'] = val_precision
            best_state['best_val_recall'] = val_recall
            best_state['best_test_precision'] = test_precision
            best_state['best_test_recall'] = test_recall
            save_model(
                conf=conf, model=milnet, optimizer=optimizer0, epoch=epoch, is_best=True)
            print("best_state",best_state)
        print('\n')

    log_writer.wandb.log(best_state)

    save_model(
        conf=conf, model=milnet, optimizer=optimizer0, epoch=epoch, is_last=True)
    print("Results on best epoch:")
    print(best_state)

def train_one_epoch(milnet, criterion, data_loader, optimizer0, device, epoch, conf, log_writer=None):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    milnet.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    metrics = {}

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), conf)
        # adjust_learning_rate(optimizer1, epoch + data_it/len(data_loader), conf)

        # Compute loss
        sub_preds, slide_preds, attn = milnet(image_patches, use_attention_mask=True)
        if conf.n_token > 1:
            loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token))
        else:
            loss0 = torch.tensor(0.)
        loss1 = criterion(slide_preds, labels)


        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attn = torch.softmax(attn, dim=-1)
        for i in range(conf.n_token):
            for j in range(i + 1, conf.n_token):
                diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                            conf.n_token * (conf.n_token - 1) / 2)

        loss = diff_loss + loss0 + loss1

        optimizer0.zero_grad()
        # Backpropagate error and update parameters
        loss.backward()
        optimizer0.step()


        metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
        metric_logger.update(sub_loss=loss0.item())
        metric_logger.update(diff_loss=diff_loss.item())
        metric_logger.update(slide_loss=loss1.item())
        # metric_logger.update(mask_drop=mask_drop.item())

        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            #log_writer.log('sub_loss', loss0, commit=False)
            #log_writer.log('diff_loss', diff_loss, commit=False)
            #log_writer.log('slide_loss', loss1)
            ## log_writer.log('mask_drop', mask_drop)
            metrics = {"loss":loss,"diff_loss":diff_loss,"slide_loss":loss1,"sub_loss":loss0}
    return metrics




# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header):

    # Set the network to evaluation mode
    net.eval()

    y_pred = []
    y_true = []

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)


        sub_preds, slide_preds, attn = net(image_patches, use_attention_mask=False)
        loss = criterion(slide_preds, labels)
        pred = torch.softmax(slide_preds, dim=-1)


        acc1 = accuracy(pred, labels, topk=(1,))[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])

        y_pred.append(pred)
        y_true.append(labels)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0).cpu().numpy()

    y_pred = y_pred.argmax(dim=1).cpu().numpy()

    auroc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    #AUROC_metric = torchmetrics.AUROC(task="multiclass",num_classes = conf.n_class, average = 'macro').to(device)
    #AUROC_metric(y_pred, y_true)
    #auroc = AUROC_metric.compute().item()
    #F1_metric = torchmetrics.F1Score(task="multiclass",num_classes = conf.n_class, average = 'macro').to(device)
    #F1_metric(y_pred, y_true)
    #f1_score = F1_metric.compute().item()
    #recall_macro = torchmetrics.classification.MulticlassRecall(num_classes = conf.n_class, average = 'macro').to(device)
    #recall_macro = recall_macro(y_pred, y_true).item()
    #precision_macro = torchmetrics.classification.MulticlassPrecision(num_classes = conf.n_class, average = 'macro').to(device)
    #precision_macro = precision_macro(y_pred, y_true).item()
    
 

    
    print(header,"acc",metric_logger.acc1.global_avg, "auroc",auroc, "f1_score", f1, "precision", precision, "recall",recall)


    return auroc, metric_logger.acc1.global_avg, f1, metric_logger.loss.global_avg, precision,recall



if __name__ == '__main__':
    main()

