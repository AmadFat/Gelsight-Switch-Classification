from pathlib import Path
import argparse
import datetime
import yaml

__all__ = ['parse_train_args', 'parse_eval_args']


def filter_none(d: dict):
    for k in list(d.keys()):
        if d[k] is None:
            d.pop(k)
        elif isinstance(d[k], dict):
            filter_none(d[k])
    return d

def parse_train_args():
    parser = argparse.ArgumentParser(description="Gelsight-Switch-Classification training script")
    
    # Experiment settings
    parser.add_argument('-e', '--exp', '--exp-name', type=str, default=datetime.datetime.now().strftime("%m.%d-%H:%M:%S"), help='Experiment name')
    parser.add_argument('-d', '--device', type=str, required=True, choices=['cpu', 'cuda'], help='cpu / cuda')
    parser.add_argument('--max-epochs', type=int, default=10, help='Max training epochs')
    parser.add_argument('--seed', type=int, required=False, help='Global random seed')
    parser.add_argument('--deterministic', action='store_true', help='If use deterministic training (REALLY SLOW)')

    # Logging settings
    parser.add_argument('--log', '--use-log', action='store_true', help='If use the log')
    parser.add_argument('--log-interval', '--log-step-interval', type=int, default=20, help='Log message interval')
    parser.add_argument('--log-dir', '--log-save-dir', type=str, default='logs', help='Log save directory')
    parser.add_argument('--tb', '--use-tb', '--use-tensorboard', action='store_true', help='If use tensorboard')
    parser.add_argument('--tb-dir', '--tb-save-dir', type=str, default='tbevents', help='Tensorboard save directory')
    
    # Save settings
    parser.add_argument('--save', '--save-ckpt', action='store_true', help='If save the checkpoint')
    parser.add_argument('--ckpt-dir', '--ckpt-save-dir', type=str, default='ckpts', help='Checkpoint save directory')

    # Data settings
    parser.add_argument('--root', '-r', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--val', '--use-val', action='store_true', help='If seperate the validation set')
    parser.add_argument('--split-ratio', type=float, nargs=2, required=False, help='Train/val split ratio')
    parser.add_argument('--val-interval', '--val-epoch-interval', type=int, default=1, help='Validation interval')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data loader')
    parser.add_argument('-b', '--batch-size', type=int, required=True, help='Batch size')
    
    # Transform settings
    parser.add_argument('--transform-train', '--transform-train-file', type=str, required=False, help='Transform train yaml file')
    parser.add_argument('--transform-val', '--transform-val-file', type=str, required=False, help='Transform val yaml file')
    
    # Model settings
    parser.add_argument('-m', '--model', '--model-name', type=str, required=True, choices=['resnet18', 'mobilenet_v3_s'], help='Model name')
    parser.add_argument('--pretrained', action='store_true', help='If use pretrained weights')
    parser.add_argument('--norm-layer', type=str, default='batchnorm', choices=['batchnorm', 'frozenbatchnorm'], help='Norm layer type')
    parser.add_argument('--dropout', type=float, required=False, help='Dropout rate for mobilenet_v3_s')
    parser.add_argument('--weight-init', '--weight-init-last-fc', type=str, default='const', help='Weight init for last fc layer',
                        choices=['const', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'], )
    parser.add_argument('--bias-init', '--bias-init-last-fc', type=str, default='const', help='Bias init for last fc layer',
                        choices=['const', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'], )

    # Optimizer settings
    parser.add_argument('-o', '--optimizer', '--optimizer-name', type=str, required=True, choices=['sgd', 'adamw'], help='Optimizer name')
    parser.add_argument('--lr', '--learning-rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--momentum', type=float, required=False, help='Momentum for SGD')
    parser.add_argument('--betas', type=float, nargs=2, required=False, help='2 betas for AdamW')
    parser.add_argument('--weight-decay', type=float, required=False, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, required=False, help='Gradient clip')

    # Learning rate scheduler settings
    parser.add_argument('-s', '--scheduler', '--scheduler-name', type=str, required=False, \
                        choices=['constlr', 'steplr', 'cosinelr'], help='Learning rate scheduler name')
    parser.add_argument('--factor', type=float, required=False, help='Initial factor for constlr')
    parser.add_argument('--milestones', type=int, nargs='+', required=False, help='Milestones for steplr')
    parser.add_argument('--gamma', type=float, required=False, help='Decay factor for steplr')
    parser.add_argument('--period', type=int, required=False, help='Period for cosinelr')
    parser.add_argument('--period-mult', type=int, required=False, help='Period multiplier for cosinelr')
    parser.add_argument('--min-lr', type=float, required=False, help='Minimum learning rate for cosinelr')

    # Criterion settings
    parser.add_argument('-c', '--criterion', '--criterion-name', type=str, required=True, choices=['celoss', 'focalloss'], help='Criterion name')
    parser.add_argument('--label-smoothing', type=float, required=False, help='Label smoothing for CELoss')
    parser.add_argument('--focal-alpha', type=float, required=False, help='Alpha for FocalLoss')
    parser.add_argument('--focal-gamma', type=float, required=False, help='Gamma for FocalLoss')

    # Evaluation settings
    parser.add_argument('--eval-loss', '--use-loss-evaluator', action='store_true', help='If evaluate loss')
    parser.add_argument('--eval-acc', '--use-acc-evaluator', action='store_true', help='If evaluate accuracy')

    args = parser.parse_args()

    # if not required, do not update to dict

    exp_dict = {
        "experiment_name": args.exp,
        "device": args.device,
        "max_epochs": args.max_epochs,
        "val": args.val,
        "save": args.save,
    }
    if args.seed is not None:
        exp_dict["seed"] = args.seed
        exp_dict["deterministic"] = args.deterministic
    if args.save is True:
        exp_dict["ckpt_save_dir"] = Path(args.ckpt_dir) / args.exp
    if args.val is True:
        exp_dict["val_interval"] = args.val_interval

    logger_dict = {
        "train_print_interval": args.log_interval,
        "window_metric": args.log_interval,
        "window_time_stamp": args.log_interval,
        "log_save_path": Path(args.log_dir) / f'{args.exp}.log',
        "use_tensorboard": args.tb,
        "tb_save_path": Path(args.tb_dir) / args.exp if args.tb else None,
    } if args.log else None

    if args.transform_train is not None:
        with open(args.transform_train, 'r') as f:
            transform_train = yaml.safe_load(f)
    if args.transform_val is not None:
        with open(args.transform_val, 'r') as f:
            transform_val = yaml.safe_load(f)
    transform_dict = {
        'train': transform_train if args.transform_train else {'totensor': {}},
        'val': transform_val if args.transform_val else {'totensor': {}} if args.val else None,
    }

    data_dict = {
        "root": args.root,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
    }
    if args.val is True:
        data_dict['split_ratio'] = args.split_ratio

    model_dict = {
        "model_name": args.model,
        'weights': 'default' if args.pretrained else 'none',
        "norm_layer": args.norm_layer,
        "last_fc_weight_init": args.weight_init,
        "last_fc_bias_init": args.bias_init,
    }
    if args.model == 'mobilenet_v3_s':
        model_dict['dropout'] = args.dropout

    optim_dict = {
        "optimizer_name": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
    }
    if args.optimizer == 'sgd':
        optim_dict['momentum'] = args.momentum
    if args.optimizer == 'adamw':
        optim_dict['betas'] = args.betas

    scheduler_dict = None if args.scheduler is None else {"scheduler_name": args.scheduler}
    if args.scheduler == 'constlr':
        scheduler_dict['factor'] = args.factor
    if args.scheduler == 'steplr':
        scheduler_dict['milestones'] = args.milestones
        scheduler_dict['gamma'] = args.gamma
    if args.scheduler == 'cosinelr':
        scheduler_dict['T_0'] = args.period
        scheduler_dict['T_mult'] = args.period_mult
        scheduler_dict['eta_min'] = args.min_lr

    criterion_dict = {"criterion_name": args.criterion}
    if args.criterion == 'celoss':
        criterion_dict['label_smoothing'] = args.label_smoothing
    if args.criterion == 'focalloss':
        criterion_dict['alpha'] = args.focal_alpha
        criterion_dict['gamma'] = args.focal_gamma

    evaluator_dict = {
        "loss": args.eval_loss,
        "acc": args.eval_acc,
    } if args.val else None

    return filter_none({
        "experiment": exp_dict,
        "logger": logger_dict,
        "transform": transform_dict,
        "data": data_dict,
        "model": model_dict,
        "optimizer": optim_dict,
        "scheduler": scheduler_dict,
        "criterion": criterion_dict,
        "evaluator": evaluator_dict
    })


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Gelsight-Switch-Classification evaluation script")

    # Experiment settings
    parser.add_argument('-e', '--eval', '--eval-name', type=str, default=datetime.datetime.now().strftime("%m.%d-%H:%M:%S")+"-eval", help='Evaluation name')
    parser.add_argument('-d', '--device', type=str, required=True, choices=['cpu', 'cuda'], help='cpu / cuda')
    parser.add_argument('--log-dir', '--log-save-dir', type=str, default='logs', help='Log save directory')

    # Data settings
    parser.add_argument('-r', '--root', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--transform', '--transform-file', type=str, required=False, help='Transform yaml file')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data loader')
    
    # Model settings
    parser.add_argument('-m', '--model', '--model-name', type=str, required=True, choices=['resnet18', 'mobilenet_v3_s'], help='Model name')
    parser.add_argument('--ckpt', '--checkpoint', '--checkpoint-path', type=str, required=True, help='Checkpoint file path')
    
    # Evaluation settings
    parser.add_argument('--eval-acc', '--use-acc-evaluator', action='store_true', help='If evaluate accuracy')

    args = parser.parse_args()
    
    with open(args.transform, 'r') as f:
        transform = yaml.safe_load(f)

    return filter_none({
        "experiment": {
            "experiment_name": args.eval,
            "device": args.device,
        },
        "logger": {
            "log_save_path": Path(args.log_dir) / f'{args.eval}.log',
        },
        "data": {
            "root": args.root,
            "transform": transform,
            "num_workers": args.num_workers,
        },
        "model": {
            "model_name": args.model,
            "checkpoint_path": args.ckpt,
        },
        "evaluator": {
            "acc": args.eval_acc,
        },
    })