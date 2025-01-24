from zoo import *
from pprint import pformat


if __name__ == '__main__':
    train_dict = parse_train_args()
    log = f'Experiment settings:\n{pformat(train_dict)}'
    set_seed(train_dict['experiment'].get('seed'), train_dict['experiment'].get('deterministic', False))
    if 'logger' in train_dict:
        strlogger = Logger(**train_dict['logger'])
        strlogger.update_log(log)
    else:
        strlogger = None
        print(log)
    device = train_dict['experiment']['device']
    transform_train = parse_transform(train_dict['transform']['train'])
    transform_val = parse_transform(train_dict['transform']['val']) if 'val' in train_dict['transform'] else None
    train_loader, val_loader, num_classes, dictionary = get_train_val_loaders(
        root=train_dict['data']['root'],
        transform_train=transform_train, transform_val=transform_val,
        split_ratio=train_dict['data']['split_ratio'] if train_dict['experiment']['val'] else [1, 0],
        batch_size=train_dict['data']['batch_size'],
        num_workers=train_dict['data']['num_workers'],
        seed=train_dict['experiment']['seed'] if hasattr(train_dict['experiment'], 'seed') else None
    )
    model = parse_model(train_dict['model'], num_classes=num_classes).to(device)
    optimizer = parse_optimizer(train_dict['optimizer'], params=model.parameters())
    scheduler = parse_scheduler(train_dict['scheduler'], optimizer=optimizer) if 'scheduler' in train_dict else None
    criterion = parse_criterion(train_dict['criterion'])
    evaluator = parse_evaluator(train_dict['evaluator'], criterion=criterion) if 'evaluator' in train_dict else None

    for epoch_idx in range(1, 1 + train_dict['experiment']['max_epochs']):
        train_one_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer,
                        grad_clip=train_dict['optimizer'].get('grad_clip'), lr_scheduler=scheduler,
                        logger=strlogger, device=device)
        if train_dict['experiment']['val'] and epoch_idx % train_dict['experiment']['val_interval'] == 0:
            val(model=model, loader=val_loader, evaluator=evaluator, logger=strlogger, device=device, dictionary=dictionary,
                save_dir=train_dict['experiment']['ckpt_save_dir'] if train_dict['experiment']['save'] else None)
