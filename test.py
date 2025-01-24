from zoo import *
from pprint import pformat
import torch


if __name__ == '__main__':
    eval_dict = parse_eval_args()
    log = f'Evaluation settings:\n{pformat(eval_dict)}'
    strlogger = Logger(**eval_dict['logger']) if 'logger' in eval_dict else None
    strlogger.update_log(log)
    device = eval_dict['experiment']['device']
    transform = parse_transform(eval_dict['data']['transform'])
    test_loader, num_classes, dictionary = get_test_loader(
        root=eval_dict['data']['root'],
        transform=transform,
        num_workers=eval_dict['data']['num_workers']
    )
    model: torch.nn.Module = parse_model(eval_dict['model'], num_classes=num_classes, weights='none')
    model.load_state_dict(torch.load(eval_dict['model']['checkpoint_path'], map_location='cpu', weights_only=True), strict=True)
    model = model.to(device)
    evaluator = parse_evaluator(eval_dict['evaluator'], criterion=None)
    test(model=model, loader=test_loader, evaluator=evaluator, logger=strlogger, device=device, dictionary=dictionary)