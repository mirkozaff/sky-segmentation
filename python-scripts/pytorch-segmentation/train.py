import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
import tensorboard
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, pretrained):
    train_logger = Logger()

    # DATA LOADERS
    val_split = config['train_loader']['args']['val_split']
    train_loader = get_instance(dataloaders, 'train_loader', config)
    if val_split > 0:
        val_loader = train_loader.get_val_loader()
    else:
        val_loader = get_instance(dataloaders, 'val_loader', config)
    
    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # Load pretrained weights
    if pretrained:
        print(f'Loading pretrinaed weights: {pretrained}')
        model.load_state_dict(torch.load(os.path.join(pretrained))['state_dict'])

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()

    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    model.load_state_dict(torch.load(os.path.join(trainer.checkpoint_dir, f'best_model.pth'))['state_dict'])
    model_scripted = torch.jit.script(model) 
    model_scripted.save('best_model_scripted.pth')

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                           help='Path to the .pth pretrained model weights')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume, args.pretrained)