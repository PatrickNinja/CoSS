from .checkpoint import checkpoint
from .makeModel import make_model
from .run import Fit
from . import constants
from torch.utils.data import DataLoader, distributed
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch
from Transformer.Module import WarmUpOpt, LabelSmoothing
from preprocess import data_loader
import pickle
import importlib


def trainer(gpu, args, seed):
    setattr(args, 'clip_grad', args.gradient_clipper)
    setattr(args, 'PAD_index', constants.PAD_index)
    setattr(args, 'BOS_index', constants.BOS_index)
    setattr(args, 'EOS_index', constants.EOS_index)

    setattr(args, 'rank', gpu)
    criterion = LabelSmoothing(smoothing=args.smoothing,
                               ignore_index=constants.PAD_index).cuda(args.rank)
    check_point = None
    check_point = checkpoint(save_path=args.checkpoint_path,
                             checkpoint_num=args.checkpoint_num,
                             restore_file=args.restore_file,
                             rank=gpu)
    params = {}
    params['checkpoint'] = check_point
    params['criterion'] = criterion

    model_state_dict = None
    optim_state_dict = None
    start_epoch = 0
    model_state_dict, optim_state_dict, start_epoch = params['checkpoint'].restore()
    model = make_model(args)

    optimizer = WarmUpOpt(optimizer=optim.Adam(params=model.parameters(),
                                               lr=args.learning_rate,
                                               betas=(args.beta_1, args.beta_2),
                                               eps=args.eps,
                                               weight_decay=args.weight_decay),
                          d_model=args.embedding_dim,
                          warmup_steps=args.warmup_steps,
                          min_learning_rate=args.min_learning_rate,
                          factor=args.factor,
                          state_dict=optim_state_dict)

    model = model

    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    params['model'] = model
    params['optimizer'] = optimizer
    del model, optimizer

    with open(args.train_file, 'rb') as f:
        train_data = pickle.load(f)

    train_dataset, batch_size = train_data.set_param(shuffle=True,
                                                     args=args,
                                                     train_flag=True,
                                                     seed=seed)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    del train_dataset
    params['train_data'] = train_loader
    del train_loader
    fit = Fit(args, params)

    fit(start_epoch=start_epoch)
