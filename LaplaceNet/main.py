import os
import time
import numpy as np
import torch
from config import datasets, cli
import helpers
import math
import sys
import logging


def laplacenet():
    #### Get the command line arguments
    args = cli.parse_commandline_args()
    logging.info(args)
    args = helpers.load_args(args)
    args.file = args.dataset + "_" + args.model + "_" + str('-'.join(args.ckpt.split('/')[-2:])) + "_mix" + str(
        args.mixup) + "_nl" + str(
        args.num_labeled) + "_checkt" + str(args.check_t)
    args.file = args.file.replace('/', '-').replace('.', '') + ".txt"
    checkpoint_file = 'checkpoints/' + args.file[:-4] + '.pth'

    args.mixup = args.mixup > 0
    args.check_t = args.check_t > 0

    # handler1 = logging.StreamHandler()
    # formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    # handler1.setFormatter(formatter)
    logger = logging.getLogger()
    # logger.addHandler(handler1)
    logger.setLevel('INFO')

    logging.info(f'nl={args.num_labeled}')
    logging.info(f'mixup={args.mixup}')
    logging.info(f'check_t={args.check_t}')
    logging.info(f'ckpt={args.ckpt}')

    #### Load the dataset
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    args.num_classes = num_classes

    #### Create loaders
    #### train_loader loads the labeled data , eval loader is for evaluation
    #### train_loader_noshuff extracts features
    #### train_loader_l, train_loader_u together create composite batches
    #### dataset is the custom dataset class
    train_loader, eval_loader, train_loader_noshuff, train_loader_l, train_loader_u, dataset = helpers.create_data_loaders_simple(
        **dataset_config, args=args)

    #### Transform steps into epochs
    num_steps = args.num_steps
    ini_steps = math.floor(args.num_labeled / args.batch_size) * 100
    ssl_steps = math.floor(len(dataset.unlabeled_idx) / (args.batch_size - args.labeled_batch_size))
    args.epochs = 10 + math.floor((num_steps - ini_steps) / ssl_steps)
    args.lr_rampdown_epochs = args.epochs + 10

    check_t = args.check_t

    n_steps_supervised_epoch = 10
    ts = [0.5]
    orig_epochs = args.epochs
    if check_t:
        args.epochs = 10
        ts = np.linspace(0, 1, 11).clip(1e-2, 1. - 1e-2)
    ts = np.array(ts)

    min_info = {'train_loss': float('inf')}

    for is_check_t_run in [True, False] if args.check_t else [False]:
        if not is_check_t_run and args.check_t:
            logging.info('Chosen min info:')
            logging.info(('t and epoch', min_info['t'], min_info['epoch']))
            logging.info(('train_loss', min_info['train_loss']))
        for t in (ts if is_check_t_run else np.array([0.5])):
            t = t.item()
            # Create Model and Optimiser
            args.device = torch.device('cuda')
            model = helpers.create_model(num_classes, args)
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov)
            epoch_results = np.zeros((args.epochs, 6))

            start_epoch = 0
            state_dict = None
            if args.ckpt != '' and args.ckpt != 'none':
                state_dict = torch.load(args.ckpt, map_location=args.device)
                logging.info(state_dict['time_embedding.param_t'])
                logging.info(state_dict['time_embedding.param_t'].shape)
                state_dict['time_embedding.param_t'] = state_dict['time_embedding.param_t'] * 0. + torch.log(
                    torch.Tensor((t / (1 - t),))[0]).to(args.device)
            if os.path.exists(checkpoint_file) and args.load_ckpt > 0:
                epoch_results = np.loadtxt(args.file)
                start_epoch = epoch_results[:, 0].max().astype(int).item() + 1
                state_dict = torch.load(checkpoint_file, map_location=args.device)
                logging.info('load ckpt')
                logging.info(('start epoch', start_epoch))
            if args.check_t and not is_check_t_run:
                state_dict = min_info['model_state']
                start_epoch = 10
                args.epochs = orig_epochs
            if state_dict is not None:
                current_model = model.module.state_dict()
                keys_vin = state_dict
                new_state_dict = {k: v if v.size() == current_model[k].size() else current_model[k] for k, v in
                                  zip(current_model.keys(), keys_vin.values())
                                  }
                dropped_keys = [k for k, v in zip(current_model.keys(), keys_vin.values()) if
                                v.size() != current_model[k].size()]
                logging.info((current_model['time_embedding.param_t'], current_model['time_embedding.param_t'].size()))
                logging.info((keys_vin['time_embedding.param_t'], keys_vin['time_embedding.param_t'].size()))
                logging.info(('dropped keys:', dropped_keys))
                model.module.load_state_dict(new_state_dict, strict=False)

            # Information store in epoch results and then saved to file
            global_step = 0

            for epoch in range(start_epoch, args.epochs):
                start_epoch_time = time.time()
                # Extract features and run label prop on graph laplacian
                if epoch >= 10:
                    dataset.feat_mode = True
                    feats = helpers.extract_features_simp(train_loader_noshuff, model, args)
                    dataset.feat_mode = False
                    dataset.one_iter_true(feats, k=args.knn, max_iter=30, l2=True, index="ip")

                # Supervised Initilisation vs Semi-supervised main loop
                start_train_time = time.time()
                if epoch < 10:
                    logging.info(("Supervised Initilisation:", (epoch + 1), "/", 10))
                    for i in range(n_steps_supervised_epoch):
                        global_step, train_loss = helpers.train_sup(train_loader, model, optimizer, epoch, global_step,
                                                                    args)
                    print((t, epoch, train_loss), file=sys.stderr)
                    if train_loss < min_info['train_loss']:
                        min_info['train_loss'] = train_loss
                        min_info['model_state'] = model.module.state_dict()
                        min_info['t'] = t
                        min_info['epoch'] = epoch
                if epoch >= 10:
                    global_step = helpers.train_semi(train_loader_l, train_loader_u, model, optimizer, epoch,
                                                     global_step,
                                                     args)

                end_train_time = time.time()
                logging.info(f'Evaluating the primary model at epoch {epoch} (global step {global_step}):')
                prec1, prec5 = helpers.validate(eval_loader, model, args, global_step, epoch + 1,
                                                num_classes=args.num_classes)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logging.info(('t=', torch.sigmoid(model.module.time_embedding.param_t).detach().cpu().numpy()))

                epoch_results[epoch, 0] = epoch
                epoch_results[epoch, 1] = prec1
                epoch_results[epoch, 2] = prec5
                epoch_results[epoch, 3] = dataset.acc
                epoch_results[epoch, 4] = time.time() - start_epoch_time
                epoch_results[epoch, 5] = end_train_time - start_train_time

                np.savetxt(args.file, epoch_results)
                torch.save(model.module.state_dict(), checkpoint_file)

                if epoch >= args.max_epochs - 1:
                    break

    logging.info('Finished')


if __name__ == '__main__':
    laplacenet()
