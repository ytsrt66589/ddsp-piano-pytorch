import os
import time
import argparse
import numpy as np 

import torch
import torch.nn as nn 

import soundfile as sf

from logger.saver import Saver 

from ddsp_piano.data_pipeline import get_training_dataset, get_validation_dataset
from ddsp_piano.default_model import get_model
from ddsp_piano.modules.loss import HybridLoss

def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

def process_args():
    # Get arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help="Number of elements per batch.\
                        (default: %(default)s)")

    parser.add_argument('--steps_per_epoch', type=int, default=16,
                        help="Number of steps of gradient descent per epoch.\
                        (default: %(default)s)")

    parser.add_argument('--epochs', '-e', type=int, default=128,
                        help="Number of epochs. (default: %(default)s)")

    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate. (default: %(default)s)")

    parser.add_argument('--phase', '-p', type=int, default=1,
                        help="Training phase strategy to apply. \
                        Set to even for fine-tuning only the detuner and \
                        inharmonicity sub-modules.\
                        (default: %(default)s)")

    parser.add_argument('--cuda', type=int, default=1,
                        help="Using Cuda or not")

    parser.add_argument('--logs_interval', type=int, default=20)

    parser.add_argument('--sampling_rate', type=int, default=16000)
    
    parser.add_argument('--restore', type=str, default=None,
                        help="Restore training step from a saved folder.\
                        (default: %(default)s)")

    parser.add_argument('maestro_path', type=str,
                        help="Path to the MAESTRO dataset folder.")
    parser.add_argument('maestro_cache_path', type=str,
                        help="Path to the MAESTRO cache dataset folder.")
    parser.add_argument('exp_dir', type=str,
                        help="Folder to store experiment results and logs.")

    return parser.parse_args()

def test(args, model, loss_func, loader_test, path_gendir='gen'):
    print(' [*] testing...')
    print(' [*] output folder:', path_gendir)
    model.eval()

    # losses
    test_loss = 0.
    test_mss_loss = 0.
    test_reverb_l1_loss = 0.
    # intialization
    num_batches = len(loader_test)
    os.makedirs(path_gendir, exist_ok=True)
    rtf_all = []

    # run 
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            audio, conditioning, pedal, piano_model = data
            if args.cuda:
                audio = audio.cuda()
                conditioning = conditioning.cuda()
                pedal = pedal.cuda()
                piano_model = piano_model.cuda()
            
            # forward
            st_time = time.time()
            signal, reverb_ir, _ = model(conditioning, pedal, piano_model)
            ed_time = time.time()

            # crop
            min_len = np.min([signal.shape[1], audio.shape[1]])
            signal        = signal[:,:min_len]
            audio = audio[:,:min_len]

            # RTF
            run_time = ed_time - st_time
            song_time = audio.shape[-1] / args.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)

            # loss
            loss, mss_loss, reverb_l1_loss = loss_func(signal, audio, reverb_ir)
            test_loss += loss.item()
            test_mss_loss += mss_loss.item()
            test_reverb_l1_loss += reverb_l1_loss.item()

            # path
            path_pred = os.path.join(path_gendir, f'pred_{bidx}.wav')
            path_anno = os.path.join(path_gendir, f'anno_{bidx}.wav')
            print(' > path_pred:', path_pred)
            print(' > path_anno:', path_anno)
            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(path_anno), exist_ok=True)

            # to numpy
            pred  = convert_tensor_to_numpy(signal)
            anno  = convert_tensor_to_numpy(audio)
            # save
            sf.write(path_pred, pred, args.sampling_rate)
            sf.write(path_anno, anno, args.sampling_rate)
    # report
    test_loss /= num_batches
    test_mss_loss /= num_batches
    test_reverb_l1_loss /= num_batches
    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' [test_mss_loss] test_mss_loss:', test_mss_loss)
    print(' [test_reverb_l1_loss] test_reverb_l1_loss:', test_reverb_l1_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss, test_mss_loss, test_reverb_l1_loss

def main(args):
    """Training loop script.
    Args:
        - batch_size (int): nb of elements per batch.
        - steps_per_epoch (int): nb of steps of gradient descent per epoch.
        - epochs (int): nb of epochs.
        - restore (path): load model and optimizer states from this folder.
        - phase (int): current training phase.
        - maestro_path (path): maestro dataset location.
        - exp_dir (path): folder to store experiment results and logs.
    """
    #print('args : ', args)
    # Format training phase strategy
    first_phase_strat = ((args.phase % 2) == 1)
    
    # Prepare model 
    model = get_model()
    model.alternate_training(first_phase=first_phase_strat)
    if args.restore:
        # load checkpoint
        print('###### LOAD MODEL #########')
        ckpt = torch.load(os.path.join(args.restore, 'ddsp-piano_11200_params.pt'))
        model.load_state_dict(ckpt)
        print('###### SUCESS     ##########')
    # Prepare dataset 
    training_dataset = get_training_dataset(args.maestro_path, args.maestro_cache_path, max_polyphony=model.n_synths)
    val_dataset = get_validation_dataset(args.maestro_path, args.maestro_cache_path, max_polyphony=model.n_synths)
    
    #exit(1)
    training_dataset_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, pin_memory=True)

    # Prepare Loss function 
    n_ffts = [2048, 1024, 512, 256, 128, 64]
    loss_func = HybridLoss(n_ffts, model.inharm_model, first_phase_strat)

    # Optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # GPU or CPU (default: GPU)
    if args.cuda:
        model = model.cuda()
        loss_func = loss_func.cuda()
    
    # Inits before the training loop
    args.exp_dir = os.path.join(args.exp_dir, f'phase_{args.phase}')
    saver = Saver(args)

    # Training loop
    lowest_val_loss = np.inf
    prev_save_time = -1
    saver.log_info('======= start training =======')
    for epoch in range(args.epochs):
        for idx, batch in enumerate(training_dataset_loader):
            saver.global_step_increment()
            audio, conditioning, pedal, piano_model = batch
            '''
                - audio: (b, sample_rate*duration)
                - conditioning: (b, n_frames, max_polyphony, 2)
                - pedal: (b, n_frames, 4)
                - piano_model: (b)
            '''
            if args.cuda:
                audio = audio.cuda()
                conditioning = conditioning.cuda()
                pedal = pedal.cuda()
                piano_model = piano_model.cuda()

            # forward 
            signal, reverb_ir, non_ir_signal = model(conditioning, pedal, piano_model)

            # loss 
            loss, loss_mss, loss_reverb_l1 = loss_func(signal, audio, reverb_ir)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if saver.global_step % args.logs_interval == 0:
                saver.log_info(
                    'epoch: {}/{} {:3d}/{:3d} | {} | t: {:.2f} | loss: {:.6f} | time: {} | counter: {}'.format(
                        epoch,
                        args.epochs,
                        idx,
                        len(training_dataset_loader),
                        saver.expdir,
                        saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_info(
                    ' > loss: {:.6f}, mss loss: {:.6f}, reverb_loss: {:.6f}'.format(
                       loss.item(),
                       loss_mss.item(),
                       loss_reverb_l1.item()
                    )
                )
                saver.log_value({
                    'train loss': loss.item(),
                    'train loss mss': loss_mss.item(),
                    'train loss reverb_l1': loss_reverb_l1.item()
                })
            if saver.global_step % 800 == 0:
                saver.save_models({'ddsp-piano': model}, postfix=f'{saver.global_step}')
                saver.make_report()
        
        # Skip validation during early training
        #saver.save_models({'ddsp-piano': model}, postfix='last_iter')
        # validation 
        cur_hour = saver.get_total_time(to_str=False) // 3600
        if cur_hour != prev_save_time:
            saver.save_models({'ddsp-piano': model}, postfix=f'{saver.global_step}_{cur_hour}')
            prev_save_time = cur_hour
            
            # run testing set
            path_testdir_runtime = os.path.join(
                        args.exp_dir,
                        'runtime_gen', 
                        f'gen_{saver.global_step}_{cur_hour}')
            test_loss, test_mss_loss, test_reverb_l1_loss = test(args, model, loss_func, val_dataset_loader, path_gendir=path_testdir_runtime)
            saver.log_info(
                ' --- <validation> --- \nloss: {:.6f}. mss_loss: {:.6f}. reverb_l1_loss {:.6f}'.format(
                    test_loss,
                    test_mss_loss,
                    test_reverb_l1_loss
                )
            )

            saver.log_value({
                'valid loss': test_loss,
                'valid loss mss': test_mss_loss,
                'valid loss reverb_l1': test_reverb_l1_loss
            })
            model.train()

            # save best model
            if test_loss < lowest_val_loss:
                saver.log_info(' [V] best model updated.')
                saver.save_models(
                    {'ddsp-piano': model}, postfix='best')
                test_loss = lowest_val_loss

            saver.make_report()

if __name__ == '__main__':
    main(process_args())