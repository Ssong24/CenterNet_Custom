from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

import os

import torch
import torch.utils.data

from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.ctdet import CtdetTrainer

# channel pruning
# from trains.train_factory import train_factory
import time


"""
python main.py ctdet --exp_id test --dataset etri_distort --data_dir etri-safety_system --arch resdcn_18 --batch_size 4 --lr 1.25e-4 --num_workers 0
"""

def main(opt):
  torch.manual_seed(opt.seed)  # Use same random number as before, set manual seed number
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  # torch.backends.cudnn.benchmark = True --> Optimize my model based on the inputs for maximum performance on the gpu.
  Dataset = get_dataset(opt.dataset, opt.task)  # ctdet
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset) # After setting task, change some opt setting.
  # print(opt)

  logger = Logger(opt)  # save the log file.

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # set cpu or (multiple) gpu(s) (default: one gpu)
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)

  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0

  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = CtdetTrainer
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=0,  #1
      pin_memory=True
  )

  print('Print all modules in the model')
  for i, m in enumerate(model.modules()):
      print(m)
      if isinstance(m, torch.nn.BatchNorm2d):
        hey = m.weight.grad.data
        hey = torch.sign(m.weight.data)
  input()


  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10

  # Save loss value per epoch
  loss_folder = opt.save_dir + "/loss/"
  if not os.path.exists(loss_folder):
    os.makedirs(loss_folder)

  log_train_path = loss_folder + "loss_train.txt"
  log_val_path = loss_folder + "loss_val.txt"

  file_train_loss = open(log_train_path, "w")
  file_val_loss = open(log_val_path, "w")

  # Start training
  t0 = time.time()
  print('Starting training for %g epochs...' % start_epoch + 1)
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    # Save training result in logger and loss.txt
    logger.write('epoch: {} |'.format(epoch))
    file_train_loss.write('epoch: {} |'.format(epoch))

    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
      file_train_loss.write('{} {:8f} | '.format(k, v))
    file_train_loss.write('\n')

    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)

      file_val_loss.write('epoch: {} |'.format(epoch))
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
        file_val_loss.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')

    if epoch % 5 == 0:
      file_val_loss.write('\n')
    if epoch in opt.lr_step :
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  print('start_epoch: %d, total num_epochs: %d', start_epoch, opt.num_epochs)
  print('%g epochs completed in %.3f hours.\n' %(opt.num_epochs - start_epoch, (time.time()- t0)/ 3600))
  file_train_loss.close()
  file_val_loss.close()
  logger.close()

if __name__ == "__main__":
  opt = opts().parse()
  main(opt)
