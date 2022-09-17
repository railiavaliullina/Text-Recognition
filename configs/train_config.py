from easydict import EasyDict

cfg = EasyDict()

cfg.batch_size = 8
cfg.lr = 1e-4
cfg.weight_decay = 5e-4
cfg.epochs = 100

cfg.epochs = 100
cfg.reg_lambda = 1e-4

cfg.log_metrics = False
cfg.experiment_name = 'ctc_loss'

cfg.evaluate_on_train_set = False
cfg.evaluate_before_training = False
# eval_plots_dir = f'../saved_files/plots/{experiment_name}/'

cfg.load_saved_model = False
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 21001
cfg.save_model = False
cfg.epochs_saving_freq = 1
