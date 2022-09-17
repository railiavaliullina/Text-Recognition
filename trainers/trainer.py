import numpy as np
from torch import nn
import torch
import os
import time

from dataloader.dataloader import get_dataloaders
from model.FullyConvNet import get_model
# from losses.ctc_loss import CTCLoss
from losses.ctc_loss_1 import CTCLoss
from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dl_train, self.dl_test, self.dl_val = get_dataloaders()
        self.model = get_model()
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.logging = None

    @staticmethod
    def get_criterion():
        """
        Gets criterion.
        :return: criterion
        """
        criterion = CTCLoss()
        return criterion

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        # optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.cfg.lr,
        #                                }])  # 'weight_decay': self.cfg.weight_decay
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                                    momentum=0.9)
        return optimizer

    def restore_model(self):
        """
        Restores saved model.
        """
        if self.cfg.load_saved_model:
            print(f'Trying to load checkpoint from epoch {self.cfg.epoch_to_load}...')
            try:
                checkpoint = torch.load(self.cfg.checkpoints_dir + f'/checkpoint_{self.cfg.epoch_to_load}.pth')
                load_state_dict = checkpoint['model']
                self.model.load_state_dict(load_state_dict)
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                self.optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint from epoch {self.cfg.epoch_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')

    def save_model(self):
        """
        Saves model.
        """
        if self.cfg.save_model and self.epoch % self.cfg.epochs_saving_freq == 0:
            print('Saving current model...')
            state = {
                'model': self.model.state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': self.optimizer.state_dict()
            }
            if not os.path.exists(self.cfg.checkpoints_dir):
                os.makedirs(self.cfg.checkpoints_dir)

            path_to_save = os.path.join(self.cfg.checkpoints_dir, f'checkpoint_{self.epoch}.pth')
            torch.save(state, path_to_save)
            print(f'Saved model to {path_to_save}.')

    def evaluate(self, dl, set_type):
        """
        Evaluates model performance. Calculates and logs model accuracy on given data set.
        :param dl: train or test dataloader
        :param set_type: 'train' or 'test' data type
        """
        all_predictions, all_labels, losses = [], [], []

        self.model.eval()
        with torch.no_grad():
            print(f'Evaluating on {set_type} data...')
            eval_start_time = time.time()

            correct_predictions, total_predictions = 0, 0
            dl_len = len(dl)
            for i, batch in enumerate(dl):
                input_vector, labels = batch[0].cuda(), batch[1].cuda()

                if i % 3e2 == 0:
                    print(f'iter: {i}/{dl_len}')

                out = self.model(input_vector)
                _, predictions = torch.max(out.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += torch.sum(predictions == labels)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                loss = self.criterion(out, labels)
                losses.append(loss.item())

            accuracy = 100 * correct_predictions.item() / total_predictions
            mean_loss = np.mean(losses)
            print(f'Accuracy on {set_type} data: {accuracy} %, {set_type} error: {100 - accuracy} %, loss: {mean_loss}')
            print(f'Evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')
        self.model.train()

    def decode_pred_and_label(self, outputs, labels):
        alphabet_chars = list(dataset_cfg.alphabet.keys())
        alphabet_num = list(dataset_cfg.alphabet.values())
        labels = [''.join([alphabet_chars[alphabet_num.index(l_)] for l_ in l if l_ != 0]) for l in labels]

        outputs_ = torch.transpose(outputs, 2, 0).cpu().detach().numpy()
        preds = [[alphabet_chars[alphabet_num.index(idx)] for idx in np.argmax(o, 1)] for o in outputs_]

        fin_preds = []
        for pred in preds:
            pred_ = [str(pred[0])]
            for i in range(len(pred) - 1):
                if pred[i + 1] != pred[i]:
                    pred_.append(str(pred[i + 1]))
            fin_preds.append(''.join([p for p in pred_]).replace("-", ""))
        return fin_preds, labels

    def train(self):
        self.start_epoch, self.global_step = 0, -1

        for e in range(self.start_epoch, self.start_epoch + self.cfg.epochs):
            for i, batch in enumerate(self.dl_train):
                img, labels, real_w = batch
                # if i == 3:
                #     break

            # for it in range(10000):
                logits = self.model(img)  # .cpu().detach().numpy()
                outputs = nn.Softmax(dim=1)(logits)

                outputs = outputs.detach()  # .detach().cpu().numpy()

                # print(f'outputs: {outputs}')
                # print(f'argmax: {np.argmax(outputs[0].numpy(), 0)}')

                loss = self.criterion(outputs, labels, real_w)

                self.optimizer.zero_grad()
                grad = self.criterion.backward()
                logits.backward(grad)
                self.optimizer.step()

                pred, label = self.decode_pred_and_label(torch.transpose(torch.transpose(outputs, 0, 1), 1, 2), labels)

                print(f'Iter: {i}\n'
                      f'Loss: {loss}\n'
                      f'GT: {label}\n'
                      f'Pred: {pred}\n\n')

                self.global_step += 1

                # pred_ = pred[0].replace('-', '')
                # if pred_ == label:
                #     break
                # a = 1


np.random.seed(0)
torch.random.manual_seed(0)
# torch.ctc_loss()
trainer = Trainer(train_cfg)
trainer.train()
