import os, sys, json, time
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import CtrlMSCDataset
from ctrl_generator import CtrlGenerator
from utils.args import get_args
from tqdm import tqdm
from tensorboardX import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # load train and dev sets
        self.train_set = CtrlMSCDataset(self.args, split="train")
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.dev_set = CtrlMSCDataset(self.args, split="valid")
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size)

        # load model
        self.model = CtrlGenerator(self.args)
        self.model_best_params = {}

    # save model checkpoint to local directory
    def save(self, epoch, step):
        ckpt_path = os.path.join(self.args.output_dir, f"epoch_{epoch}_step_{step}")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        checkpoint = {"model": self.model.state_dict()}
        torch.save(checkpoint, os.path.join(ckpt_path, "model_state.pth"))
        print(f"*** Model checkpoint saved to {ckpt_path}/model_state.pth ***")

    # calculate dev loss and update best model parameters
    def update_best(self, epoch, step, train_loss):
        dev_loss = self.dev()
        print(f"*** Epoch {epoch} Step {step}/{len(self.train_loader)}: ", \
              f"train loss {train_loss / step:.4f}, dev loss {dev_loss:.4f} ***")
        self.model_best_params = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if dev_loss < self.best_dev:
            self.best_dev = dev_loss
        self.save(epoch, step)

        return dev_loss

    def train(self):
        self.best_dev = 999999, 0

        tb_writer = SummaryWriter(self.args.output_dir)
        params = []
        params.append({'params': self.model.model.parameters(), 'lr': self.args.lr})
        # optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        t_total = len(self.train_loader) // self.args.gradient_accumulation_steps * self.args.num_epochs
        optimizer = AdamW(params, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        self.best_dev = self.dev()
        print(f"*** Initial dev loss: {self.best_dev:.4f} ***")

        global_step = 0
        self.model.zero_grad()

        for epoch in range(self.args.num_epochs):
            train_loss = 0
            for step, batch in enumerate(tqdm(self.train_loader), start=1):
                self.model.train()
                loss = self.model.forward(batch)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                train_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    global_step += 1

                # evaluate dev loss every 500 steps
                if step % (self.args.dev_at_step) == 0:
                    dev_loss = self.update_best(epoch, step, train_loss)
                    tb_writer.add_scalar("eval_ppl", dev_loss, global_step)

            # scheduler.step()
            train_loss /= step
            tb_writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("train/loss", train_loss, global_step)

            _ = self.update_best(epoch, step, train_loss)

    # calculate dev loss and print intermediate outputs
    def dev(self):
        self.model.eval()
        tot_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dev_loader), start=1):
                loss = self.model.forward(batch)
                tot_loss += loss.item()
                if step % 100 == 0:
                    output = self.model.generate(batch[0][0])
                    print(f"dev sentence from: {batch[0][0]}")
                    print(f"dev gold resp: {batch[1][0]}")
                    print(f"dev sentence to: {output}")
        return tot_loss / step


def main():
    args = get_args()
    start_time = time.time()
    trainer = Trainer(args)
    print(f"**************start training**************")
    trainer.train()
    print(f"**************end training**************")
    print(f"training time: {(time.time()-start_time)/60:.2f} min")

if __name__ == '__main__':
    main()
