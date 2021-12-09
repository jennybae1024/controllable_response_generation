import os, sys, json, time
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import CtrlTopicShiftDataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AdamW,
                          get_linear_schedule_with_warmup)
from utils.args import get_args
from tqdm import tqdm
from tensorboardX import SummaryWriter

TS_SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ['<dialog>', '<persona>',
                                  '<user>', '<agent>',
                                  '<topic-shift>', '<no-topic-shift>']}

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.add_special_tokens(TS_SPECIAL_TOKENS)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.special_tokens_values = ['<bos>', '<pad>', '<eos>', '<dialog>', '<persona>',
                                      '<user>', '<agent>', '<topic-shift>', '<no-topic-shift>']

        self.model_best_params = {}

        # TODO should be able to interchange the dataset type by args,
        #  not fixed to the topic-shift dataset
        # load train and dev sets
        self.train_set = CtrlTopicShiftDataset(self.args, split="train", tokenizer=self.tokenizer)
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, collate_fn=self.train_set.collate_fn, shuffle=True, drop_last=True)
        self.dev_set = CtrlTopicShiftDataset(self.args, split="valid", tokenizer=self.tokenizer)
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size,  collate_fn=self.dev_set.collate_fn)


    def save(self, step):
        ckpt_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        model_to_save.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        torch.save(self.args, os.path.join(ckpt_path, "training_args.bin"))

        print(f"*** Model checkpoint saved to {ckpt_path} ***")


    def update_best(self, epoch, step, train_loss, global_step, epoch_end = None):
        dev_loss = self.dev()
        if epoch_end:
            print(f"*** Epoch {epoch} Step {step}/{len(self.train_loader)}: ", \
                  f"train loss {train_loss :.4f}, dev loss {dev_loss:.4f} ***")
        else:
            print(f"*** Epoch {epoch} Step {step}/{len(self.train_loader)}: ", \
              f"train loss {train_loss / step:.4f}, dev loss {dev_loss:.4f} ***")
        self.model_best_params = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if dev_loss < self.best_dev:
            self.best_dev = dev_loss
        self.save(global_step)

        return dev_loss

    def train(self):
        self.best_dev = 999999, 0

        tb_writer = SummaryWriter(self.args.output_dir)
        params = []
        params.append({'params': self.model.parameters(), 'lr': self.args.lr})
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
                input_ids, attention_mask, labels = batch
                model_outputs = self.model(input_ids=input_ids.to(self.device),
                                  attention_mask=attention_mask.to(self.device),
                                  labels=labels.to(self.device))
                loss = model_outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                train_loss += loss.item()
                global_step += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()

                # evaluate dev loss every 500 steps
                if step % (self.args.dev_at_step) == 0:
                    dev_loss = self.update_best(epoch, step, train_loss, global_step)
                    tb_writer.add_scalar("eval_ppl", dev_loss, global_step)
                    tb_writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train/loss", train_loss/step, global_step)

            # scheduler.step()
            train_loss /= step
            tb_writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("train/loss", train_loss, global_step)

            _ = self.update_best(epoch, step, train_loss, global_step, epoch_end=True)

    # calculate dev loss and print intermediate outputs
    def dev(self):
        self.model.eval()
        tot_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dev_loader), start=1):
                input_ids, attention_mask, labels = batch
                model_outputs = self.model(input_ids=input_ids.to(self.device),
                                  attention_mask=attention_mask.to(self.device),
                                  labels=labels.to(self.device))
                loss = model_outputs[0]
                tot_loss += loss.item()
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
