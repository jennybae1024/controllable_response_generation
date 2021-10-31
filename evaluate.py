import os
import sys
import time
import json
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from utils.args import get_args
from dataset import CtrlMSCDataset
from ctrl_generator import CtrlGenerator
from prefix_tuner import PrefixEmbTuning


model_class = {"prefix_emb_tuning": PrefixEmbTuning,
               "ctrl_generator": CtrlGenerator}

class Evaluator(object):
    def __init__(self, args):
        self.args = args

        # load test set
        self.model_path = self.args.model_name_or_path
        self.output_path = self.args.output_dir
        self.test_set = CtrlMSCDataset(self.args, split="test")
        self.test_loader = DataLoader(self.test_set, batch_size=1)

        self.model = model_class[self.args.model_type](self.args)
        self.model.to(self.args.device)
        # Load trained model weights
        self.last_ckpt = max([int(ele.split("-")[1]) for ele in os.listdir(self.args.output_dir) if ele.startswith("checkpoint-")])
        ckpt_path = os.path.join(self.args.output_dir, f"checkpoint-{self.last_ckpt}", "model_state.pth")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {ckpt_path}")

    def evaluate(self, tag=None):
        outfile_name = f"{self.output_path}/ckpt{self.last_ckpt}-generations.txt"
        self.model.eval()
        # generated_resps = []
        with open(outfile_name, 'w') as fout:
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(self.test_loader)):
                    output = self.model.generate(batch[0][0],
                                                 batch[2][0])
                    fout.write(output+"\n")
        fout.close()


def main():
    args = get_args()

    args.mode = "eval"
    start_time = time.time()
    evaluator = Evaluator(args)
    print(f"**************Evaluation start**************")
    evaluator.evaluate()
    print(f"Evaluation time: {(time.time() - start_time) / 60:.2f} min")


if __name__ == '__main__':
    main()
