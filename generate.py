

import os
import sys
import time
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AdamW,
                          get_linear_schedule_with_warmup)
from dataset import CtrlTopicShiftDataset, CtrlTopicShiftTestDataset
from utils.args import get_args


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        # Load trained model weights
        self.model = AutoModelForCausalLM.from_pretrained(self.args.checkpoint).to(self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.checkpoint)
        print(f"Loaded checkpoint from {self.args.checkpoint}")

        # load test set
        self.test_set = CtrlTopicShiftTestDataset(self.args, split="test", tokenizer=self.tokenizer)
        self.test_loader = DataLoader(self.test_set, batch_size=1, collate_fn=self.test_set.collate_fn)
        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.test_set.special_token_values)

        # output dir path
        self.output_path = self.args.output_dir

    def evaluate(self, tag=None):
        if self.args.top_k:
            decoding_key=f"top_k_{self.args.top_k}"
        elif self.args.top_p:
            decoding_key = f"top_p_{self.args.top_p}"
        else:
            print("Specify decoding method!")
            assert 1==0
        ckpt = self.args.checkpoint.split("/")[-1].split("-")[1]
        outfile_name = f"{self.output_path}/ckpt{ckpt}-{decoding_key}-generations.txt"
        self.model.eval()
        # generated_resps = []
        with open(outfile_name, 'w') as fout:
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(self.test_loader)):
                    example = batch[0]
                    context, persona, topic_shift = example["dialog"], example["persona"], example["topic_shift"]
                    response_text = example["response_text"]
                    dial_id = example["dialog_id"]
                    current_output = []
                    # print(context)
                    # print(persona)
                    # print(response_text)
                    for i in range(self.args.max_sent_len):
                        instance, sequence = self.test_set.generate_input_seq(context, persona, current_output,
                                                                              topic_shift, with_eos=False)

                        input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
                        # print(input_ids)
                        model_outputs = self.model(input_ids=input_ids)
                        logits = model_outputs[0]

                        logits = logits[0, -1, :] / self.args.temperature
                        logits = top_filtering(logits, top_k=self.args.top_k, top_p=self.args.top_p)
                        probs = F.softmax(logits, dim=-1)

                        prev = torch.topk(probs, 1)[1] if self.args.no_sample else torch.multinomial(probs, 1)

                        if i < self.args.min_sent_len and prev.item() in self.special_tokens_ids:
                            while prev.item() in self.special_tokens_ids:
                                if probs.max().item() == 1:
                                    print("Warning: model generating special token with probability 1! Breaking...")
                                    break
                                prev = torch.multinomial(probs, num_samples=1)

                        if prev.item() in self.special_tokens_ids:
                            break
                        current_output.append(prev.item())

                    sampled_output_text = self.tokenizer.decode(current_output)
                    # print(dial_id)
                    # print("*************")
                    # print(current_output)
                    print(sampled_output_text)
                    print("*************")

                    fout.write("\t".join([sampled_output_text, response_text, dial_id])+"\n")
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
