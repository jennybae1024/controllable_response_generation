import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "sep_token": "<sep>",
    "eos_token": "<eos>",
    "additional_special_tokens": ['<V0>', '<V1>', '<V2>', '<V3>', '<V4>', '<V5>',
                                  '<V6>', '<V7>', '<V8>', '<V9>', '<V10>',
                                  '<dialogue>', '<persona>', '<agent>', '<user>']
}


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


class CtrlGenerator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self.args.device

        # load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def forward(self, batch):
        context = batch[0]
        # print(max([len(ele.split()) for ele in context]))
        response = batch[1]
        bz = len(response)
        input_ids = [torch.LongTensor([
            [self.bos_token_id]
            + self.tokenizer.encode(context[i])
            + [self.sep_token_id]
            + self.tokenizer.encode(response[i])
            + [self.eos_token_id]
        ]).squeeze(0) for i in range(bz)]

        input_ids = pad_sequence(input_ids, True, padding_value=self.tokenizer.pad_token_id).long().to(self.device)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = [torch.LongTensor([
            (len(self.tokenizer.encode(context[i])) + 2) * [-100]
            + self.tokenizer.encode(response[i])
            + [self.eos_token_id]
        ]).squeeze(0) for i in range(bz)]
        labels = pad_sequence(labels, True, padding_value=-100).long().to(self.device)
        output = self.model(input_ids=input_ids.to(self.device),
                            attention_mask=attention_mask.to(self.device).half(),
                            labels=labels.to(self.device))
        loss = output[0]
        return loss

    def generate(self, context, ctrl_ids):
        del ctrl_ids
        input_ids = torch.LongTensor([
            [self.bos_token_id]
            + self.tokenizer.encode(context)
            + [self.sep_token_id]
        ]).to(self.device)

        output = []
        org_input_ids = input_ids

        model_output = self.model(input_ids=input_ids.to(self.device), \
                                  past_key_values=None)
        logits = model_output.logits
        past = model_output.past_key_values

        regenerate_try = 0
        count = 0
        while True:
            logits = logits[:, -1, :] / self.args.temperature
            logits_topk = top_k_logits(logits, k=self.args.top_k)
            stmax_logits = F.softmax(logits_topk, dim=-1)
            input_ids = torch.multinomial(stmax_logits, num_samples=1)
            count += 1
            next_token = self.tokenizer.decode(input_ids.item())
            # if input_ids == self.eos_token_id:
            if input_ids == self.eos_token_id or \
                    (input_ids == 199 and output):
                break

            # if > 30 tokens generated, regenerate from beginning
            if count > self.args.max_sent_len or (input_ids == 199 and not output):
                count = 0
                regenerate_try += 1
                if regenerate_try > self.args.regenerate_try:
                    print("regeneration try exceeds max.")
                    #                     print(f"original input: {context}")
                    #                     print(f"prev output: {''.join(output)}")
                    # output = []
                    # output.append("EXCEED MAX LENGTH.")
                    break
                else:
                    output = []
                    model_output = self.model(input_ids=input_ids.to(self.device), \
                                              past_key_values=None)
                    logits = model_output.logits
                    past = model_output.past_key_values
                    continue
            output.append(next_token)

            model_output = self.model(input_ids=input_ids.to(self.device), \
                                      past_key_values=past)
            logits = model_output.logits
            past = model_output.past_key_values

        return ("".join(output))