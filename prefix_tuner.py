import torch
from torch import nn
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

# from ctrl_generator import top_k_logits

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "sep_token": "<sep>",
    "eos_token": "<eos>",
    "additional_special_tokens": ['<dialogue>', '<persona>', '<agent>', '<user>']
        # '<V0>', '<V1>', '<V2>', '<V3>', '<V4>', '<V5>',
        #  '<V6>', '<V7>', '<V8>', '<V9>', '<V10>',
}

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

class PrefixEmbTuning(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device =self.args.device

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path)
        model.resize_token_embeddings(len(self.tokenizer))
        config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        for param in model.base_model.parameters():
            param.requires_grad = False
        self.model = model

        print('under the PrefixEmbTuning model')
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        if hasattr(args, 'preseqlen'):
            self.preseqlen = args.preseqlen
        else:
            self.preseqlen = 1

        if hasattr(args, 'num_ctrl'):
            self.num_ctrl = args.num_ctrl
        else:
            self.num_ctrl = 11

        self.prefix_dropout = args.prefix_dropout

        if hasattr(args, 'mid_dim'):
            self.mid_dim = args.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(args, 'parametrize_emb'):
            self.parametrize_emb = args.parametrize_emb
        else:
            self.parametrize_emb = 'MLP'


        if True:
            # DIFFERENT PARAMETRIZATION:
            if True:
                if self.parametrize_emb == 'MLP':
                    print('MLP: UNDER PARAMETRIZATION 1 FOR embeddings. With the mid_dim = {}'.format(self.mid_dim))
                    self.input_tokens = torch.arange(self.preseqlen*self.num_ctrl).long()
                    self.wte = nn.Embedding(self.preseqlen*self.num_ctrl, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_embd))
                    self.get_prompt = self.get_prompt_p5
                elif self.parametrize_emb == 'Emb':
                    print('Emb: UNDER PARAMETRIZATION 2 FOR embeddings.')
                    self.input_tokens = torch.arange(self.preseqlen*self.num_ctrl).long()
                    self.wte = nn.Embedding(self.preseqlen*self.num_ctrl, config.n_embd)
                    self.get_prompt = self.get_prompt_p7

        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### print total # params #########
        # total_param = 0
        # for name, param in self.named_parameters():
        #     print(param.shape)
        #     total_param += param.numel()
        # print('total param is {}'.format(total_param))
        ############################################################################

    def get_prompt_p5(self, ctrl_ids, bsz=None):
        input_tokens = ((ctrl_ids * self.preseqlen).unsqueeze(-1).expand(-1, self.preseqlen) \
                       + torch.ones_like(ctrl_ids.unsqueeze(-1)) * torch.arange(self.preseqlen).unsqueeze(0)).to(self.device)
        temp_control = self.wte(input_tokens)
        assert bsz == len(input_tokens)
        # input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        input_embs = self.control_trans(temp_control) #bsz, seqlen, emb_dim
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = self.model(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values
        return past_key_values


    def get_prompt_p7(self, ctrl_ids, bsz=None):
        # input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        input_tokens = ((ctrl_ids * self.preseqlen).unsqueeze(-1).expand(-1, self.preseqlen) \
                       + torch.ones_like(ctrl_ids.unsqueeze(-1)) * torch.arange(self.preseqlen).unsqueeze(0)).to(self.device)
        assert bsz == len(input_tokens)
        input_embs = self.wte(input_tokens)
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = self.model(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values
        return past_key_values

    def forward(self, batch):
        context = batch[0]
        response = batch[1]
        ctrl_ids = batch[2]

        bsz = len(response)

        past_key_values_prompt = self.get_prompt(ctrl_ids, bsz=bsz)

        input_ids = [torch.LongTensor([
            [self.bos_token_id]
            + self.tokenizer.encode(context[i])
            + [self.sep_token_id]
            + self.tokenizer.encode(response[i])
            + [self.eos_token_id]
        ]).squeeze(0) for i in range(bsz)]

        input_ids = pad_sequence(input_ids, True, padding_value=self.tokenizer.pad_token_id).long()

        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = [torch.LongTensor([
            (len(self.tokenizer.encode(context[i])) + 2 ) * [-100]
            + self.tokenizer.encode(response[i])
            + [self.eos_token_id]
        ]).squeeze(0) for i in range(bsz)]
        labels = pad_sequence(labels, True, padding_value=-100).long().to(self.device)

        output = self.model(input_ids=input_ids.to(self.device),
                            past_key_values=past_key_values_prompt,
                            # attention_mask=attention_mask.to(self.device).half(),
                            labels=labels)

        loss = output[0]
        return loss

    def generate(self, context, ctrl_ids):

        past_key_values_prompt = self.get_prompt(torch.tensor([ctrl_ids]), bsz=1)

        input_ids = torch.LongTensor([
            [self.bos_token_id]
            + self.tokenizer.encode(context)
            + [self.sep_token_id]
        ]).to(self.device)

        output = []

        model_output = self.model(input_ids=input_ids.to(self.device),
                            past_key_values=past_key_values_prompt)

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