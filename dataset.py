import os, sys, json, tqdm
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from itertools import chain

data_dir = "./data/controllable_generation"


###### Topic-shift triggering Response Generation TASK ########

TS_SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ['<dialog>', '<persona>',
                                  '<user>', '<agent>',
                                  '<topic-shift>', '<no-topic-shift>']}

class CtrlTopicShiftDataset(Dataset):
    def __init__(self, args, split, tokenizer=None):
        super().__init__()
        self.args = args
        self.split = split
        if not tokenizer:
            self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.tok.add_special_tokens(TS_SPECIAL_TOKENS)
        else:
            self.tok = tokenizer
        self.special_tokens = TS_SPECIAL_TOKENS
        self.special_token_values = ['<bos>', '<pad>', '<eos>', '<dialog>', '<persona>', '<user>', '<agent>',
                                     '<topic-shift>', '<no-topic-shift>']
        self.dial_tag, self.prsn_tag, self.user_tag, self.agent_tag, self.topic_shift, self.no_topic_shift = self.tok.convert_tokens_to_ids(
            self.special_tokens["additional_special_tokens"])
        self._create_examples()

    def _create_examples(self):
        data = json.load(
            open(os.path.join(data_dir, 'topic_shift', f"{self.args.datafile_name}_{self.split}.json"), 'r'))
        self.examples = []
        for sample in tqdm(data):
            context = self.tok.encode(sample["context"])[-self.args.max_context_len:]
            response = self.tok.encode(sample["response"])[:self.args.max_response_len]
            agent_persona = self.tok.encode(sample["agent_persona"])[-self.args.max_persona_len:]
            # is_topicShift = sample["topic_shift_0.25"]
            is_topicShift = sample["topic_shift_by_tsd_orig"]


            self.examples.append({
                "dialog": context,
                "persona": agent_persona,
                "response": response,
                "topic_shift": is_topicShift,
                "response_text": sample["response"],
                "dialog_id": "-".join([sample["session"], str(sample["dial_idx"])])
            })

    def __len__(self):
        return len(self.examples)

    def generate_input_seq(self, context, persona, response, topic_shift=False, with_eos=True):
        instance = {}
        control_code = self.topic_shift if topic_shift else self.no_topic_shift
        sequence = [[self.tok.bos_token_id] + [control_code] + [self.prsn_tag] + persona] + [
            [self.dial_tag] + context] + [response + ([self.tok.eos_token_id] if with_eos else [])]
        instance["input_ids"] = list(chain(*sequence))
        instance["attention_mask"] = [1] * len(instance["input_ids"])
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + sequence[-1]

        return instance, sequence

    def __getitem__(self, idx):
        example = self.examples[idx]
        instance, _ = self.generate_input_seq(example["dialog"],
                                           example["persona"],
                                           example["response"],
                                           example["topic_shift"]
                                           )
        return instance

    def collate_fn(self, batch):
        input_ids = [torch.LongTensor(ele["input_ids"]) for ele in batch]
        attention_mask = [torch.LongTensor(ele["attention_mask"]) for ele in batch]
        lm_labels = [torch.LongTensor(ele["lm_labels"]) for ele in batch]

        input_ids = torch.LongTensor(pad_sequence(input_ids, True, padding_value=self.tok.pad_token_id))
        attention_mask = torch.LongTensor(pad_sequence(attention_mask, True, padding_value=self.tok.pad_token_id))
        lm_labels = torch.LongTensor(pad_sequence(lm_labels, True, padding_value=self.tok.pad_token_id))

        return input_ids, attention_mask, lm_labels

class CtrlTopicShiftTestDataset(CtrlTopicShiftDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        return example

    def collate_fn(self, batch):
        return batch


###### Question Style Response Generation TASK ########
# TODO update CtrlMSCDataset into CtrlTopicShiftDataset style
#  s.t. no need to define task-specific models when using PLMs

prefix = {k: f"<V{k}> " for k in range(11)}

def gen_hist_and_agent(history, aprsn, ctrl):
    del ctrl
    history = " ".join(history.split()[-200:])
    aprsn = " ".join(aprsn.split()[:128])
    return ("<persona> " + aprsn + " <dialogue> " + history)

def gen_qn_feat_code_with_agent(history, aprsn, ctrl):
    history = " ".join(history.split()[-200:])
    aprsn = " ".join(aprsn.split()[:128])
    return (prefix[ctrl] + "<persona> " + aprsn + " <dialogue> " + history)

context_fn = {"msc_qn_featCode_agent_persona": gen_qn_feat_code_with_agent,
             "msc_hist_and_agent_persona": gen_hist_and_agent}

class CtrlMSCDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        self.load_data()

    def load_data(self):
        raw_df = pd.read_csv(os.path.join(data_dir, f"{self.args.datafile_name}_{self.split}.csv"), index_col=0)
        raw_df.reset_index(drop=True, inplace=True)

        context = raw_df["history"].to_list()
        agent_persona = raw_df["agent_persona"].to_list()
        response = raw_df["response"].to_list()
        ctrl_ids = raw_df["is_qn_buck"].to_list()
        assert len(context) == len(agent_persona) == len(response) == len(ctrl_ids)

        del raw_df

        self.context = [context_fn[self.args.dataset_name](context[idx],
                                                           agent_persona[idx],
                                                           ctrl_ids[idx]) for idx in range(len(context))]
        self.response = [" ".join(resp.split()[:self.args.max_sent_len]) for resp in response]

        self.ctrl_ids = ctrl_ids

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        return self.context[idx], self.response[idx], self.ctrl_ids[idx]





