import os, sys, json
import pandas as pd
from torch.utils.data import Dataset, DataLoader

data_dir = "/media/disk1/jennybae/data/controllable_generation"
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

