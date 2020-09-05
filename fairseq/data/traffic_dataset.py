
import os
import numpy as np
import sys
import torch
import torch.nn.functional as F

import pandas as pd
import itertools

try:
    import wandb
except Exception as e:
    print(e)

from . import FairseqDataset

from itertools import cycle, islice


class TrafficDataset(FairseqDataset):
    def __init__(self, csv_file, output_seq_len=360,
                scale_input=True, scale_output=True, input_seq_len=1440,
                num_segments=12, variables_per_segment=0,
                max_vals=None,
                train_from="2019-08-01 00:00:00",
                train_to="2019-09-01 00:00:00",
                valid_from="2019-07-01 00:00:00",
                valid_to="2019-08-01 00:00:00",
                test_from="2019-09-01 00:00:00",
                test_to="2019-10-01 00:00:00",
                mainlines_to_include_in_input=None,
                mainlines_to_include_in_output=None,
                active_onramps=None,
                active_offramps=None,
                shuffle=True, input_feeding=True, 
                max_sample_size=None, min_sample_size=None,split='train'
                ):
        super().__init__()
        
        self.num_segments = num_segments
        self.variables_per_segment = variables_per_segment

        self.output_seq_len = output_seq_len
        self.input_seq_len = input_seq_len
        self.scale_input = scale_input
        self.scale_output = scale_output
        self.input_feeding = input_feeding

        self.max_vals = max_vals*self.num_segments 
        
        self.all_data = pd.read_csv(csv_file,index_col=0,parse_dates=[0])
        print("##Length of Whole Dataset: ",len(self.all_data))

        # input_cols = ['Seg00_q', 'Seg00_speed','Seg04_q', 'Seg04_speed','Seg04_r', 'Seg02_s']
        input_cols = ['q0', 'v0', 'q2', 'v2', 'rho5', 'beta2', 'r4']
        self.all_data = self.all_data.loc[:,input_cols]
        self.all_data = self.all_data.replace(np.inf, np.nan)
        self.all_data = self.all_data.replace(-np.inf, np.nan)
        self.all_data.loc[:,['v0','v2']] = self.all_data.loc[:,['v0','v2']].fillna(100.0)
        self.all_data.loc[:,['q0', 'q2', 'rho5', 'beta2', 'r4']] = self.all_data.loc[:,['q0', 'q2', 'rho5', 'beta2', 'r4']].fillna(0.0)
        
        self.split = split
        if split == 'train':
            train_from_idx = self.all_data.index.get_loc(pd.to_datetime(train_from), method='nearest')
            train_to_idx = self.all_data.index.get_loc(pd.to_datetime(train_to), method='nearest')
            self.all_data = self.all_data.iloc[train_from_idx:train_to_idx, :]
            print("t##Length of Train Dataset: ", len(self.all_data))
            self.shuffle = shuffle
        elif split == 'valid':
            valid_from_idx = self.all_data.index.get_loc(pd.to_datetime(valid_from), method='nearest')
            valid_to_idx = self.all_data.index.get_loc(pd.to_datetime(valid_to), method='nearest')
            self.all_data =self.all_data.iloc[valid_from_idx:valid_to_idx, :]
            print("v##Length of Valid Dataset: ", len(self.all_data))
            self.shuffle = False
        else:
            test_from_idx = self.all_data.index.get_loc(pd.to_datetime(test_from), method='nearest')
            test_to_idx = self.all_data.index.get_loc(pd.to_datetime(test_to), method='nearest')
            self.all_data = self.all_data.iloc[test_from_idx:test_to_idx, :]
            print("t??##Length of Test Dataset: ",len(self.all_data))
            self.shuffle = False

        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size

        print(self.all_data.columns)
 

        # ['Seg00_q', 'Seg00_speed','Seg04_q', 'Seg04_speed','Seg04_r', 'Seg02_s']
        # ['q0', 'v0', 'q2', 'v2', 'rho5', 'beta2', 'r4']
        # self.dataset_means = torch.Tensor([[2369.76, 92.85, 1862.98, 95.25, 6.79, 0.2, 299.92]])
        # self.dataset_std = torch.Tensor([[1573.51, 9.69, 1271.13, 10.27, 5.76, 0.21, 309.29]])

        self.all_means = [3000., 90., 3000., 90., 7., 0.2, 300.]
        self.all_stds  = [2000., 10., 2000., 10., 6., 0.2, 300.]
        
        self.all_means = np.array(self.all_means)
        self.all_stds = np.array(self.all_stds)
        
        print("means:",self.all_means)
        print("stds:",self.all_stds)

        self.all_data = (self.all_data-self.all_means)/self.all_stds
        self.all_input_data = self.all_data
        
        self.shuffle = shuffle
    
    def get_means_stds(self):
        return (torch.Tensor(self.all_means),torch.Tensor(self.all_stds))
    
    def get_max_vals(self):
        return torch.Tensor(self.max_vals)
    
    def __getitem__(self, index):
        idx = index
        # if not self.split=='test':
        #     rand = np.random.randint(self.output_seq_len, size=1)[0]
        #     idx = (index * (self.output_seq_len-1)) + rand
        # else:
        #     idx = index

        input_len = self.input_seq_len
        label_len = self.output_seq_len

        if self.scale_input:
            one_input = self.all_input_data.iloc[idx:idx+input_len, :].values

        one_label = self.all_data.iloc[idx+input_len:idx+input_len+label_len, :].values

        one_label = one_label.transpose(0,1)

        return {
            'id': index,
            'source': one_input.astype('float'),
            'target': one_label.astype('float'),
        }

    def __len__(self):
        if not self.split=='test':
            data_len = (len(self.all_data) - (1*self.output_seq_len+self.input_seq_len*1) - 512)#//self.output_seq_len
        else:
            data_len = (len(self.all_data) - (1*self.output_seq_len+self.input_seq_len*1) - 512)
        return data_len


    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])

        src_tokens = torch.FloatTensor([s['source'] for s in samples])
        src_lengths = torch.LongTensor([len(s['source']) for s in samples])

        target = torch.FloatTensor([s['target'] for s in samples])

        ntokens = sum(len(s['target']) for s in samples)

        if self.scale_input and self.scale_output:
            last_inputs = [s['source'][-1:] for s in samples]
            previous_output = [np.concatenate([l,s['target'][:-1]]) for s,l in zip(samples,last_inputs)]

        if self.input_feeding:
            prev_output_tokens = torch.FloatTensor(previous_output)
        else:
            prev_output_tokens = None
        
        try:
            #print("src_tokens",src_tokens.mean())
            if src_tokens.mean().item()==np.nan:
                print(src_tokens)
                from fairseq import pdb; pdb.set_trace();
        except Exception as e:
            print(e)
            from fairseq import pdb; pdb.set_trace();

        try:
            #print("target",target.mean())
            if target.mean().item()==np.nan:
                print(target)
                from fairseq import pdb; pdb.set_trace();
        except Exception as e:
            print(e)
            from fairseq import pdb; pdb.set_trace();
        
        
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }

        try:
            if pd.DataFrame(src_tokens).mean().mean()==np.nan:
                print("src_tokens",src_tokens,src_tokens.mean())
        except Exception as e:
            print(e)
            from fairseq import pdb; pdb.set_trace();

        try:
            print(pd.DataFrame(target).median().T)
        except Exception as e:
            print(e)
            from fairseq import pdb; pdb.set_trace();

        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return len(self.all_data.iloc[index:index+self.output_seq_len, :].values.reshape(-1))
    



