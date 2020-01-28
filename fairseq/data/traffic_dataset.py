
import os
import numpy as np
import sys
import torch
import torch.nn.functional as F

import pandas as pd
import itertools

try:
    import wandb
    # wandb.init("traffic_calibration")
except Exception as e:
    print(e)

from . import FairseqDataset

from itertools import cycle, islice

#TODO: Delete train_size

class TrafficDataset(FairseqDataset):
    def __init__(self, csv_file, output_seq_len=360,
                scale_input = True, scale_output = True, input_seq_len=1440,
                num_segments = 12, variables_per_segment = 4,
                max_vals = [10000,100,5000,5000],
                last_train_datetime = "2018-03-01 00:00:00",
                shuffle=True, input_feeding=True, 
                max_sample_size=None, min_sample_size=None,split='train'
                ):
        super().__init__()
        
        self.last_train_datetime = pd.to_datetime(last_train_datetime)

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

        #get only num_segments
        total_input_variables = self.num_segments*self.variables_per_segment
        if total_input_variables!=self.all_data.shape[1]:
            print("total_input_variables:",total_input_variables)
            print("self.all_data.shape[1]:",self.all_data.shape[1],"trimming cols")
        # self.all_data = self.all_data.iloc[:,2*4:total_input_variables+2*4]
        self.all_data = self.all_data.iloc[:,:total_input_variables]

        last_train_idx = self.all_data.index.get_loc(self.last_train_datetime, method='nearest')
        self.train_size = last_train_idx#int(len(self.all_data)*2//3)#100000#360*16
        
        valid_size = len(self.all_data) - self.train_size

        # vol_multiple = 120.
        # self.all_data.iloc[:,::self.variables_per_segment] = self.all_data.iloc[:,::self.variables_per_segment] * vol_multiple
        # self.all_data.iloc[:,2::self.variables_per_segment] = self.all_data.iloc[:,2::self.variables_per_segment] * vol_multiple
        # self.all_data.iloc[:,3::self.variables_per_segment] = self.all_data.iloc[:,3::self.variables_per_segment] * vol_multiple

        if split == 'train':
            self.all_data = self.all_data.iloc[:self.train_size, :]
            print("t##Length of Train Dataset: ", len(self.all_data))
            self.shuffle = shuffle
        elif split == 'valid':
            self.all_data =self.all_data.iloc[self.train_size:self.train_size+valid_size, :]
            print("v##Length of Valid Dataset: ", len(self.all_data))
            self.shuffle = False
        else:
            self.all_data = self.all_data.iloc[self.train_size+valid_size:, :]
            print("t??##Length of Test Dataset: ",len(self.all_data))
            self.shuffle = False

        broken_detector_id = 4*7
        simulate_detector_breakdown = True
        if simulate_detector_breakdown == True and split!='train':
            self.all_data.iloc[:, broken_detector_id] = -1e-6
        
        simulate_no_detector = True
        if simulate_no_detector == True:
            self.all_data.iloc[:, broken_detector_id] = -1e-6


        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size

        # self.shuffle = shuffle
    
    def get_max_vals(self):
        return self.max_vals
    
    def __getitem__(self, index):

        #from fairseq import pdb; pdb.set_trace()

        #rand = torch.randint(0, self.output_seq_len, (1,))[0].item()#0#torch.randint(0, self.output_seq_len, (1,))[0].item()
        idx = index * 1 #* self.output_seq_len + rand#(index+rand) #* self.output_seq_len

        input_len = self.input_seq_len
        label_len = self.output_seq_len

        NEG = -1e-3

        one_input = self.all_data.iloc[idx:idx+input_len, :].values
        if self.scale_input:
          one_input = one_input/self.max_vals
        #one_input = np.reshape(one_input,-1)
        
        one_label = self.all_data.iloc[idx+input_len:idx+input_len+label_len, :].values
        if self.scale_output:
          one_label = one_label/self.max_vals
        #one_label = np.reshape(one_label,-1)
        one_label = one_label.transpose(0,1)#1, 2)

        # print(one_label.size())
        # from fairseq import pdb; pdb.set_trace()
        return {
            'id': index,
            'source': one_input.astype('float'),
            'target': one_label.astype('float'),
        }

    def resample(self, x, factor):
        return F.interpolate(x.view(1, 1, -1), scale_factor=factor).squeeze()

    def __len__(self):
        return (len(self.all_data) - (1*self.output_seq_len+self.input_seq_len) - 1)//1#self.output_seq_len#- self.output_seq_len# - 1 #- 4* self.output_seq_len# - 2 * self.output_seq_len - 1


    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        #from fairseq import pdb; pdb.set_trace()
        src_tokens = torch.FloatTensor([s['source'] for s in samples])
        src_lengths = torch.LongTensor([len(s['source']) for s in samples])

        target = torch.FloatTensor([s['target'] for s in samples])
        #target = target.transpose(1,2)
        ntokens = sum(len(s['target']) for s in samples)
        #max_list = list(islice(cycle(self.max_vals), 32400))
        #from fairseq import pdb; pdb.set_trace();
        previous_output = [s['target'][:-1] for s in samples] # [samples[0]['target'][0]]
        # from fairseq import pdb; pdb.set_trace();
        previous_output = [np.insert(previous_output[x],0,samples[x]['source'][-1],axis=0) for x in range(len(samples))]
        #from fairseq import pdb; pdb.set_trace();
        # previous_output = [samples[0]['target']] + [s['target'] for s in samples[:-1]] #BUG: not quite right
        prev_output_tokens = torch.FloatTensor(previous_output)

        # prev_output_tokens = None
        # target = None
        # if samples[0].get('target', None) is not None:
        #     target = torch.FloatTensor([s['target'] for s in samples])
        #     ntokens = sum(len(s['target']) for s in samples)

        #     if self.input_feeding:
        #         # we create a shifted version of targets for feeding the
        #         # previous output token(s) into the next decoder step
        #         # previous_output = itertools.chain([samples[-1]['source']],[s['target'] for s in samples[:-1]])
        #         previous_output = [self.max_vals*samples[-1]['source']] + [s['target'] for s in samples[:-1]]
        #         prev_output_tokens = torch.FloatTensor(previous_output)
        #         # prev_output_tokens = merge(
        #         #     'target',
        #         #     left_pad=left_pad_target,
        #         #     move_eos_to_beginning=True,
        #         # )
        #         # prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        # else:
        #     ntokens = sum(len(s['source']) for s in samples)
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
        # batch = {
        #     'id': id,
        #     'net_input': {
        #         'sources': sources,
        #         'src_lens': torch.ones(sources.size())
        #     },
        #     'target': target,
        # }
        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens
        # from fairseq import pdb; pdb.set_trace()
        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return len(self.all_data.iloc[index:index+self.output_seq_len, :].values.reshape(-1))
    

    # def ordered_indices(self):
    #     """Return an ordered list of indices. Batches will be constructed based
    #     on this order."""

    #     if self.shuffle:
    #         order = [np.random.permutation(len(self))]
    #     else:
    #         order = [np.arange(len(self))]

    #     order.append(self.output_seq_len*np.ones(len(self)))#self.sizes)
    #     from fairseq import pdb; pdb.set_trace()
    #     return np.lexsort(order)
