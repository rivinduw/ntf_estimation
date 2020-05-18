
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
                scale_input=True, scale_output=True, input_seq_len=1440,
                num_segments=12, variables_per_segment=0,
                max_vals=None,
                train_from="2018-08-01 00:00:00",
                train_to="2018-09-01 00:00:00",
                valid_from="2018-07-01 00:00:00",
                valid_to="2018-08-01 00:00:00",
                test_from="2018-09-01 00:00:00",
                test_to="2018-10-01 00:00:00",
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

        #get only num_segments
        total_input_variables = self.num_segments*self.variables_per_segment
        self.all_data = self.all_data.loc[:,[x for x in self.all_data.columns if not ('_q' in x)]]
        self.all_data = self.all_data.iloc[:,:total_input_variables]
        #make the extreme values equal to not found 
        #self.all_data[self.all_data.quantile(0.99)<=self.all_data] = -1e-6
        
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


        if mainlines_to_include_in_input == None:
            mainlines_to_include_in_input = [1.0]*self.num_segments
            mainlines_to_include_in_input[1:-1] = [0.0]*len(mainlines_to_include_in_input[1:-1])
            # mainlines_to_include_in_input[15] = 0.0
            # mainlines_to_include_in_input[16] = 0.0
            # mainlines_to_include_in_input[17] = 0.0
            mainlines_to_include_in_input[0] = 1.0
            mainlines_to_include_in_input[-1] = 1.0
            self.mainlines_to_include_in_input = np.array(mainlines_to_include_in_input)
        else:
            self.mainlines_to_include_in_input = np.array(mainlines_to_include_in_input)
        
        if mainlines_to_include_in_output == None:
            mainlines_to_include_in_output = [1.0]*self.num_segments
            mainlines_to_include_in_output[1:-1] = [0.0]*len(mainlines_to_include_in_output[1:-1])
            mainlines_to_include_in_output[0] = 1.0
            mainlines_to_include_in_output[-1] = 1.0
            # mainlines_to_include_in_output[15] = 0.0
            # mainlines_to_include_in_output[16] = 0.0
            # mainlines_to_include_in_output[17] = 0.0
            # mainlines_to_include_in_output[0] = 1.0
            # mainlines_to_include_in_output[-1] = 1.0
            self.mainlines_to_include_in_output = np.array(mainlines_to_include_in_output)
        else:
            self.mainlines_to_include_in_output = np.array(mainlines_to_include_in_output)

        all_active_vars = np.zeros(total_input_variables)
        all_active_vars[::self.variables_per_segment] = self.mainlines_to_include_in_input
        all_active_vars[1::self.variables_per_segment] = self.mainlines_to_include_in_input
        all_active_vars[2::self.variables_per_segment] = np.array(active_onramps)
        all_active_vars[3::self.variables_per_segment] = np.array(active_offramps)
        # all_active_vars[2::self.variables_per_segment] = self.mainlines_to_include_in_input
        # all_active_vars[3::self.variables_per_segment] = np.array(active_onramps)
        # all_active_vars[4::self.variables_per_segment] = np.array(active_offramps)
        all_active_vars = all_active_vars.astype(bool).tolist()
        
        # broken_detector_id = self.variables_per_segment*10
        # simulate_detector_breakdown = True
        # if simulate_detector_breakdown == True and split!='train':
        #     self.all_data.iloc[:, broken_detector_id:broken_detector_id+self.variables_per_segment-2] = -1e-6
        # no_detector_id = self.variables_per_segment*29
        # simulate_no_detector = True
        # if simulate_no_detector == True:
        #     self.all_data.iloc[:, no_detector_id:no_detector_id+self.variables_per_segment-2] = -1e-6
        
        # simulate_no_detector = True
        # no_detector_id_from = self.variables_per_segment*16
        # no_detector_id_to = self.variables_per_segment*(23+1)
        # if simulate_no_detector == True:
        #     self.all_data.iloc[:, no_detector_id_from:no_detector_id_to+1:self.variables_per_segment] = -1e-6
        #     self.all_data.iloc[:, no_detector_id_from+1:no_detector_id_to+1:self.variables_per_segment] = -1e-6

        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size

        # self.all_data_5min = self.all_data
        # self.all_data_5min[self.all_data_5min<0] = np.nan
        # self.all_data_5min = self.all_data_5min.fillna(method='pad')
        # self.all_data_5min = self.all_data_5min.rolling('5min').mean()
        # self.all_data_5min = self.all_data_5min.iloc[10:,:]
        # self.all_data_5min = self.all_data_5min.fillna(0.0)
        # self.all_data      = self.all_data_5min#self.all_data.iloc[10:,:]


        NEG = -1e-6
        self.all_data[:,::self.variables_per_segment] = self.mainlines_to_include_in_output * self.all_data[:,::self.variables_per_segment]
        self.all_data[:,1::self.variables_per_segment] = self.mainlines_to_include_in_output * self.all_data[:,1::self.variables_per_segment]
        self.all_data[self.all_data==0] = NEG

        ##print 
        print(self.all_data.columns)
        # all_flows = self.all_data.iloc[:,::self.variables_per_segment]
        # all_flows = all_flows[all_flows>0]
        # self.mean_flow = all_flows.mean().mean()
        # self.std_flow = all_flows.std().mean()
        
        all_densities = self.all_data.iloc[:,0::self.variables_per_segment]
        all_densities = all_densities[all_densities>0]
        self.mean_density = all_densities.mean().mean()
        self.std_density = all_densities.std().mean()

        all_speeds = self.all_data.iloc[:,1::self.variables_per_segment]
        all_speeds = all_speeds[all_speeds>0]
        self.mean_speed = all_speeds.mean().mean()
        self.std_speed = all_speeds.std().mean()

        all_onramps = self.all_data.iloc[:,2::self.variables_per_segment]
        all_onramps = all_onramps[all_onramps>0]
        self.mean_onramp = all_onramps.mean().mean()
        self.std_onramp = all_onramps.std().mean()

        all_offramps = self.all_data.iloc[:,3::self.variables_per_segment]
        all_offramps = all_offramps[all_offramps>0]
        self.mean_offramp = all_offramps.mean().mean()
        self.std_offramp = all_offramps.std().mean()

        # self.all_means = [self.mean_flow,self.mean_density,self.mean_speed,self.mean_onramp,self.mean_offramp]*self.num_segments 
        # self.all_stds = [self.std_flow,self.std_density,self.std_speed,self.std_onramp,self.std_offramp]*self.num_segments 

        # print("means:",self.mean_flow,self.mean_density,self.mean_speed,self.mean_onramp,self.mean_offramp)
        # print("stds:",self.std_flow,self.std_density,self.std_speed,self.std_onramp,self.std_offramp)

        self.all_means = [self.mean_density,self.mean_speed,self.mean_onramp,self.mean_offramp]*self.num_segments 
        self.all_stds = [self.std_density,self.std_speed,self.std_onramp,self.std_offramp]*self.num_segments 

        print("means:",self.mean_density,self.mean_speed,self.mean_onramp,self.mean_offramp)
        print("stds:",self.std_density,self.std_speed,self.std_onramp,self.std_offramp)

        # add context
        # all_dates = pd.to_datetime(self.all_data.index)
        # all_context = pd.DataFrame(all_dates.dayofweek.values/6.0,columns=['dow'])
        # all_context['tod'] = (all_dates.hour * 60. + all_dates.minute) / 1440.
        # all_context.index = self.all_data.index
        input_cols_to_include = np.arange(len(all_active_vars))[all_active_vars]
        # self.all_input_data = self.all_data/self.max_vals
        self.all_data = (self.all_data-self.all_means)/self.all_stds
        self.all_input_data = self.all_data#(self.all_data-self.all_means)/self.all_stds
        self.all_input_data = self.all_input_data.iloc[:,input_cols_to_include]
        # self.all_input_data = pd.concat([self.all_input_data,all_context],axis=1)
        # self.all_input_data = self.all_data
        
        self.shuffle = shuffle
    
    def get_means_stds(self):
        return (torch.Tensor(self.all_means),torch.Tensor(self.all_stds))
    
    def get_max_vals(self):
        return torch.Tensor(self.max_vals)
    
    def __getitem__(self, index):
        
        if not self.split=='test':
            rand = np.random.randint(self.output_seq_len, size=1)[0]
            idx = (index * (self.output_seq_len-1)) + rand
        else:
            idx = index

        input_len = self.input_seq_len #* 10
        label_len = self.output_seq_len

        NEG = -1e-6


        if self.scale_input:
            one_input = self.all_input_data.iloc[idx:idx+input_len, :].values

        # # one_input = self.all_data.iloc[idx:idx+input_len, :].values
        # one_input = self.all_data_5min.iloc[idx:idx+input_len*1:1, :].values

        # one_input[:,::self.variables_per_segment] = self.mainlines_to_include_in_input * one_input[:,::self.variables_per_segment]
        # one_input[:,1::self.variables_per_segment] = self.mainlines_to_include_in_input * one_input[:,1::self.variables_per_segment]
        # #one_input[one_input==0] = NEG
        # one_input[:,2::self.variables_per_segment] = one_input[:,2::self.variables_per_segment] #+ 1e-3
        # one_input[:,3::self.variables_per_segment] = one_input[:,3::self.variables_per_segment] #+ 1e-3
        
        # if self.scale_input:
        #   one_input = one_input/self.max_vals
        # #   one_input = one_input - 0.5
        
        one_label = self.all_data.iloc[idx+input_len:idx+input_len+label_len, :].values

        # one_label[:,::self.variables_per_segment] = self.mainlines_to_include_in_output * one_label[:,::self.variables_per_segment]
        # one_label[:,1::self.variables_per_segment] = self.mainlines_to_include_in_output * one_label[:,1::self.variables_per_segment]
        # # one_label[:,2::self.variables_per_segment] = self.mainlines_to_include_in_output * one_label[:,2::self.variables_per_segment]
        # one_label[one_label==0] = NEG
        # # one_label[:,3::self.variables_per_segment] = one_label[:,3::self.variables_per_segment] #+ 1e-3
        # # one_label[:,4::self.variables_per_segment] = one_label[:,4::self.variables_per_segment] #+ 1e-3
        
        # if self.scale_output:
        #   one_label = one_label/self.max_vals
        
        one_label = one_label.transpose(0,1)

        return {
            'id': index,
            'source': one_input.astype('float'),
            'target': one_label.astype('float'),
        }

    # def resample(self, x, factor):
    #     return F.interpolate(x.view(1, 1, -1), scale_factor=factor).squeeze()

    def __len__(self):
        if not self.split=='test':
            data_len = (len(self.all_data) - (1*self.output_seq_len+self.input_seq_len*1) - 1)//self.output_seq_len#self.output_seq_len#- self.output_seq_len# - 1 #- 4* self.output_seq_len# - 2 * self.output_seq_len - 1
        else:
            data_len = (len(self.all_data) - (1*self.output_seq_len+self.input_seq_len*1) - 1)
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
            last_inputs = [s['target'][:1]*0-1e-6 for s in samples]
            previous_output = [np.concatenate([l,s['target'][:-1]]) for s,l in zip(samples,last_inputs)]

        # # previous_output = [s['source'][-1:]+s['target'][:-1] for s in samples] # [samples[0]['target'][0]]
        # if self.scale_input and self.scale_output:
        #     last_inputs = [s['source'][-1:]*0-1e-6 for s in samples]
        #     previous_output = [np.concatenate([l,s['target'][:-1]]) for s,l in zip(samples,last_inputs)]
        #     # previous_output = [np.concatenate([s['source'][-1:],s['target'][:-1]]) for s in samples]
        # elif self.scale_input and not self.scale_output:
        #     last_inputs = [s['source'][-1:]*self.max_vals for s in samples]
        #     previous_output = [np.concatenate([l,s['target'][:-1]]) for s,l in zip(samples,last_inputs)]

        #previous_output = [np.insert(previous_output[x],0,samples[x]['source'][-1],axis=0) for x in range(len(samples))]

        if self.input_feeding:
            prev_output_tokens = torch.FloatTensor(previous_output)
        else:
            prev_output_tokens = None

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

        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return len(self.all_data.iloc[index:index+self.output_seq_len, :].values.reshape(-1))
    