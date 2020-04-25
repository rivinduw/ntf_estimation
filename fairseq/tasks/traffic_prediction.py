import os
import torch
from fairseq.tasks import FairseqTask, register_task

from fairseq.data import TrafficDataset

import pandas as pd
import matplotlib.pyplot as plt


try:
    import wandb
    from matplotlib.lines import Line2D
    import numpy as np
    # wandb.init("traffic_calibration")
except Exception as e:
    print(e)

# python train.py data --task traffic_prediction --arch lstm_traffic --criterion mse_loss --batch-size 16
# C:\Users\rwe180\Documents\python-scripts\pytorch\pyNTF\ 

#python3 train.py 'data' --task traffic_prediction --criterion mse_loss --arch NTF_traffic --batch-size 8 --optimizer adam --lr 4e-3  --max-tokens 42000 --clip-norm 5.0 --warmup-updates 10 --lr-scheduler inverse_sqrt --update-freq 8

@register_task('traffic_prediction')
class TrafficPredictionTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--segment_lengths_file', default='segment_lengths.txt', type=str,
                            help='file prefix for data')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # with open(args.segment_lengths_file) as f:
        #     segment_lengths = f.read().splitlines()
        # print('| segment_lengths_file had {} segments'.format(len(segment_lengths)))
        return TrafficPredictionTask(args)

    def __init__(self, args):
        super().__init__(args)
        self.valid_step_num = 0
        # self.segment_lengths = segment_lengths
        metadata = pd.read_csv('data/estimation_sites5.csv')
        self.segment_lengths = list(metadata.loc[metadata['type']=='q','distance']/1000.)
        self.num_lanes = list(metadata.loc[metadata['type']=='q','num_lanes'])
        self.num_segments = 4#len(self.num_lanes)#12
        self.num_lanes = self.num_lanes[:self.num_segments]
        self.segment_lengths = self.segment_lengths[:self.num_segments]

        self.active_onramps = [x>0 for x in list(metadata.loc[metadata['type']=='r','num_lanes'])]
        self.active_offramps = [x>0 for x in list(metadata.loc[metadata['type']=='s','num_lanes'])]
        self.active_onramps = self.active_onramps[:self.num_segments]
        self.active_offramps = self.active_offramps[:self.num_segments]
        
        self.output_seq_len = 10
        self.input_seq_len = 120
        
        self.variables_per_segment = 4
        self.total_input_variables = self.num_segments*self.variables_per_segment

        # self.max_vals = [10000,100,5000,5000]
        self.print_count = 0

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        data_file = self.args.data#os.path.join(self.args.data, '{}.csv'.format('valid_data_109'))#split))
        max_vals = [10000,100,5000,5000]
        self.datasets[split] = TrafficDataset(data_file, output_seq_len=self.output_seq_len, split=split, \
                        input_seq_len=self.input_seq_len, num_segments=self.num_segments,max_vals=max_vals,\
                        train_from = "2019-02-01 00:00:00",\
                        train_to = "2019-10-01 00:00:00",\
                        valid_from = "2019-01-01 00:00:00",\
                        valid_to = "2019-02-01 00:00:00",\
                        test_from = "2019-10-01 00:00:00",\
                        test_to = "2019-11-01 00:00:00",\
                        mainlines_to_include_in_input = None,\
                        mainlines_to_include_in_output = None,\
                        scale_input=True,\
                        scale_output=True)
        #if split=='train':
        self.max_vals = self.datasets[split].get_max_vals()

        print('| {} {} {} examples'.format(self.args.data, split, len(self.datasets[split])))
    
    def get_active_onramps(self):
        print("onramps",self.active_onramps)
        return self.active_onramps
    
    def get_active_offramps(self):
        print("offramps",self.active_offramps)
        return self.active_offramps
    
    def get_segment_lengths(self):
        return self.segment_lengths
    
    def get_num_lanes(self):
        return self.num_lanes
    
    def get_max_vals(self):
        return self.max_vals
    
    def get_num_segments(self):
        return self.num_segments
    
    def get_variables_per_segment(self):
        return self.variables_per_segment
    
    def get_total_input_variables(self):
        return self.total_input_variables

    def get_output_seq_len(self):
        return self.output_seq_len

    def get_input_seq_len(self):
        return self.input_seq_len 

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (1e20,1)#(self.args.max_positions*self.seq_len, 1)

    # @property
    # def source_dictionary(self):
    #     """Return the source :class:`~fairseq.data.Dictionary`."""
    #     return None#self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return None#self.label_vocab

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
            try:
                wandb.log({'valid_loss':loss})
                if self.valid_step_num%100 == 0:
                    net_output = model(**sample['net_input'])
                    print("****")
                #     # plt.ion()
                #     # plt.pause(0.1)
                #     # plt.close('all')
                #     # plt.pause(0.1)
                #     print(net_output[0].size())
                    
                    preds = net_output[0].view(-1,self.output_seq_len,self.total_input_variables).detach().cpu().numpy()#[0,:,0]#model.get_normalized_probs(net_output, log_probs=True).float()
                    src = sample['net_input']['src_tokens'].view(-1,self.output_seq_len,self.total_input_variables).detach().cpu().numpy()#[0,:,0]# model.get_targets(sample, net_output).float()
                    target = sample['target'].view(-1,self.output_seq_len,self.total_input_variables).detach().cpu().numpy()
                    for i in range(4):
                        pd.DataFrame(preds[i,:,:]).to_csv('preds_'+str(i)+'_.csv')
                        pd.DataFrame(src[i,:,:]).to_csv('src_'+str(i)+'_.csv')
                        pd.DataFrame(target[i,:,:]).to_csv('target_'+str(i)+'_.csv')
                        try:
                            ax = pd.DataFrame(10000*target[0,:,i*4]).plot()
                            pd.DataFrame(10000*preds[0,:,i*4]).plot(ax=ax)
                            pd.DataFrame(10000*src[0,:,i*4]).plot(ax=ax)
                            ax.legend(['target','pred','input'])
                            wandb.log({"mainline"+str(i+1): ax})
                            plt.close('all')
                        except Exception as e:
                            print(e)
                    try:
                        # for i in range(4):
                        #     ax = pd.DataFrame(5000*target[0,:,i*4+2]).fillna(0.0).plot()
                        #     pd.DataFrame(5000*preds[0,:,i*4+2]).fillna(0.0).plot(ax=ax)
                        #     pd.DataFrame(5000*src[0,:,i*4+2]).fillna(0.0).plot(ax=ax)
                        #     plt.show()
                        #     wandb.log({"onramp"+str(i+1): ax})
                        #     ax2 = pd.DataFrame(5000*target[0,:,i*4+3]).fillna(0.0).plot()
                        #     pd.DataFrame(5000*preds[0,:,i*4+3]).fillna(0.0).plot(ax=ax2)
                        #     pd.DataFrame(5000*src[0,:,i*4+3]).fillna(0.0).plot(ax=ax2)
                        #     plt.show()
                        #     wandb.log({"offramp"+str(i+1): ax2})

                        ax = pd.DataFrame(5000*target[0,:,7]).fillna(0.0).plot()
                        pd.DataFrame(5000*preds[0,:,7]).fillna(0.0).plot(ax=ax)
                        pd.DataFrame(5000*src[0,:,7]).fillna(0.0).plot(ax=ax)
                        ax.legend(['target','pred','input'])
                        wandb.log({"offramp": ax})
                        ax2 = pd.DataFrame(5000*target[0,:,14]).fillna(0.0).plot()
                        pd.DataFrame(5000*preds[0,:,14]).fillna(0.0).plot(ax=ax2)
                        pd.DataFrame(5000*src[0,:,14]).fillna(0.0).plot(ax=ax2)
                        ax.legend(['target','pred','input'])
                        wandb.log({"onramp": ax2})
                    except Exception as e:
                        print(e)
                        
                #         for seg in range(0,10):
                #             ax = pd.DataFrame(preds[i,:,seg*1]).plot()
                #             pd.DataFrame(target[i,:,seg*1]).plot(ax=ax)
                #             plt.title(str(i)+"***"+str(seg))
                #             plt.pause(0.1)
                #             plt.show(block=False)
                #             plt.pause(3.0)
                #             plt.pause(0.1)
                #             try:
                #                 wandb.log({"chart"+str(i)+"_"+str(seg): plt})
                #             except Exception as e:
                #                 print(e)
                #         plt.pause(2.0)
                #         plt.close('all')
                #     plt.pause(5.0)
                #     plt.close('all')
                    # wandb.save('checkpoints/checkpoint_best.pt')
                    # wandb.save('checkpoints/checkpoint_last.pt')
            except Exception as e:
                print(e)
        self.valid_step_num += 1
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        for n, p in model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                if(p.grad.abs().max()>1.0):
                    p.grad = torch.clamp(p.grad, min=-5., max=5.)
                    #p.grad = p.grad/p.grad.abs().max()

        self.print_count += 0
        if (self.print_count // 100) == 0:
          #plot_grad_flow(model.named_parameters())
          wandb.log({'train_loss':loss})
          for n, p in model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                wandb.log({n: wandb.Histogram(p.grad)})
        
        # wandb.log({'train_loss':loss})
        return loss, sample_size, logging_output

    # We could override this method if we wanted more control over how batches
    # are constructed, but it's not necessary for this tutorial since we can
    # reuse the batching provided by LanguagePairDataset.
    #
    # def get_batch_iterator(
    #     self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
    #     ignore_invalid_inputs=False, required_batch_size_multiple=1,
    #     seed=1, num_shards=1, shard_id=0,
    # ):
    #     (...)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    plt.figure(figsize=(10,10))

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            wandb.log({n: wandb.Histogram(p.grad)})
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=45)
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    # plt.ylim(bottom = -0.001, top=1.1)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    plt.tight_layout()
    wandb.log({"gradients": wandb.Image(plt)})