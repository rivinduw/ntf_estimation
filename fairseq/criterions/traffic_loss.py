# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F
import torch
import numpy as np

from fairseq import utils

from . import FairseqCriterion, register_criterion

try:
    import wandb
except Exception as e:
    print(e)


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

@register_criterion('traffic_loss')
class MSECriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        # wandb.init(job_type='mse_loss', config=args)
        # self.mse_loss = torch.nn.MSELoss()#F.mse_loss(reduction='mean')
        # self.loss_fn = torch.nn.L1Loss()
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = RMSLELoss()
        # self.loss_fn = torch.nn.SmoothL1Loss()
        #self.loss_fn = nn.KLDivLoss(reduction='batchmean')

        self.common_lambda = 1.0
        self.segment_lambda = 1.0
        self.segment_time_lambda = 1.0

        self.active_onramps = task.get_active_onramps()
        self.active_offramps = task.get_active_offramps()
        self.inactive_onramps = [not x for x in self.active_onramps]
        self.inactive_offramps = [not x for x in self.active_offramps]

        self.active_onramps = torch.tensor(self.active_onramps, dtype=torch.bool)
        self.active_offramps = torch.tensor(self.active_offramps, dtype=torch.bool)
        self.inactive_onramps = torch.tensor(self.inactive_onramps, dtype=torch.bool)
        self.inactive_offramps = torch.tensor(self.inactive_offramps, dtype=torch.bool)


        self.max_vals = task.get_max_vals()

        self.all_means, self.all_stds = task.get_means_stds()


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, internal_params = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'internal_params': internal_params,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # import fairseq.pdb as pdb; pdb.set_trace()
        lprobs, common_params, segment, extra_params = model.get_normalized_probs(net_output, log_probs=False)
        flow_res = extra_params['mean_flow_res']
        first_input_feed = extra_params['first_input_feed']

        input_feed_consistancy_loss = self.loss_fn(first_input_feed[:,4::4],first_input_feed[:,:-5:4]) + self.loss_fn(first_input_feed[:,4+1::4],first_input_feed[:,1:(-5+1):4])
        # input_feed_consistancy_loss = input_feed_consistancy_loss

        # lprobs = lprobs.float() #self.max_vals*
        # internal_params = {}
        # internal_params['common_params'] = common_params
        # internal_params['segment_params'] = segment

        #model.parameters()
        
        #bsz, ts, var
        # torch.Size([32, 10, 8])
        # v0, q0, rhoNp1, t_var, tau, nu, delta, kappa = torch.unbind(torch.Tensor([200.0, 10000.0, 100.0, 0.01, 0.01, 50.0, 5.0, 20.0]).to(self.device)*common_params, dim=1)      
        # common_loss = 0.0#self.loss_fn(common_params[:,1:,4], common_params[:,:-1,4]) + self.loss_fn(common_params[:,1:,6], common_params[:,:-1,6])
        # common_loss = self.loss_fn(common_params[:,1:,3:], common_params[:,:-1,3:])
        #NEv0, q0, rhoNp1, vf, a_var, rhocr, g_var
        

        # boundry_mean = torch.mean(boundry,dim=1,keepdim=True) #[1,360,3]
        # boundry_loss = torch.mean((boundry-boundry_mean)**2,dim=1) #KL divergence later
        # boundry_loss = torch.mean(boundry_loss)
        
        #bsz, t, var, segments
        # torch.Size([32, 10, 10, 12])
        #cap_delta, lambda_var, vf, a_var, rhocr, g_var, future_r, future_s, epsq, epsv =  torch.unbind(segment_params* torch.Tensor([[1.0],[10.0],[200.0],[5.0],[100.0],[10.0],[1000.0],[1000.0],[10.0],[10.0]]).to(self.device),dim=1)  
        #between segments
        #segment_loss = self.loss_fn(segment[:,:,:,1:],segment[:,:,:,:-1])
        # keep_ons_zero = self.loss_fn(segment[:,:,0,self.inactive_onramps]/5000.,0.0*segment[:,:,0,self.inactive_onramps])
        # keep_offs_zero = self.loss_fn(segment[:,:,1,self.inactive_offramps]/5000.,0.0*segment[:,:,1,self.inactive_offramps])
        # segment_loss = keep_ons_zero + keep_offs_zero
        
        
        # segment_mean = torch.mean(segment,dim=2,keepdim=True) #[1,360,18,8]
        # segment_loss = torch.mean((segment-segment_mean)**2,dim=2)
        # segment_loss = torch.mean(segment_loss)

        #import fairseq.pdb as pdb; pdb.set_trace()

        # segment_time_loss = self.loss_fn(segment[:,1:,:,:],segment[:,:-1,:,:])
        
        # segment_time_mean = torch.mean(segment,dim=1,keepdim=True) #[1,360,18,8]
        # segment_time_loss = torch.mean((segment-segment_time_mean)**2,dim=1)
        # segment_time_loss = torch.mean(segment_time_loss)

        #smooth_loss = F.mse_loss(lprobs[:,1:,:],lprobs[:,:-1,:],reduction='sum')

        #from fairseq import pdb; pdb.set_trace();
        #lprobs = lprobs.float().view(-1)#, lprobs.size(-1))
        target =  model.get_targets(sample, net_output).float()#.view(-1)#,360,85).float()
        #from fairseq import pdb; pdb.set_trace();
        #target = target.transpose(0, 1)
        #target = target#.view(-1)

        # target_mask = (self.max_vals * target) > 1e-6

        # volume_target = target[:,:,::4] * 10000
        # volume_outputs = lprobs[:,:,::4] *10000
        # volume_mask = (10000* volume_target) > 1e-6
        # vol_y = volume_target[volume_mask]
        # vol_outs =  volume_outputs[volume_mask]#                    'volume_loss': volume_loss,                    'volume_acc': vol_acc,
        # # wandb.log({"flows_actual": wandb.Histogram(vol_y.detach().numpy())})
        # # wandb.log({"flows_predictions": wandb.Histogram(vol_outs.detach().numpy())})
        # vol_mape = torch.mean((torch.abs(torch.div(torch.sub(vol_outs,vol_y),(vol_y + 1e-6)))).clamp(0,1))
        # vol_accuracy = 1. - vol_mape
        # vol_accuracy = vol_accuracy.clamp(0,1)

        # y = target * self.max_vals
        # target_mask = y > 1e-6
        # y = y[target_mask]
        # # y = target[target_mask]
        # outs = lprobs * self.max_vals
        # outputs = outs[target_mask]
        # # outputs = lprobs[target_mask]

        y_b = (target * self.all_stds) + self.all_means
        target_mask = y_b > 1e-6
        y = y_b[target_mask]
        outs = (lprobs * self.all_stds) + self.all_means
        outputs = outs[target_mask]

        # try:
        #   import pandas as pd
        #   print(pd.DataFrame(y_b).median().T)
        #   print(pd.DataFrame(outs).median().T)
        #   print(target_mask.float().sum())
        # except Exception as e:
        #   print(e)

        num_valid = target_mask.float().sum()

        wmape = 100. * torch.div( torch.div(torch.sum(torch.abs(torch.sub(outputs,y))),torch.sum(torch.abs(y))),num_valid)
        mape_loss = torch.mean((torch.abs(torch.div(torch.sub(outputs,y),(y + 1e-6)))).clamp(0,1))
        # mape_loss = mape_loss / num_valid
        accuracy = 1. - mape_loss
        accuracy = accuracy.clamp(0,1)

        # vol_acc = 1 - torch.mean((torch.abs(torch.div(torch.sub(vol_outs,vol_y),(vol_y + 1e-6)))).clamp(0,1))
        
        # print("mask sum",target_mask.float().sum(),target.sum())
        if num_valid>=1:
            target_loss = self.loss_fn(outputs, y)
            # self.loss_fn(target[target_mask], lprobs[target_mask])
            # l1 = torch.nn.L1Loss()
            #  self.loss_fn(target[target_mask]*1000,lprobs[target_mask]*1000)
            #  self.loss_fn(target*1,lprobs*1)
            # volume_loss = self.loss_fn(vol_outs, vol_y)
            #wandb.log({"all_actual": wandb.Histogram(y.detach().numpy())})
            #wandb.log({"all_predictions": wandb.Histogram(outputs.detach().numpy())})
        else:
            target_loss = 0.0
            # volume_loss = 0.0
        # + flow_res.mean()
        total_loss = target_loss #+ 100*input_feed_consistancy_loss  #+ self.common_lambda*common_loss + self.segment_time_lambda*segment_time_loss + self.segment_lambda*segment_loss #+ volume_loss
        
        try:
            wandb.log(
                    {'normal_loss':total_loss,
                    # 'vol_accuracy':vol_accuracy,
                    # 'common_loss':common_loss,
                    # 'segment_loss':segment_loss,
                    # 'segment_time_loss':segment_time_loss,
                    'mape_loss': mape_loss,
                    'target_loss': target_loss,
                    'accuracy': accuracy,
                    'wmape': wmape,
                    'w-accuracy': 100. - wmape,
                    'flow_res.mean': flow_res.mean(),
                    'input_feed_consistancy_loss' : input_feed_consistancy_loss,
                    'first_input_feed': wandb.Histogram(first_input_feed.detach()),
                    'target_v0' : wandb.Histogram(y_b[:,:,1].detach()),
                    'target_q4' : wandb.Histogram(y_b[:,:,2].detach()),
                    'output_v0' : wandb.Histogram(outs[:,:,1].detach()),
                    'output_q4' : wandb.Histogram(outs[:,:,2].detach()),
                    }
                )
        except Exception as e:
            print(e)

        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction='sum' if reduce else 'none',
        # )
        return total_loss, internal_params

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
