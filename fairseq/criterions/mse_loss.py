# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion

try:
    import wandb
except Exception as e:
    print(e)

@register_criterion('traffic_loss')
class MSECriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        # wandb.init(job_type='mse_loss', config=args)
        # self.mse_loss = torch.nn.MSELoss()#F.mse_loss(reduction='mean')
        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn = torch.nn.MSELoss()

        self.common_lambda = 1.0
        self.segment_lambda = 1.0
        self.segment_time_lambda = 1.0

        self.max_vals = task.get_max_vals()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # import fairseq.pdb as pdb; pdb.set_trace()
        lprobs, common_params, segment = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = lprobs.float() #self.max_vals*
        
        #bsz, ts, var
        # torch.Size([32, 10, 8])
        # v0, q0, rhoNp1, t_var, tau, nu, delta, kappa = torch.unbind(torch.Tensor([200.0, 10000.0, 100.0, 0.01, 0.01, 50.0, 5.0, 20.0]).to(self.device)*common_params, dim=1)      
        common_loss = self.loss_fn(common_params[:,1:,:], common_params[:,:-1,:])
        

        # boundry_mean = torch.mean(boundry,dim=1,keepdim=True) #[1,360,3]
        # boundry_loss = torch.mean((boundry-boundry_mean)**2,dim=1) #KL divergence later
        # boundry_loss = torch.mean(boundry_loss)
        
        #bsz, t, var, segments
        # torch.Size([32, 10, 10, 12])
        #cap_delta, lambda_var, vf, a_var, rhocr, g_var, future_r, future_s, epsq, epsv =  torch.unbind(segment_params* torch.Tensor([[1.0],[10.0],[200.0],[5.0],[100.0],[10.0],[1000.0],[1000.0],[10.0],[10.0]]).to(self.device),dim=1)  
        segment_loss = self.loss_fn(segment[:,:,:3,1:],segment[:,:,:3,:-1])
        
        
        # segment_mean = torch.mean(segment,dim=2,keepdim=True) #[1,360,18,8]
        # segment_loss = torch.mean((segment-segment_mean)**2,dim=2)
        # segment_loss = torch.mean(segment_loss)

        #import fairseq.pdb as pdb; pdb.set_trace()

        segment_time_loss = self.loss_fn(segment[:,1:,:6,:],segment[:,:-1,:6,:])
        
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

        target_mask = (self.max_vals * target) > 1e-6

        y = target[target_mask]
        outputs = lprobs[target_mask]
        num_valid = target_mask.float().sum()

        wmape = 100. * torch.div( torch.div(torch.sum(torch.abs(torch.sub(outputs,y))),torch.sum(torch.abs(y))),num_valid)
        mape_loss = torch.mean((torch.abs(torch.div(torch.sub(outputs,y),(y + 1e-6)))).clamp(0,1))
        # mape_loss = mape_loss / num_valid
        accuracy = 1. - mape_loss
        accuracy = accuracy.clamp(0,1)
        
        # print("mask sum",target_mask.float().sum(),target.sum())
        target_loss = self.loss_fn(outputs, y)
        
        total_loss = target_loss + self.common_lambda*common_loss + self.segment_time_lambda*segment_time_loss + self.segment_lambda*segment_loss
        
        try:
            wandb.log(
                    {'normal_loss':total_loss,
                    'common_loss':common_loss,
                    'segment_loss':segment_loss,
                    'segment_time_loss':segment_time_loss,
                    'mape_loss': mape_loss,
                    'target_loss': target_loss,
                    'accuracy': accuracy,
                    'wmape': wmape,
                    'w-accuracy': 100. - wmape,
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
        return total_loss, total_loss

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
