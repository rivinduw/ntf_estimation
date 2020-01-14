# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion

import wandb

@register_criterion('mse_loss')
class MSECriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

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
        lprobs, boundry, segment = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = lprobs.float()
        
        boundry_loss = F.mse_loss(boundry[:,1:,:],boundry[:,:-1,:],reduction='mean')
        boundry_lambda = 10.0

        # boundry_mean = torch.mean(boundry,dim=1,keepdim=True) #[1,360,3]
        # boundry_loss = torch.mean((boundry-boundry_mean)**2,dim=1) #KL divergence later
        # boundry_loss = torch.mean(boundry_loss)
        
        segment_loss = F.mse_loss(segment[:,:,:4,1:],segment[:,:,:4,:-1],reduction='mean')
        segment_lambda = 1.0
        
        # segment_mean = torch.mean(segment,dim=2,keepdim=True) #[1,360,18,8]
        # segment_loss = torch.mean((segment-segment_mean)**2,dim=2)
        # segment_loss = torch.mean(segment_loss)

        segment_time_loss = F.mse_loss(segment[:,1:,:,:],segment[:,:-1,:,:],reduction='mean')
        # segment_time_lambda = 1.0#0.01

        # segment_time_mean = torch.mean(segment,dim=1,keepdim=True) #[1,360,18,8]
        # segment_time_loss = torch.mean((segment-segment_time_mean)**2,dim=1)
        # segment_time_loss = torch.mean(segment_time_loss)

        #smooth_loss = F.mse_loss(lprobs[:,1:,:],lprobs[:,:-1,:],reduction='sum')

        #from fairseq import pdb; pdb.set_trace();
        #lprobs = lprobs.float().view(-1)#, lprobs.size(-1))
        target = model.get_targets(sample, net_output).float()#.view(-1)#,360,85).float()
        #from fairseq import pdb; pdb.set_trace();
        #target = target.transpose(0, 1)
        #target = target#.view(-1)

        target_mask = target > 1e-6

        y = target[target_mask]#*target_mask.float()
        outputs = lprobs[target_mask]#*target_mask.float()
        #num_valid = target_mask.float().sum()

        mape_loss = torch.mean(torch.abs(torch.div(torch.sub(outputs,y),(y + 1e-6))))
        # mape_loss = mape_loss / num_valid
        accuracy = 1 - mape_loss
        accuracy = accuracy.clamp(0,1)
        

        
        # print("mask sum",target_mask.float().sum(),target.sum())
        loss = F.mse_loss(#1.0*torch.sqrt(
            #F.mse_loss(#F.l1_loss(# F.mse_loss(
            outputs, 
            y,
            reduction='mean'
            )#)
        loss = 10. * loss  #/ num_valid
        wandb.log(
            {'normal_loss':loss,
            'boundry_loss':boundry_loss,
            'segment_loss':segment_loss,
            'segment_time_loss':segment_time_loss,
            'mape_loss': mape_loss,
            'accuracy': accuracy,
            #'smooth_loss':smooth_loss
            }
            )

        total_loss = loss + segment_lambda*segment_loss  + boundry_lambda*boundry_loss # + segment_time_lambda*segment_time_loss + segment_lambda*smooth_loss
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
