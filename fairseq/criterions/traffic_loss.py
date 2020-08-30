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
        # return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))

@register_criterion('traffic_loss')
class MSECriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
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

        lprobs, common_params, segment, extra_params = model.get_normalized_probs(net_output, log_probs=False)
        # flow_res = extra_params['mean_flow_res']
        # first_input_feed = extra_params['first_input_feed']

        internal_params = {}
        internal_params['common_params'] = common_params
        internal_params['segment_params'] = segment

        target =  model.get_targets(sample, net_output).float()

        y_b = (target * self.all_stds) + self.all_means
        target_mask = y_b > 1e-6
        y = y_b[target_mask]
        outs = (lprobs * self.all_stds) + self.all_means
        outputs = outs[target_mask]

        num_valid = target_mask.float().sum()

        wmape = 100. * torch.div( torch.div(torch.sum(torch.abs(torch.sub(outputs,y))),torch.sum(torch.abs(y))),num_valid)
        mape_loss = torch.mean((torch.abs(torch.div(torch.sub(outputs,y),(y + 1e-6)))).clamp(0,1))
        # mape_loss = mape_loss / num_valid
        accuracy = 1. - mape_loss
        accuracy = accuracy.clamp(0,1)

        if num_valid>=1:
            target_loss = self.loss_fn(outputs, y)
        else:
            target_loss = 0.0

        total_loss = target_loss #+ 100*input_feed_consistancy_loss  #+ self.common_lambda*common_loss + self.segment_time_lambda*segment_time_loss + self.segment_lambda*segment_loss #+ volume_loss
        
        try:
            wandb.log(
                    {'normal_loss':total_loss,
                    'mape_loss': mape_loss,
                    'target_loss': target_loss,
                    'accuracy': accuracy,
                    'wmape': wmape,
                    'w-accuracy': 100. - wmape,
                    'target_v0' : wandb.Histogram(y_b[:,:,1].detach()),
                    'target_q4' : wandb.Histogram(y_b[:,:,2].detach()),
                    'output_v0' : wandb.Histogram(outs[:,:,1].detach()),
                    'output_q4' : wandb.Histogram(outs[:,:,2].detach()),
                    'target_q0' : wandb.Histogram(y_b[:,:,0].detach()),
                    'target_v4' : wandb.Histogram(y_b[:,:,3].detach()),
                    'output_q0' : wandb.Histogram(outs[:,:,0].detach()),
                    'output_v4' : wandb.Histogram(outs[:,:,3].detach()),
                    'target_s2' : wandb.Histogram(y_b[:,:,5].detach()),
                    'target_r4' : wandb.Histogram(y_b[:,:,4].detach()),
                    'output_s2' : wandb.Histogram(outs[:,:,5].detach()),
                    'output_r4' : wandb.Histogram(outs[:,:,4].detach()),
                    }
                )
        except Exception as e:
            print(e)

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
