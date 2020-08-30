import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax

import random

import numpy as np

try:
    import wandb
except Exception as e:
    print(e)


from .ntf_module import NTF_Module
@register_model('NTF_traffic')
class NTFModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        max_vals = torch.Tensor(task.get_max_vals()).to(device)
        output_seq_len = task.get_output_seq_len()
        input_seq_len = task.get_input_seq_len()
        num_segments = task.get_num_segments()
        segment_lengths = task.get_segment_lengths()
        num_lanes = task.get_num_lanes()

        all_means, all_stds = task.get_means_stds()
        
        active_onramps = torch.Tensor(task.get_active_onramps()).to(device)
        active_offramps = torch.Tensor(task.get_active_offramps()).to(device)

        num_var_per_segment = task.get_variables_per_segment()
        total_input_variables = 7#task.get_total_input_variables()
        encoder_input_variables = 7#task.get_encoder_input_variables()
        
        encoder_hidden_size = 256#total_input_variables * 16 #// 2
        is_encoder_bidirectional = True
        decoder_hidden_size = 256#total_input_variables * 16 #// 2

        encoder = TrafficNTFEncoder(input_size=encoder_input_variables, seq_len=input_seq_len, \
            num_segments=num_segments, hidden_size=encoder_hidden_size, \
            bidirectional=is_encoder_bidirectional, dropout_in=0.5, dropout_out=0.5, device=device)

        decoder = TrafficNTFDecoder(input_size=total_input_variables, hidden_size=decoder_hidden_size, max_vals=max_vals,\
            all_means=all_means,all_stds=all_stds, segment_lengths=segment_lengths, num_lanes=num_lanes, num_segments=num_segments, \
            seq_len=output_seq_len, encoder_output_units=encoder_hidden_size, dropout_in=0.5, dropout_out=0.5, \
            num_var_per_segment=num_var_per_segment, active_onramps=active_onramps, active_offramps=active_offramps, device=device)
        return cls(encoder, decoder)


class TrafficNTFEncoder(FairseqEncoder):
    """NTF encoder."""
    def __init__(
        self, input_size=32, hidden_size=32, \
        seq_len=360, num_segments=12, num_layers=1,\
        dropout_in=0.1,dropout_out=0.1, bidirectional=False, padding_value=0, device=None):
        super().__init__(dictionary=None)
        # num_var_per_segment=4, \
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = input_size

        self.seq_len = seq_len
        
        if bidirectional:
            self.hidden_size = self.hidden_size // 2

        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bidirectional=bidirectional,
        )
        
        
    def forward(self, src_tokens, src_lengths=None):
        
        bsz, ts, _ = src_tokens.size() #input_size [32, 120, 16]

        x = src_tokens

        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * 1, bsz, self.hidden_size
        else:
            state_size = 1, bsz, self.hidden_size
        
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        lstm_outs, (final_hiddens, final_cells) = self.lstm(x, (h0, c0))
        
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(1, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(1, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)
        
        x = lstm_outs

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': None
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e6)  # an arbitrary large number



class TrafficNTFDecoder(FairseqIncrementalDecoder):
    """Traffic NTF decoder."""
    def __init__(
        self, hidden_size=512, input_size=90, output_size=90,
        num_segments=12, segment_lengths=None, num_lanes=None,
        num_var_per_segment=4, seq_len=360,
        active_onramps=None, active_offramps=None,
        num_layers=1, dropout_in=0.1, dropout_out=0.1,
        encoder_output_units=None, device=None, 
        share_input_output_embed=False, adaptive_softmax_cutoff=None, 
        max_vals = None, all_means=None,all_stds=None,
    ):
        super().__init__(dictionary=None)
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        

        self.num_segments = num_segments
        self.num_var_per_segment = num_var_per_segment

        self.input_size = input_size
        self.ntf_state_size = 13
        self.output_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.encoder_output_units = encoder_output_units

        self.extra_hidden_layer = False

        if self.encoder_output_units != self.ntf_state_size:
            if self.extra_hidden_layer:
                self.encoder_hidden_to_decoder_input_feed_hidden_layer = nn.Linear(self.encoder_output_units, self.encoder_output_units)
            self.encoder_hidden_to_input_feed_proj = nn.Linear(self.encoder_output_units, self.ntf_state_size)
        else:
            self.encoder_hidden_to_input_feed_proj = None

        if self.encoder_output_units != self.hidden_size:
            self.encoder_hidden_proj = nn.Linear(self.encoder_output_units, self.hidden_size)
            self.encoder_cell_proj = nn.Linear(self.encoder_output_units, self.hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        
        self.rnn = LSTMCell(input_size=self.input_size,hidden_size=hidden_size)
        
        self.num_segments = num_segments
        self.max_vals = max_vals

        self.all_means = [3000., 90., 3000., 90., 3000., 90., 3000., 90., 3000., 90., 7., 0.2, 300.]#[15., 90., 500., 800.]*self.num_segments 
        self.all_stds  = [2000., 10., 2000., 10., 2000., 10., 2000., 10., 2000., 10., 6., 0.2, 300.]#[15., 20., 300., 400.]*self.num_segments 
        self.all_means = torch.Tensor(np.array(self.all_means))
        self.all_stds = torch.Tensor(np.array(self.all_stds))

        self.input_means = torch.Tensor(np.array([3000., 90., 3000., 90., 7., 0.2, 300.]))
        self.input_stds  = torch.Tensor(np.array([2000., 10., 2000., 10., 6., 0.2, 300.]))

        self.print_count = 0
                
        self.segment_lengths = torch.Tensor(segment_lengths)
        self.num_lanes = torch.Tensor(num_lanes)

        #self.epsq = torch.Tensor([[0.0]])
        #self.epsv = torch.Tensor([[0.0]])
        self.tau = torch.Tensor([[20./3600.]])
        self.nu = torch.Tensor([[35.0]])
        self.delta = torch.Tensor([[13.0]])
        self.kappa = torch.Tensor([[1.4]])

        self.t_var = torch.Tensor([[10./3600.]])
    
        self.num_common_params = 8
        self.num_segment_specific_params = 0

        #self.active_onramps = active_onramps
        #self.active_offramps = active_offramps

        self.vmin = 1.
        self.vmax = 120.
        self.shortest_segment_length = 0.278
        self.num_ntf_steps = 3

        self.amin = 1.0
        self.amax = 2.0
        self.beta_min = 0.0
        self.beta_max = 1.0
        self.rhocr_min = 1.0
        self.rhocr_max = 100.0
        self.rhoNp1_min = 0.0
        self.rhoNp1_max = 100.0
        self.flow_max = 10000.0
        self.flow_min = 0.0
        self.ramp_max = 3000.0

        # q0_a, v0_a, rhoNp1_a, beta2_a, r4_a, vf_a, a_var_a, rhocr_a 
        self.common_param_multipliers = torch.Tensor(\
                [self.flow_max-self.flow_min, \
                self.vmax-self.vmin, \
                self.rhoNp1_max-self.rhoNp1_min, \
                self.beta_max-self.beta_min, \
                self.ramp_max-self.flow_min, \
                self.vmax-(self.vmin), \
                self.amax-self.amin, \
                self.rhocr_max-self.rhocr_min\
                ]).to(self.device)
        self.common_param_additions = torch.Tensor([\
                self.flow_min, \
                self.vmin, \
                self.rhoNp1_min,\
                self.beta_min, \
                self.flow_min, \
                self.vmin,\
                self.amin,\
                self.rhocr_min]).to(self.device)

        #self.segment_param_multipliers = torch.Tensor([[self.ramp_max],[self.ramp_max]]).to(self.device)
        #self.segment_param_additions = torch.Tensor([[self.flow_min],[self.flow_min]]).to(self.device)

        self.common_param_activation = nn.Sigmoid()
        #self.segment_param_activation = None
        self.input_feed_activation = None

        self.total_segment_specific_params = self.num_segment_specific_params*self.num_segments

        self.total_ntf_parameters = self.num_segment_specific_params*self.num_segments+self.num_common_params

        self.ntf_projection = nn.Linear(self.hidden_size, self.total_ntf_parameters)
        # self.ntf_projection = Custom_Linear(self.hidden_size, self.total_ntf_parameters, min_val=-5.0, max_val=5.0, bias=True)

        # self.ntf_module = NTF_Module(num_segments=self.num_segments, cap_delta=self.segment_lengths, \
        #         lambda_var=self.num_lanes, t_var=self.t_var, num_var_per_segment=self.num_var_per_segment, \
        #         active_onramps=self.active_onramps, active_offramps=self.active_offramps, \
        #         epsq=self.epsq,epsv=self.epsv, tau=self.tau, nu=self.nu, delta=self.delta, kappa=self.kappa,\
        #         device=self.device)

        self.traffic_model = NTF_Module(t_var=self.t_var,tau=self.tau, nu=self.nu, delta=self.delta, \
                                        kappa=self.kappa,cap_delta=self.segment_lengths, lambda_var=self.num_lanes)
        


    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        # encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']
        
        bsz, seqlen, _ = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]

        x = prev_output_tokens
        
        # B x T x C -> T x B x C 10,32,16
        x = x.transpose(0, 1)

        if self.encoder_hidden_proj != None:
            prev_hiddens = self.encoder_hidden_proj(encoder_hiddens[0,:,:])
            prev_cells = self.encoder_cell_proj(encoder_cells[0,:,:])
        else:
            prev_hiddens = encoder_hiddens[0,:,:]
            prev_cells = encoder_cells[0,:,:]
        
        if self.encoder_hidden_to_input_feed_proj != None:
            if self.input_feed_activation != None:
                input_feed = self.input_feed_activation(self.encoder_hidden_to_input_feed_proj(encoder_hiddens[0,:,:]))
            else:
                if self.extra_hidden_layer:
                    extra_hidden = torch.relu(self.encoder_hidden_to_decoder_input_feed_hidden_layer(encoder_hiddens[0,:,:]))
                else:
                    extra_hidden = encoder_hiddens[0,:,:]
                input_feed = self.encoder_hidden_to_input_feed_proj(extra_hidden)
        else:
            if self.input_feed_activation != None:
                input_feed = self.input_feed_activation(encoder_hiddens[0,:,:])
            else:
                input_feed = encoder_hiddens[0,:,:]

        self.first_input_feed = input_feed
        
        outs = []
        common_params_list = []
        segment_params_list = []

        flow_res_list = []
        
        for j in range(seqlen):

            input_x =  ((x[j, :,:]*self.input_stds)+self.input_means) 

            # T x (B x C)
            q0_i,v0_i,q2_i,v2_i,rho5_i,beta2_i,r4_i = torch.unbind(input_x, dim=1)

            unnormed_input_feed = (input_feed * self.all_stds) + self.all_means
            q1, v1, q2, v2, q3, v3, q4, v4, q0, v0, rho5, beta2, r4 = torch.unbind(unnormed_input_feed, dim=1)

            # rho4 = (q4_i/(v4_i*3.0)) * ((q4_i/(v4_i*3.0))>0).float() +  rho4*((q4_i/(v4_i*3.0))<=0).float() #if (q4_i/(v4_i*3.0))>0 else rho4
            # v4   = v4_i * ((v4_i>0).float()) + v4*((v4_i<=0).float())#if v4_i>0 else v4
            # r4   = r4_i * ((r4_i>0).float()) + r4*((r4_i<=0).float())#if r4_i>0 else r4
            # s2   = s2_i * ((s2_i>0).float()) + s2*((s2_i<=0).float())#if s2_i>0 else s2
            q2 = q2_i
            v2 = v2_i
            beta2 = beta2_i
            r4 = r4_i
            rho5 = rho5_i

            v0 = v0_i #* ((v0_i>0).float()) + v0*((v0_i<=0).float())
            q0 = q0_i #* ((q0_i>0).float()) + q0*((q0_i<=0).float())

            #real_size_input = torch.stack([q1, v1, q2, v2, q3, v3, q4, v4, rho5, beta2, r4],dim=1)

            current_velocities = torch.stack([v1, v2, v3, v4],dim=1)
            current_flows = torch.stack([q1, q2, q3, q4],dim=1)           
            zero_tensor = torch.Tensor(256*[0.])
            current_onramps = torch.stack([zero_tensor, zero_tensor, zero_tensor, r4],dim=1)   
            current_offramp_props = torch.stack([zero_tensor, beta2, zero_tensor, zero_tensor],dim=1)   #torch.Tensor([[0.,beta2,0.,0.]])
            

            hidden, cell = self.rnn(x[j, :,:], (prev_hiddens, prev_cells))
            
            prev_hiddens = hidden #for next loop
            prev_cells = cell

            ntf_params = self.ntf_projection(hidden)

            #NTF
            common_params, segment_params = torch.split(ntf_params, [self.num_common_params, self.total_segment_specific_params], dim=1)
            if self.common_param_activation != None:
                common_params = self.common_param_activation(common_params)
            common_params = (self.common_param_multipliers*common_params)+self.common_param_additions
            q0_f, v0_f, rhoNp1_f, beta2_f, r4_f, vf, a_var, rhocr = torch.unbind(common_params, dim=1)

            
            model_steps = []

            for _ in range(self.num_ntf_steps):
                traffic_model_output = self.traffic_model.forward(current_flows=current_flows, \
                                                current_velocities=current_velocities, \
                                                current_onramps=current_onramps, \
                                                current_offramp_props=current_offramp_props,\
                                                v0=v0, q0=q0, rhoNp1=rho5,\
                                                vf=vf, a_var=a_var, rhocr=rhocr)
                current_flows = traffic_model_output.get('future_flows')
                current_velocities = traffic_model_output.get('future_velocities')

                # one_ntf_output, flow_res = self.ntf_module(
                #     x=real_size_input, v0=v0, q0=q0, rhoNp1=rhoNp1, vf=vf, a_var=a_var, rhocr=rhocr,\
                #     g_var=g_var, future_r=future_r, future_s=future_s)
                # real_size_input = one_ntf_output
                # flow_res_list.append(flow_res)

                model_steps.append(traffic_model_output)
            
            # mean_ntf_output = real_size_input
            q1, q2, q3, q4 = torch.unbind(current_flows, dim=1)
            v1, v2, v3, v4 = torch.unbind(current_flows, dim=1)
            
            mean_ntf_output = torch.stack([q1, v1, q2, v2, q3, v3, q4, v4, q0_f, v0_f, rhoNp1_f, beta2_f, r4_f],dim=1)

            normed_output = (mean_ntf_output-self.all_means)/(self.all_stds)


            common_params_list.append(common_params)
            segment_params_list.append(segment_params)
            outs.append(normed_output)

            input_feed = normed_output
            
        # collect outputs across time steps
        # dim=1 to go from T x B x C -> B x T x C
        self.returned_out = torch.stack(outs, dim=1)
        self.all_common_params = torch.stack(common_params_list, dim=1)
        self.all_segment_params = torch.stack(segment_params_list, dim=1)

        q0_a, v0_a, rhoNp1_a, beta2_a, r4_a, vf_a, a_var_a, rhocr_a = torch.unbind(self.all_common_params, dim=2)
        q0_a = (q0_a-3000.)/2000.
        v0_a = (v0_a-90.)/10.
        rho5_a = (rhoNp1_a-7.)/6.
        beta2_a = (beta2_a-0.2)/0.2
        r4_a = (r4_a-300.)/300.
        q1_a, v1_a, q2_a, v2_a, q3_a, v3_a, q4_a, v4_a, q0_d, v0_d, rhoNp1_d, beta2_d, r4_d = torch.unbind(self.returned_out, dim=2)


        new_out = torch.stack([q0_a,v0_a,q2_a,v2_a,rho5_a,beta2_a,r4_a],dim=2)

        self.mean_flow_res = None#torch.stack(flow_res_list, dim=2).sum(axis=1).abs().mean(axis=1)

        # return returned_out, self.all_common_params, self.all_segment_params
        return new_out, self.all_common_params, self.all_segment_params
    
    #my implementation
    def get_normalized_probs(self, net_output, log_probs=None, sample=None):
        extra_params = {
            'first_input_feed':self.first_input_feed,
            'mean_flow_res': self.mean_flow_res,
            'all_states': self.returned_out,
        }
        return net_output[0], self.all_common_params, self.all_segment_params, extra_params
    
    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e6)  # an arbitrary large number


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

def Custom_Linear(in_features, out_features, min_val=0.0, max_val=1.0, bias=True):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(min_val, max_val)
    if bias:
        m.bias.data.uniform_(min_val, max_val)
    return m

def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


# @register_model_architecture('NTF', 'NTF')
@register_model_architecture('NTF_traffic', 'NTF_traffic')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')

