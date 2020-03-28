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

try:
    import wandb
    # wandb.init("traffic_calibration")
except Exception as e:
    print(e)

from .ntf_module import NTF_Module

@register_model('NTF_traffic')
class NTFModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        max_vals = task.get_max_vals()
        output_seq_len = task.get_output_seq_len()
        input_seq_len = task.get_input_seq_len()
        num_segments = task.get_num_segments()
        big_t = 10.0/3600. #hours
        segment_lengths = task.get_segment_lengths()
        num_lanes = task.get_num_lanes()

        num_encoder_layers = 1
        
        active_onramps = task.get_active_onramps()
        active_offramps = task.get_active_offramps()

        num_var_per_segment = task.get_variables_per_segment()
        total_input_variables = task.get_total_input_variables()
        device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        decoder_hidden_size = total_input_variables
        use_attention = True
        encoder = TrafficNTFEncoder(seq_len=input_seq_len, num_layers=num_encoder_layers, num_segments=num_segments, device=device)#.to(device)
        decoder = TrafficNTFDecoder(hidden_size=decoder_hidden_size, max_vals=max_vals, segment_lengths=segment_lengths, num_lanes=num_lanes, num_segments=num_segments, \
            seq_len = output_seq_len, encoder_output_units=total_input_variables, t_var=big_t, \
            active_onramps=active_onramps, active_offramps=active_offramps, attention=use_attention, \
            device=device)#.to(device)
        return cls(encoder, decoder)
    
    # def forward(self, src_tokens,  prev_output_tokens, **kwargs):#src_lengths,
    #     """
    #     Run the forward pass for an encoder-decoder model.

    #     First feed a batch of source tokens through the encoder. Then, feed the
    #     encoder output and previous decoder outputs (i.e., teacher forcing) to
    #     the decoder to produce the next outputs::

    #         encoder_out = self.encoder(src_tokens, src_lengths)
    #         return self.decoder(prev_output_tokens, encoder_out)

    #     Args:
    #         src_tokens (LongTensor): tokens in the source language of shape
    #             `(batch, src_len)`
    #         src_lengths (LongTensor): source sentence lengths of shape `(batch)`
    #         prev_output_tokens (LongTensor): previous decoder outputs of shape
    #             `(batch, tgt_len)`, for teacher forcing

    #     Returns:
    #         tuple:
    #             - the decoder's output of shape `(batch, tgt_len, vocab)`
    #             - a dictionary with any model-specific outputs
    #     """
    #     encoder_out = self.encoder(src_tokens, **kwargs) # src_lengths=src_lengths,
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        
    #     # negatives = self.sample_negatives(decoder_out)
    #     # y = decoder_out.unsqueeze(0)
    #     # targets = torch.cat([y, negatives], dim=0)
    #     print(decoder_out.size())
        
    #     return decoder_out


class TrafficNTFEncoder(FairseqEncoder):
    """NTF encoder."""
    def __init__(
        self, num_layers=2, #input_size=90,hidden_size=512
        seq_len=360, num_segments=12, num_var_per_segment=4, device=None,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False, padding_value=0):
        super().__init__(dictionary=None)
        
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        #self.hidden_size = hidden_size

        self.seq_len = seq_len
        self.num_segments = num_segments
        self.num_var_per_segment = num_var_per_segment

        self.input_size = num_segments * num_var_per_segment
        self.output_units = self.input_size

        # self.input_fc = Linear(self.input_size,self.hidden_size)

        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=self.input_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )

        # self.additional_fc = Linear(self.hidden_size, self.input_size)

        self.padding_value = padding_value

        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths=None):#def forward(self, input_x):#input_x,
        
        input_x = src_tokens

        # bsz, one_sample_length = input_x.size()
        bsz, ts, n_seg_var = input_x.size()
        
        one_timestep_size = self.num_segments*self.num_var_per_segment
        #print(input_x.size())
        assert one_timestep_size == n_seg_var

        x = input_x.view(-1,self.seq_len,one_timestep_size).float()
        
        # x = F.dropout(x, p=self.dropout_in, training=self.training)
        # print("x2",x.size())

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # x = self.input_fc(x)

        x_mask = x < 1e-6 #BUG
        # print("x_mask",x_mask.size())

        # apply LSTM
        if self.bidirectional:
            #state_size = 2 * self.num_layers, bsz, self.hidden_size
            state_size = 2 * self.num_layers, bsz, one_timestep_size
        else:
            # state_size = self.num_layers, bsz, self.hidden_size
            state_size = self.num_layers, bsz, one_timestep_size
        
        
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        lstm_outs, (final_hiddens, final_cells) = self.lstm(x, (h0, c0))

        # print("lstm_outs",lstm_outs.size())
        x = lstm_outs # self.additional_fc(lstm_outs)

        # unpack outputs and apply dropout
        # x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        # x = self.output_projection(lstm_outs)
        # x = F.dropout(x, p=self.dropout_out, training=self.training)
        # assert list(x.size()) == [seqlen, bsz, self.output_units]
        # assert list(x.size()) == [seqlen, bsz, segment_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = x_mask#.sum(dim=2)<1 #BUG: need multidim mask

        # print("x",x.size())t
        # print("final_hiddens",final_hiddens.size())
        # print("final_cells",final_cells.size())

        # from fairseq import pdb; pdb.set_trace()

        # return {
        #     'encoder_out': (x, final_hiddens, final_cells),
        #     'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        # }
        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e6)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)
        
        # compute attention
        #from fairseq import pdb; pdb.set_trace();
        #[360, 16, 90] * 1, [16, 90]
        attn_scores = (source_hids * x.unsqueeze(0))#.sum(dim=2)
        #[srclen, bsz]

        # attn_scores[attn_scores!=attn_scores] = float('-inf')

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float(-1e6)#float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        #attn_scores = attn_scores.float().masked_fill_(encoder_padding_mask,float('-inf'))
        #attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz
        attn_scores = nn.Softmax(dim=2)(attn_scores)

        #from fairseq import pdb; pdb.set_trace();
        # sum weighted sources
        x = (attn_scores * source_hids).sum(dim=0)
        #x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        # attn_scores [360, 16, 90]
        attn_scores = attn_scores.sum(dim=2)
        return x, attn_scores#.sum(dim=2)


class TrafficNTFDecoder(FairseqIncrementalDecoder):
    """Traffic NTF decoder."""
    def __init__(
        self, hidden_size=512, #input_size=90, output_size=90,
        num_segments=12, segment_lengths=None, num_lanes=None, t_var=None,
        num_var_per_segment=4, seq_len=360,
        active_onramps=None, active_offramps=None,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=None, pretrained_embed=None, device=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None, max_vals = None
    ):
        super().__init__(dictionary=None)
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=75, profile=None, sci_mode=False)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        # attention=False
        # self.need_attn = False#True 

        self.max_vals = torch.Tensor(max_vals).to(self.device)

        #attention=False
        self.num_segments = num_segments
        self.num_var_per_segment = num_var_per_segment

        self.input_size = self.num_segments * num_var_per_segment
        self.output_size = self.input_size
        self.seq_len = seq_len
        #self.hidden_size = hidden_size #self.input_size

        if encoder_output_units == None:
            encoder_output_units = self.input_size 

        self.adaptive_softmax = None

        self.encoder_output_units = encoder_output_units

        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        
        #num_layers=2
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=self.input_size if layer == 0 else hidden_size,#hidden_size + 
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        
        # if hidden_size != self.output_size:
        #     self.additional_fc = Linear(hidden_size, self.output_size)
        # if adaptive_softmax_cutoff is not None:
        #     # setting adaptive_softmax dropout to dropout_out for now but can be redefined
        #     self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
        #                                             dropout=dropout_out)
        # elif not self.share_input_output_embed:
            # self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)
        
        self.num_segments = num_segments#12#
        #self.input_dim = self.num_segments * 5
        #self.max_vals = #torch.Tensor([10000., 100., 1000., 1000] * self.num_segments).to(self.device) 
        
        self.num_common_params = 3+5 #num_boundry
        self.num_segment_specific_params = 8+2
        #  = 3+3


        if segment_lengths!=None:
            self.segment_lengths = torch.Tensor(segment_lengths)
            self.num_segment_specific_params -= 1
        else:
            self.segment_lengths = None
        
        if num_lanes!=None:
            self.num_lanes = torch.Tensor(num_lanes)
            self.num_segment_specific_params = 8
        else:
            self.num_lanes = None
        
        if t_var!=None:
            self.t_var = torch.Tensor([[t_var]])#torch.Tensor(t_var)
            self.num_common_params = 7
        else:
            self.t_var = None

        self.active_onramps = active_onramps
        self.active_offramps = active_offramps

        self.vmin = 10
        self.vmax = 110
        self.shortest_segment_length = 0.278
        self.num_ntf_steps = 3

        # self.num_model_params = 8+2#num_model_params
        self.total_model_params = self.num_segment_specific_params*self.num_segments

        self.ntf_proj = self.num_segment_specific_params*self.num_segments+self.num_common_params
        self.ntf_projection = nn.Linear(hidden_size, self.ntf_proj)
        # self.output_layer = nn.Linear(lin_layer_sizes[-1],
        #                           self.ntf_proj)

        if segment_lengths!=None and t_var!=None:
            print(self.segment_lengths,self.num_lanes)
            self.ntf_module = NTF_Module(num_segments=self.num_segments, cap_delta=self.segment_lengths, \
                lambda_var=self.num_lanes, t_var=self.t_var, \
                active_onramps=self.active_onramps, active_offramps=self.active_offramps, \
                device=self.device)
        else:
            print("no num lanes segment lengths")
            self.ntf_module = NTF_Module(num_segments=self.num_segments, \
                active_onramps=self.active_onramps, active_offramps=self.active_offramps, \
                device=self.device)

        ##OLD###
        # self.segment_fixed = torch.Tensor([[582.0/1000.,3.],[318.0/1000.,4.],[703.0/1000.,4.],[ 387.0/1000.,4.],[ 300.0/1000.,5.],[ 348.0/1000.,5.],[ 375.0/1000.,4.],[ 300.0/1000.,4.],[ 257.0/1000.,4.],[ 500.0/1000.,4.],[ 484.0/1000.,4.],[ 400.0/1000.,3.],[ 420.0/1000.,3.],[ 589.0/1000.,3.],[ 427./1000.,3.],[ 400.0/1000.,2.],[ 515.0/1000.,2.0],[ 495.0/1000.,3.0]]).to(self.device)#torch.Tensor(self.num_segments, 2)
        # #self.segment_fixed = torch.Tensor([[0.57237,3.],[0.92267,3.],[0.4238,3.],[0.8092,3.],[1.183,3.],[1.5899,3.],[0.2161,3.],[0.88367,3.],[0.59879,3.],[0.91263,3.],[0.672,3.],[1.7492,3.],[0.28183,3.],[0.85799,3.],[1.0847,3.],[1.1839,3.],[0.56727,3.],[0.5147,3.]]).to(self.device)
        # self.model_fixed = torch.Tensor([[10./3600.,17./3600.,23.,1.7,13.]]).to(self.device)

        # num_boundry = 3
        # num_model_params=8
        # self.ntf_vars = num_model_params*self.num_segments+num_boundry
        # self.ntf_projection = nn.Linear(self.output_size, self.ntf_vars)

        # self.ntf_module = NTF_Module(num_segments=self.num_segments,\
        #                              segment_fixed = self.segment_fixed,\
        #                              model_fixed = self.model_fixed
        #                             )

        self.print_count = 0
        ##END OLD

        #self.fc_out = Linear(self.output_size, self.output_size, dropout=dropout_out)


    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
    #def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        #print(prev_output_tokens.size(), encoder_out.size())
        #from fairseq import pdb; pdb.set_trace()
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            #print(prev_output_tokens.size())
            # prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_tokens = prev_output_tokens[:, -1:,:]
            
        # bsz, one_input_size = prev_output_tokens.size()
        # self.seq_len = 360
        # seqlen = self.seq_len
        
        bsz, seqlen, segment_units = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        # x = self.embed_tokens(prev_output_tokens)
        
        x = prev_output_tokens.view(-1,self.seq_len,self.input_size).float()
        #print(x.size())
        # x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            #input_feed = x.new_ones(bsz, self.input_size) * encoder_outs[-1,:bsz,:self.input_size]#[0.5,0.1,1.0,0.0,0.0]#0.5 
            input_feed = encoder_outs[-1,:bsz,:self.input_size]#[0.5,0.1,1.0,0.0,0.0]#0.5 
            
            input_feed = nn.functional.relu(input_feed)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)#x.new_zeros(segment_units, seqlen, bsz)  #x.new_zeros(srclen, seqlen, bsz)
        outs = []
        common_params_list = []
        segment_params_list = []
        
        for j in range(seqlen):
            #print(x.shape)
            # from fairseq import pdb; pdb.set_trace()
            # input feeding: concatenate context vector from previous time step
            input_d = F.dropout(x[j, :], p=0.5, training=self.training)
            input_mask = input_d > 1e-6#0.#-1e-6
            input_in = (x[j, :]*input_mask.float()) + ( (1-input_mask.float())*input_feed)
            #input = torch.clamp(input, min=-1.0, max=1.0)
            #import pdb; pdb.set_trace()
            self.print_count += 1
            # if self.print_count%1000==0:#random.random() > 0.9999:
            #     #from fairseq import pdb; pdb.set_trace()
            #     means = (input_in*(self.max_vals+1e-6)).view(-1,18,5).mean(dim=1).cpu().detach().numpy()
            #     print("\n\ninput means\t",means)
            #     wandb.log({"input0": wandb.Histogram(means[:,0])})
            #     wandb.log({"input1": wandb.Histogram(means[:,1])})
            #     wandb.log({"input2": wandb.Histogram(means[:,2])})
            #     wandb.log({"input3": wandb.Histogram(means[:,3])})
            #     #wandb.log({"input4": wandb.Histogram(means[4])})
            #     mean_x = x[j, :, :].view(-1,18,5).mean(dim=1)
            #     print("x[j, :, :] means\t",mean_x.cpu().detach().numpy())
            #     mean_feed = input_feed.view(-1,18,5).mean(dim=1)
            #     print("input_feed means\t",mean_feed.cpu().detach().numpy())
            
            # if random.random()>0.0:
            #     input = x[j, :, :]#torch.cat((x[j, :, :], input_feed), dim=1)
            # else:
            #     input = input_feed
            input = input_in
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                #input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                # out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
                out, attn_scores[:, j, :] = self.attention(cell, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            
            #out = self.additional_fc(out)
            
            # from fairseq import pdb; pdb.set_trace()
            ntf_input = self.ntf_projection(out)

            #NTF
            #ntf_input = x#self.out(output[0])
            common_params, segment_params = torch.split(ntf_input, [self.num_common_params, self.num_segment_specific_params*self.num_segments], dim=1)
            #assert self.num_common_params == 8
            #10./3600.,17./3600.,23.,1.7,13.
            common_params = torch.cat([torch.sigmoid(common_params[:, :1]), torch.sigmoid(common_params[:, 1:4]), torch.sigmoid(common_params[:, 4:])], dim=1).to(self.device)
            
            # if True:#self.segment_lengths!=None and self.t_var!=None:
            v0, q0, rhoNp1, tau, nu, delta, kappa = torch.unbind(torch.Tensor([self.vmax, 10000.0, 100.0, 0.01, 50.0, 5.0, 20.0]).to(self.device)*common_params, dim=1)
            v0 = v0 + self.vmin
            tau = 1./3600. + tau
            delta = 1.0 + delta
            kappa = 1.0 + kappa
            nu = 1.0 + nu
            
            segment_params = segment_params.view((-1, self.num_segment_specific_params, self.num_segments))
            segment_params = torch.cat([torch.sigmoid(segment_params[:,:4,:]),torch.sigmoid(segment_params[:,4:5,:]),torch.sigmoid(segment_params[:,5:6,:]),torch.tanh(segment_params[:,6:,:])],dim=1)
                                                                                                                                #  vf, a, rhocr, g, omegar, omegas, epsq, epsv 
            vf, a_var, rhocr, g_var, future_r, offramp_prop, epsq, epsv = torch.unbind(segment_params* torch.Tensor(\
                [[self.vmax],[2.0],[100.0],[10.0],[4000.0],[1.0],[100.0],[10.0]]).to(self.device),dim=1)#.to(self.device)
            rhocr = 1.0 + rhocr
            vf = self.vmin + vf
            a_var = 1.0 + a_var
            g_var = 1.0 + g_var

            x_input = input_in*self.max_vals
            model_steps = []
            
            for _ in range(self.num_ntf_steps):
                output1 = self.ntf_module(
                    x=x_input, v0=v0, q0=q0, rhoNp1=rhoNp1, vf=vf, a_var=a_var, rhocr=rhocr,\
                    g_var=g_var, future_r=future_r, offramp_prop=offramp_prop, epsq=epsq, epsv=epsv,\
                    tau=tau, nu=nu, delta=delta, kappa=kappa)
                model_steps.append(output1)
                x_input = output1
            # else:
            #     #                                                          v0,  q0,    ,rhoNp1, T, tau, nu, delta, kappa
            #     v0, q0, rhoNp1, t_var, tau, nu, delta, kappa = torch.unbind(torch.Tensor([110.0, 10000.0, 100.0, 0.01, 0.01, 50.0, 5.0, 20.0]).to(self.device)*common_params, dim=1)#.to(self.device)
            #     #from fairseq import pdb; pdb.set_trace()
            #     # v0 = torch.clamp(v0, min=5.0)
            #     # t_var = torch.clamp(t_var, min=0.001)
            #     v0 = v0 + self.vmin
            #     #v0 = torch.clamp(v0, max=120.0)
            #     t_var = torch.Tensor([10*0.00028]*bsz)# + t_var
            #     tau = 1./3600. + tau
            #     delta = 1.0 + delta
            #     kappa = 1.0 + kappa
            #     nu = 1.0 + nu

            #     # delta = 1.7 + delta
            #     # kappa = 13.0+ 0.0*kappa# + 1.0
            #     # nu = 23.0 + 0.0*nu #+ 1.0

            #     segment_params = segment_params.view((-1, self.num_segment_specific_params, self.num_segments))
            #     #assert self.num_segment_specific_params==10
            #     segment_params = torch.cat([torch.sigmoid(segment_params[:,:8,:]),F.relu(segment_params[:,8:10,:]),torch.tanh(segment_params[:,10:,:])],dim=1)
            #     # import pdb; pdb.set_trace()
            #                                                                                                                     #  self.Delta, self.lambda_var, vf, a, rhocr, g, omegar, omegas, epsq, epsv 
            #     cap_delta, lambda_var, vf, a_var, rhocr, g_var, future_r, offramp_prop, epsq, epsv =  torch.unbind(segment_params* torch.Tensor([[1.0],[10.0],[110.0],[5.0],[100.0],[10.0],[1000.0],[1.0],[1000.0],[10.0]]).to(self.device),dim=1)#.to(self.device)
            #     # cap_delta = torch.clamp(cap_delta, min=0.278)
            #     # vf = torch.clamp(vf, min=5.0)
            #     # lambda_var = torch.clamp(lambda_var, min=3.0)
                
            #     rhocr = 1.0 + rhocr
            #     cap_delta = cap_delta + self.shortest_segment_length
            #     vf = self.vmin + vf
            #     lambda_var = 1.0 + lambda_var
            #     a_var = 1.0 + a_var
            #     g_var = 1.0 + g_var

                
            #     # rhocr = 33.5 + 0.0*rhocr
            #     # cap_delta = cap_delta + 0.278
            #     # vf = 110. + 0.0*vf #+5.0
            #     # lambda_var = 3.0 + 0.0*lambda_var#+1.0
            #     # a_var = 1.8 + 0.0*a_var# + 0.5
                
            #     # print(pd.DataFrame(segment_params))
            #     # segment_params = segment_params.permute(0, 2, 1)

            #     #x = input_data*self.max_vals
            #     #x_unscaled = input_in*self.max_vals 
            #     x_input = input_in*self.max_vals
            #     model_steps = []
                
            #     for _ in range(self.num_ntf_steps):#x_input,segment_params,boundry_params)
            #         output1 = self.ntf_module(
            #             x=x_input, v0=v0, q0=q0, rhoNp1=rhoNp1, vf=vf, a_var=a_var, rhocr=rhocr,\
            #             g_var=g_var, future_r=future_r, offramp_prop=offramp_prop, epsq=epsq, epsv=epsv,t_var=t_var,\
            #             tau=tau, nu=nu, delta=delta, kappa=kappa, cap_delta=cap_delta,\
            #             lambda_var=lambda_var)
            #         model_steps.append(output1)
            #         x_input = output1

            output = torch.stack(model_steps,dim=0).mean(dim=0)
            #output = output.view(-1)
            output = output/(self.max_vals+1e-6)
            
            #output = torch.clamp(output, min=0, max=1)
            #x_output = output
            ##
            common_params_list.append(common_params)
            segment_params_list.append(segment_params)
            #return x, segment_params,common_params

            # #########OLD
            # boundry_params, segment_params = torch.split(ntf_input, [3,8*self.num_segments],dim=1)
            # segment_params = segment_params.view((-1,8,self.num_segments))
            # boundry_param_list.append(boundry_params)
            # segment_param_list.append(segment_params)
            # # boundry_params = torch.Tensor([200.0,10000.0,200.0]).to(self.device)*torch.sigmoid(boundry_params)
            # boundry_params = torch.Tensor([150.0,10000.0,100.0]).to(self.device)*torch.sigmoid(boundry_params)
            
            # segment_params = torch.cat([torch.sigmoid(segment_params[:,:4,:]),torch.tanh(segment_params[:,4:,:])],dim=1)
            #                                                 # vf, a, rhocr, g, omegar, omegas, epsq, epsv 
            # segment_params =  segment_params* torch.Tensor([[150.0],[2.0],[100.0],[5.0],[10.0],[10.0],[10.0],[10.0]]).to(self.device)
            # segment_params = segment_params.permute(0, 2, 1)
            # unscaled_input = input_in * self.max_vals

            # # print("boundry_params",boundry_params[0,::5].mean().item(),boundry_params.size())
            # # print("segment_params",segment_params[0,::5,0].mean().item(),segment_params.size())
            # # print(unscaled_input)

            # model_steps = []
            # num_steps = 3#18
            # for _ in range(num_steps):
            #     out1 = self.ntf_module(unscaled_input,segment_params,boundry_params)
            #     model_steps.append(out1)
            #     unscaled_input = out1

            # out = torch.stack(model_steps, dim=0).mean(dim=0)

            # avg_sum_max_vals = self.max_vals # summing everything above but speed and occ need to be avg
            # # avg_sum_max_vals[1::5] *= num_steps #mean occupancy
            # # avg_sum_max_vals[2::5] *= num_steps #mean speed
            # out  = out / (avg_sum_max_vals+1e-6) 

            # # print(out.mean().item())

            # # out = out / (self.max_vals+1e-6)
            # #from fairseq import pdb; pdb.set_trace()

            # #out = F.dropout(out, p=self.dropout_out, training=self.training)
            # # input feeding
            # input_feed = out#.view(-1,360,90)
            # #####END OLD

            input_feed = output#.view(-1,360,90)


            # save final output
            outs.append(output)
            
        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.stack(outs, dim=1)
        #print(x.size())

        self.all_common_params = torch.stack(common_params_list, dim=1)
        self.all_segment_params = torch.stack(segment_params_list, dim=1)

        # T x B x C -> B x T x C
        #x = x.transpose(1, 0)
        #print(x.size())

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        # if self.adaptive_softmax is None:
        #     if hasattr(self, 'additional_fc'):
        #         x = self.additional_fc(x)
        #         x = F.dropout(x, p=self.dropout_out, training=self.training)
        #     if self.share_input_output_embed:
        #         x = F.linear(x, self.embed_tokens.weight)
            # else:
            #     x = self.fc_out(x)
        # import fairseq.pdb as pdb; pdb.set_trace()#[:,-1,:]
        
        #x = x.contiguous().view(bsz,-1)#self.output_size)#self.fc_out(x)
        return x, attn_scores
    
    #my implementation
    def get_normalized_probs(self, net_output, log_probs=None, sample=None):
        #import pdb; pdb.set_trace()
        return net_output[0], self.all_common_params, self.all_segment_params
    
    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']
        

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e6)  # an arbitrary large number

    # def make_generation_fast_(self, need_attn=False, **kwargs):
    #     self.need_attn = need_attn

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

