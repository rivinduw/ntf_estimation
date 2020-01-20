from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

try:
    import wandb
    # wandb.init("traffic_calibration")
except Exception as e:
    print(e)

class NTF_Module(nn.Module):
    def __init__(self, num_segments=18,\
                 t_var=None, tau=None, nu=None, delta=None, kappa=None,\
                 cap_delta=None, lambda_var=None,\
                 v0=None, q0=None, rhoNp1=None, vf=None, a_var=None, rhocr=None,\
                 g_var=None, future_r=None, offramp_prop=None, epsq=None, epsv=None, \
                 device=None, print_every=100
                ):
        super(NTF_Module, self).__init__()

        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device   

        if v0 is not None: self.v0 = v0
        if q0 is not None: self.q0 = q0
        if rhoNp1 is not None: self.rhoNp1 = rhoNp1
        if vf is not None: self.vf = vf
        if a_var is not None: self.a_var = a_var
        if rhocr is not None: self.rhocr = rhocr
        if g_var is not None: self.g_var = g_var
        if future_r is not None: self.future_r = future_r
        if offramp_prop is not None: self.offramp_prop = offramp_prop
        if epsq is not None: self.epsq = epsq
        if epsv is not None: self.epsv = epsv
        if t_var is not None: self.t_var = t_var
        if tau is not None: self.tau = tau
        if nu is not None: self.nu = nu
        if delta is not None: self.delta = delta
        if kappa is not None: self.kappa = kappa
        if cap_delta is not None: self.cap_delta = cap_delta
        if lambda_var is not None: self.lambda_var = lambda_var

        self.num_segments = num_segments
        self.vmin = 10
        self.vmax = 120

        self.inputs_per_segment = 4

        self.TINY = 1e-6

        self.q_index = 0
        self.rho_index = 1
        self.v_index = -1
        self.r_index = 2
        self.s_index = 3

        self.print_count = 0
        self.print_every = print_every


    def future_v(self):       
        self.stat_speed = self.vf * torch.exp(torch.div(-1,self.a_var+self.TINY)\
                                         *torch.pow(torch.div(self.current_densities,self.rhocr+self.TINY)+self.TINY,self.a_var))
        self.stat_speed = torch.clamp(self.stat_speed, min=self.vmin, max=self.vmax)
        try:
            if self.print_count%self.print_every==0:
                wandb.log({"vf": wandb.Histogram(self.vf.cpu().detach().numpy())})
                wandb.log({"a_var": wandb.Histogram(self.a_var.cpu().detach().numpy())})
                wandb.log({"rhocr": wandb.Histogram(self.rhocr.cpu().detach().numpy())})
                wandb.log({"g": wandb.Histogram(self.g_var.cpu().detach().numpy())})
                wandb.log({"q0": wandb.Histogram(self.q0.cpu().detach().numpy())})
                wandb.log({"rhoNp1": wandb.Histogram(self.rhoNp1.cpu().detach().numpy())})
                wandb.log({"current_velocities": wandb.Histogram(self.current_velocities.cpu().detach().numpy())})
                wandb.log({"current_densities": wandb.Histogram(self.current_densities.cpu().detach().numpy())})
                wandb.log({"current_flows": wandb.Histogram(self.current_flows.cpu().detach().numpy())})
                wandb.log({"stat_speed": wandb.Histogram(self.stat_speed.cpu().detach().numpy())})
                wandb.log({"v0": wandb.Histogram(self.v0.cpu().detach().numpy())})
                wandb.log({"q0": wandb.Histogram(self.q0.cpu().detach().numpy())})
                wandb.log({"rhoNp1": wandb.Histogram(self.rhoNp1.cpu().detach().numpy())})

                wandb.log({"t_var": wandb.Histogram(self.t_var.cpu().detach().numpy())})
                wandb.log({"tau": wandb.Histogram(self.tau.cpu().detach().numpy())})
                wandb.log({"nu": wandb.Histogram(self.nu.cpu().detach().numpy())})
                wandb.log({"delta": wandb.Histogram(self.delta.cpu().detach().numpy())})
                wandb.log({"kappa": wandb.Histogram(self.kappa.cpu().detach().numpy())})
                wandb.log({"cap_delta": wandb.Histogram(self.cap_delta.cpu().detach().numpy())})
                wandb.log({"lambda_var": wandb.Histogram(self.lambda_var.cpu().detach().numpy())})
                wandb.log({"epsv": wandb.Histogram(self.epsv.cpu().detach().numpy())})
        except Exception as e:
            print(e)
        
        return self.current_velocities + (torch.div(self.t_var,self.tau+self.TINY)) * (self.stat_speed - self.current_velocities )  \
              + (torch.div(self.t_var,self.cap_delta) * self.current_velocities * (self.prev_velocities - self.current_velocities)) \
              - (torch.div(self.nu*self.t_var, (self.tau*self.cap_delta)) * torch.div( (self.next_densities - self.current_densities), (self.current_densities+self.kappa)) ) \
              - (torch.div( (self.delta*self.t_var) , (self.cap_delta * self.lambda_var+self.TINY) ) * torch.div( (self.current_onramp*self.current_velocities),(self.current_densities+self.kappa) ) ) \
              + self.epsv

    def future_rho(self):
        return self.current_densities + \
            torch.mul(torch.div(self.t_var,torch.mul(self.cap_delta,self.lambda_var)),\
                      (self.prev_flows - self.current_flows + self.current_onramp - self.current_offramp))

    def forward(self,x=None, v0=None, q0=None, rhoNp1=None, \
                vf=None, a_var=None, rhocr=None, g_var=None, future_r=None, offramp_prop=None, epsq=None, epsv=None,\
                t_var=None, tau=None, nu=None, delta=None, kappa=None,\
                cap_delta=None, lambda_var=None):
        self.print_count+=1
        
        if v0 is not None: self.v0 = v0.view(-1,1)
        if q0 is not None: self.q0 = q0.view(-1,1)
        if rhoNp1 is not None: self.rhoNp1 = rhoNp1.view(-1,1)
        if vf is not None: self.vf = vf
        if a_var is not None: self.a_var = a_var
        if rhocr is not None: self.rhocr = rhocr
        if g_var is not None: self.g_var = g_var
        if future_r is not None: self.future_r = future_r
        if offramp_prop is not None: self.offramp_prop = offramp_prop
        if epsq is not None: self.epsq = epsq
        if epsv is not None: self.epsv = epsv
        if t_var is not None: self.t_var = t_var.view(-1,1)
        if tau is not None: self.tau = tau.view(-1,1)
        if nu is not None: self.nu = nu.view(-1,1)
        if delta is not None: self.delta = delta.view(-1,1)
        if kappa is not None: self.kappa = kappa.view(-1,1)
        if cap_delta is not None: self.cap_delta = cap_delta
        if lambda_var is not None: self.lambda_var = lambda_var
            
        x = x.view(-1, self.num_segments, self.inputs_per_segment)

        self.current_densities = x[:, :, self.rho_index] * self.g_var
        self.current_flows = x[:, :, self.q_index]
        self.current_onramp = x[:, :, self.r_index]
        self.current_offramp = x[:, :, self.s_index]

        active_onramps = self.current_onramp < 1e-6
        active_offramps = self.current_offramp < 1e-6
        
        self.current_velocities = self.current_flows / (self.current_densities*self.lambda_var+self.TINY)
        self.current_velocities = torch.clamp(self.current_velocities, min=self.vmin, max=self.vmax)
        self.v0 = torch.clamp(self.v0, min=self.vmin, max=self.vmax)

        self.prev_velocities = torch.cat([self.v0,self.current_velocities[:,:-1]],dim=1)
        self.next_densities = torch.cat([self.current_densities[:,1:],self.rhoNp1],dim=1)
        self.prev_flows = torch.cat([self.q0,self.current_flows[:,:-1]],dim=1)
        
        future_velocities = self.future_v()
        future_velocities = torch.clamp(future_velocities, min=self.vmin, max=self.vmax)
        future_densities = self.future_rho()
        future_occupancies = (future_densities) / (self.g_var+self.TINY)
#         future_occupancies = (future_densities / self.lambda_var) / (self.g_var+1e-6)

        future_flows = future_densities * future_velocities * self.lambda_var + self.epsq

        future_s =  active_offramps.float() * (self.offramp_prop*self.current_flows)
        future_r = active_onramps.float() * future_r

        try:
            if self.print_count%self.print_every==0:
                wandb.log({"future_velocities": wandb.Histogram(future_velocities.cpu().detach().numpy())})
                wandb.log({"future_densities": wandb.Histogram(future_densities.cpu().detach().numpy())})
                wandb.log({"future_occupancies": wandb.Histogram(future_occupancies.cpu().detach().numpy())})
                wandb.log({"future_flows": wandb.Histogram(future_flows.cpu().detach().numpy())})
                wandb.log({"future_r": wandb.Histogram(future_r.cpu().detach().numpy())})
                wandb.log({"future_s": wandb.Histogram(future_s.cpu().detach().numpy())})
                wandb.log({"epsq": wandb.Histogram(self.epsq.cpu().detach().numpy())})
        except Exception as e:
            print(e)

        #future_velocities = torch.clamp(future_velocities, min=10, max=120)
        #future_densities = torch.clamp(future_densities, min=0, max=1000)
        future_occupancies = torch.clamp(future_occupancies, min=0, max=100)
        future_flows = torch.clamp(future_flows, min=1, max=10000)
        future_r = torch.clamp(future_r, min=0, max=10000)
        future_s = torch.clamp(future_s, min=0, max=10000)

        one_stack =  torch.stack((future_flows,future_occupancies,future_r,future_s),dim=2)

        return one_stack.view(-1,self.num_segments*self.inputs_per_segment)
