from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

try:
    import wandb
except Exception as e:
    print(e)

class NTF_Module(nn.Module):
    def __init__(self, num_segments=4,\
                 t_var=None, tau=None, nu=None, delta=None, kappa=None,\
                 cap_delta=None, lambda_var=None,\
                 active_onramps=None, active_offramps=None, \
                 v0=None, q0=None, rhoNp1=None, vf=None, a_var=None, rhocr=None,\
                 device=None, print_every=100
                ):
        super(NTF_Module, self).__init__()
        
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.num_segments = num_segments
        
        if t_var is not None: self.t_var = t_var.view(-1,1)
        if tau is not None: self.tau = tau.view(-1,1)
        if nu is not None: self.nu = nu.view(-1,1)
        if delta is not None: self.delta = delta.view(-1,1)
        if kappa is not None: self.kappa = kappa.view(-1,1)
            
        if cap_delta is not None: self.cap_delta = cap_delta.view(-1,self.num_segments)
        if lambda_var is not None: self.lambda_var = lambda_var.view(-1,self.num_segments)
        
        if active_onramps != None:
            self.active_onramps = torch.Tensor(active_onramps)
        else:
            self.active_onramps = torch.zeros(self.num_segments)
        
        if active_offramps != None:
            self.active_offramps = torch.Tensor(active_offramps)
        else:
            self.active_offramps = torch.zeros(self.num_segments)
        
        if v0 is not None: self.v0 = v0.view(-1,1)
        if q0 is not None: self.q0 = q0.view(-1,1)
        if rhoNp1 is not None: self.rhoNp1 = rhoNp1.view(-1,1)
        
        if vf is not None: self.vf = vf.view(-1,1)
        if a_var is not None: self.a_var = a_var.view(-1,1)
        if rhocr is not None: self.rhocr = rhocr.view(-1,1)        
        
        #clamp params
        self.vmin = 1.
        self.vmax = 120.
        self.min_flow = 0.
        self.max_flow = 10000.

        self.TINY = 1e-6

        self.print_count = 0
        self.print_every = print_every


    def future_v(self):       
        self.stat_speed = self.vf * torch.exp(torch.div(-1,self.a_var+self.TINY)\
                                *torch.pow(torch.div(self.current_densities,self.rhocr+self.TINY)+self.TINY,self.a_var))
        self.stat_speed = torch.clamp(self.stat_speed, min=self.vmin, max=self.vmax)
        
        return self.current_velocities + (torch.div(self.t_var,self.tau+self.TINY)) * (self.stat_speed - self.current_velocities )  \
              + (torch.div(self.t_var,self.cap_delta) * self.current_velocities * (self.prev_velocities - self.current_velocities)) \
              - (torch.div(self.nu*self.t_var, (self.tau*self.cap_delta)) * torch.div( (self.next_densities - self.current_densities), (self.current_densities+self.kappa)) ) \
              - (torch.div( (self.delta*self.t_var) , (self.cap_delta * self.lambda_var+self.TINY) ) \
                 * torch.div( (self.current_onramps*self.current_velocities),(self.current_densities+self.kappa) ) )

    def future_rho(self):
        self.flow_residual = (self.prev_flows - self.current_flows + self.current_onramps - self.current_offramps)
        return self.current_densities + \
            torch.mul(torch.div(self.t_var,torch.mul(self.cap_delta,self.lambda_var)),self.flow_residual)

    def forward(self,current_flows=None, current_densities=None, current_velocities=None, \
                current_onramps=None, current_offramp_props=None, \
                v0=None, q0=None, rhoNp1=None, \
                vf=None, a_var=None, rhocr=None, \
                t_var=None, tau=None, nu=None, delta=None, kappa=None, \
                cap_delta=None, lambda_var=None):
        
        self.print_count+=1
        #print("current_flows",current_flows,"current_densities",current_densities,"current_velocities",current_velocities)
        if v0 is not None: self.v0 = v0.view(-1,1)
        if q0 is not None: self.q0 = q0.view(-1,1)
        if rhoNp1 is not None: self.rhoNp1 = rhoNp1.view(-1,1)
        
        if vf is not None: self.vf = vf.view(-1,1)
        if a_var is not None: self.a_var = a_var.view(-1,1)
        if rhocr is not None: self.rhocr = rhocr.view(-1,1)
        
        if t_var is not None: self.t_var = t_var.view(-1,1)
        if tau is not None: self.tau = tau.view(-1,1)
        if nu is not None: self.nu = nu.view(-1,1)
        if delta is not None: self.delta = delta.view(-1,1)
        if kappa is not None: self.kappa = kappa.view(-1,1)
            
        if cap_delta is not None: self.cap_delta = cap_delta.view(-1,self.num_segments)
        if lambda_var is not None: self.lambda_var = lambda_var.view(-1,self.num_segments)
            
        if current_flows is not None: 
            self.current_flows = current_flows.view(-1,self.num_segments)
        else:
            self.current_flows = current_velocities * (current_densities*self.lambda_var)
        
        if current_densities is not None: 
            self.current_densities = current_densities.view(-1,self.num_segments)
        else:
            self.current_densities = (current_flows / current_velocities)/self.lambda_var
        
        if current_velocities is not None: 
            self.current_velocities = current_velocities.view(-1,self.num_segments)
        else:
            self.current_velocities = current_flows / (current_densities*self.lambda_var)
        self.current_velocities = torch.clamp(self.current_velocities, min=self.vmin, max=self.vmax)
        
        self.v0 = torch.clamp(self.v0, min=self.vmin, max=self.vmax)  
        
        self.prev_velocities = torch.cat([self.v0,self.current_velocities[:,:-1]],dim=1)
        self.next_densities = torch.cat([self.current_densities[:,1:],self.rhoNp1],dim=1)
        self.prev_flows = torch.cat([self.q0,self.current_flows[:,:-1]],dim=1)
        
        
        if current_onramps is not None: 
            self.current_onramps = self.active_onramps.float() * current_onramps.view(-1,self.num_segments)
        else:
            self.current_onramps = torch.zeros(self.num_segments).view(-1,self.num_segments)
            
        if current_offramp_props is not None: 
            self.current_offramp_props = self.active_offramps.float() * current_offramp_props.view(-1,self.num_segments)
        else:
            self.current_offramp_props = torch.zeros(self.num_segments).view(-1,self.num_segments)
        self.current_offramps = self.current_offramp_props * self.prev_flows
        
        future_velocities = self.future_v()
        #print("future_velocities",future_velocities)
        future_velocities = torch.clamp(future_velocities, min=self.vmin, max=self.vmax)
        future_densities = self.future_rho()
        future_densities = torch.clamp(future_densities, min=self.min_flow, max=self.max_flow)
        future_flows = future_densities * future_velocities * self.lambda_var #- self.epsq
        future_flows = torch.clamp(future_flows, min=self.min_flow, max=self.max_flow)   

        try:
            #logging
            if self.print_count%self.print_every==0:
                wandb.log({"vf": wandb.Histogram(self.vf.cpu().detach().numpy())})
                wandb.log({"a_var": wandb.Histogram(self.a_var.cpu().detach().numpy())})
                wandb.log({"rhocr": wandb.Histogram(self.rhocr.cpu().detach().numpy())})
                wandb.log({"q0": wandb.Histogram(self.q0.cpu().detach().numpy())})
                wandb.log({"rhoNp1": wandb.Histogram(self.rhoNp1.cpu().detach().numpy())})
                wandb.log({"current_velocities": wandb.Histogram(self.current_velocities.cpu().detach().numpy())})
                wandb.log(
                    {'mean_current_velocities': self.current_velocities.cpu().detach().numpy().mean(),
                    'mean_current_densities': self.current_densities.cpu().detach().numpy().mean(),
                    'mean_current_flows': self.current_flows.cpu().detach().numpy().mean(),
                    'mean_current_onramps': self.current_onramps.cpu().detach().numpy().mean(),
                    'mean_current_offramps': self.current_offramps.cpu().detach().numpy().mean(),
                    'mean_v0': self.v0.cpu().detach().numpy().mean(),
                    'mean_q0': self.q0.cpu().detach().numpy().mean(),
                    'mean_rhoNp1': self.rhoNp1.cpu().detach().numpy().mean()
                    }
                )
                wandb.log({"current_densities": wandb.Histogram(self.current_densities.cpu().detach().numpy())})
                wandb.log({"current_flows": wandb.Histogram(self.current_flows.cpu().detach().numpy())})
                wandb.log({"current_onramps": wandb.Histogram(self.current_onramps.cpu().detach().numpy())})
                wandb.log({"current_offramps": wandb.Histogram(self.current_offramps.cpu().detach().numpy())})
                
                wandb.log({"current_r_4": wandb.Histogram(self.current_onramps[:, 3].cpu().detach().numpy())})
                wandb.log({"current_s_2": wandb.Histogram(self.current_offramps[:, 1].cpu().detach().numpy())})
                
                wandb.log({"current_flows_1": wandb.Histogram(self.current_flows[:, 0].cpu().detach().numpy())})
                wandb.log({"current_flows_2": wandb.Histogram(self.current_flows[:, 1].cpu().detach().numpy())})
                wandb.log({"current_flows_3": wandb.Histogram(self.current_flows[:, 2].cpu().detach().numpy())})
                wandb.log({"current_flows_4": wandb.Histogram(self.current_flows[:, 3].cpu().detach().numpy())})

                wandb.log({"stat_speed": wandb.Histogram(self.stat_speed.cpu().detach().numpy())})
                wandb.log({"v0": wandb.Histogram(self.v0.cpu().detach().numpy())})
                wandb.log({"q0": wandb.Histogram(self.q0.cpu().detach().numpy())})
                wandb.log({"rhoNp1": wandb.Histogram(self.rhoNp1.cpu().detach().numpy())})
                q_max = self.rhocr * self.vf * torch.exp(torch.div(-1,self.a_var+self.TINY))
                wandb.log({"q_max": wandb.Histogram(q_max.cpu().detach().numpy())})
                wandb.log({"t_var": wandb.Histogram(self.t_var.cpu().detach().numpy())})
                wandb.log({"tau": wandb.Histogram(self.tau.cpu().detach().numpy())})
                wandb.log({"nu": wandb.Histogram(self.nu.cpu().detach().numpy())})
                wandb.log({"delta": wandb.Histogram(self.delta.cpu().detach().numpy())})
                wandb.log({"kappa": wandb.Histogram(self.kappa.cpu().detach().numpy())})
                wandb.log({"cap_delta": wandb.Histogram(self.cap_delta.cpu().detach().numpy())})
                wandb.log({"lambda_var": wandb.Histogram(self.lambda_var.cpu().detach().numpy())})
                
                #future
                wandb.log({"future_velocities": wandb.Histogram(future_velocities.cpu().detach().numpy())})
                wandb.log({"future_densities": wandb.Histogram(future_densities.cpu().detach().numpy())})
                wandb.log({"future_flows": wandb.Histogram(future_flows.cpu().detach().numpy())})
                wandb.log(
                    {'mean_future_velocities': future_velocities.cpu().detach().numpy().mean(),
                    'mean_future_densities': future_densities.cpu().detach().numpy().mean(),
                    'mean_future_flows': future_flows.cpu().detach().numpy().mean()
                    }
                )
                wandb.log({"future_flows_1": wandb.Histogram(future_flows[:, 0].cpu().detach().numpy())})
                wandb.log({"future_flows_2": wandb.Histogram(future_flows[:, 1].cpu().detach().numpy())})
                wandb.log({"future_flows_3": wandb.Histogram(future_flows[:, 2].cpu().detach().numpy())})
                wandb.log({"future_flows_4": wandb.Histogram(future_flows[:, 3].cpu().detach().numpy())})

                wandb.log({"flow_residual": wandb.Histogram(self.flow_residual.cpu().detach().numpy())})
        except Exception as e:
            print(e)
        
        output_dict = {
            "future_flows": future_flows,
            "future_velocities": future_velocities,
            "future_densities": future_densities
        }

        return output_dict 
    