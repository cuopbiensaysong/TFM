import torch
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
import pytorch_lightning as pl
from torch import optim
import torch.functional as F
import wandb

from utils.visualize import *
from utils.metric_calc import *
from utils.sde import SDE
from model.components.positional_encoding import *
from model.components.mlp import * 
from model.components.sde_func_solver import *
from model.components.grad_util import *
from utils.loss import (
    gaussian_nll_loss, mdn_loss, quantile_loss, l1_variance_loss,
)

class MLP_conditional_memory(torch.nn.Module):
    """ Conditional with many available classes

    return the class as is
    """
    def __init__(self, 
                 dim, 
                 treatment_cond,
                 memory, # how many time steps
                 out_dim=None, 
                 w=64, 
                 time_varying=False, 
                 conditional=False,  
                 time_dim = NUM_FREQS * 2,
                 clip = None,
                 distributional = None,
                 n_components = 3,
                 n_quantiles = 3,
                 ):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = dim 
        self.out_dim += 1 # for the time dimension
        self.treatment_cond = treatment_cond
        self.memory = memory
        self.dim = dim
        self.distributional = distributional
        self.n_components = n_components
        self.n_quantiles = n_quantiles
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0) + (dim * memory)

        if distributional == "gaussian_nll":
            net_out = dim * 2 + 1
        elif distributional == "mdn":
            net_out = n_components + n_components * dim * 2 + 1
        elif distributional == "quantile":
            net_out = n_quantiles * dim + 1
        elif distributional == "l1_variance":
            net_out = self.out_dim
        else:
            net_out = self.out_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.indim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, net_out),
        )
        self.default_class = 0
        self.clip = clip

        if distributional == "gaussian_nll":
            with torch.no_grad():
                self.net[-1].bias[dim:2*dim].fill_(-2.0)

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor)    

    def _encode_input(self, x):
        time_tensor = x[:, -1]
        encoded_time_span = self.encoding_function(time_tensor).reshape(-1, NUM_FREQS * 2)
        return torch.cat([x[:, :-1], encoded_time_span], dim=1)

    def forward_train(self, x):
        """forward pass
        Assume first two dimensions are x, c, then t
        """
        new_x = self._encode_input(x)
        result = self.net(new_x)

        if self.distributional == "gaussian_nll":
            mu = result[:, :self.dim]
            log_var = torch.clamp(result[:, self.dim:2*self.dim], min=-6.0, max=4.0)
            time_remaining = result[:, -1:]
            return mu, log_var, x[:, self.dim:-1], time_remaining
        elif self.distributional == "mdn":
            K, D = self.n_components, self.dim
            pi = torch.log_softmax(result[:, :K], dim=1)
            mu = result[:, K:K + K*D].reshape(-1, K, D)
            sigma = torch.nn.functional.softplus(result[:, K + K*D:K + 2*K*D]).reshape(-1, K, D) + 1e-4
            time_remaining = result[:, -1:]
            return pi, mu, sigma, x[:, self.dim:-1], time_remaining
        elif self.distributional == "quantile":
            Q, D = self.n_quantiles, self.dim
            quantile_preds = result[:, :Q*D].reshape(-1, Q, D)
            time_remaining = result[:, -1:]
            return quantile_preds, x[:, self.dim:-1], time_remaining
        else:
            return torch.cat([result[:,:-1], x[:,self.dim:-1], result[:,-1].unsqueeze(1)], dim=1)

    def forward(self, x):
        """ ODE-compatible forward: returns velocity field.
        Handles both point-estimate and distributional modes.
        """
        new_x = self._encode_input(x)
        result = self.net(new_x)
        x_coord = x[:, :self.dim]

        if self.distributional == "gaussian_nll":
            mu = result[:, :self.dim]
            log_var = torch.clamp(result[:, self.dim:2*self.dim], min=-6.0, max=4.0)
            time_remaining = result[:, -1:]
            std = torch.exp(0.5 * log_var)
            x1_coord = mu + std * torch.randn_like(std)
        elif self.distributional == "mdn":
            K, D = self.n_components, self.dim
            pi_logits = result[:, :K]
            mu = result[:, K:K + K*D].reshape(-1, K, D)
            component_idx = torch.multinomial(torch.softmax(pi_logits, dim=1), 1).squeeze(1)
            batch_idx = torch.arange(mu.shape[0], device=mu.device)
            x1_coord = mu[batch_idx, component_idx]
            sigma = torch.nn.functional.softplus(result[:, K + K*D:K + 2*K*D]).reshape(-1, K, D) + 1e-4
            x1_coord = x1_coord + sigma[batch_idx, component_idx] * torch.randn_like(x1_coord)
            time_remaining = result[:, -1:]
        elif self.distributional == "quantile":
            Q, D = self.n_quantiles, self.dim
            quantile_preds = result[:, :Q*D].reshape(-1, Q, D)
            x1_coord = quantile_preds[:, Q // 2, :]
            time_remaining = result[:, -1:]
        else:
            x1_coord = result[:, :self.dim]
            time_remaining = result[:, -1:]

        if self.clip is None:
            vt = (x1_coord - x_coord) / (time_remaining + 1e-8)
        else:
            vt = (x1_coord - x_coord) / torch.clip(time_remaining, min=self.clip)

        final_vt = torch.cat([vt, torch.zeros_like(x[:, self.dim:-1])], dim=1)
        return final_vt


class MLP_conditional_memory_sde_noise(torch.nn.Module):
    """ Conditional with many available classes

    return the class as is
    """
    def __init__(self, 
                 dim, 
                 treatment_cond,
                 memory, # how many time steps
                 out_dim=None, 
                 w=64, 
                 time_varying=False, 
                 conditional=False,  
                 time_dim = NUM_FREQS * 2,
                 clip = None,
                 ):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = 1 # for noise 
        self.treatment_cond = treatment_cond
        self.memory = memory
        self.dim = dim
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0) + (dim * memory)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.indim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w,self.out_dim),
        )
        self.default_class = 0
        self.clip = clip
        # self.encoding_function = positional_encoding_tensor()

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor)    

    def forward_train(self, x):
        """forward pass
        Assume first two dimensions are x, c, then t
        """
        time_tensor = x[:,-1]
        encoded_time_span = self.encoding_function(time_tensor).reshape(-1, NUM_FREQS * 2)
        new_x = torch.cat([x[:,:-1], encoded_time_span], dim=1)
        result = self.net(new_x)
        return result
    
    def forward(self,x):
        result = self.forward_train(x)
        return torch.cat([result, torch.zeros_like(x[:,1:-1])], dim=1)

""" Lightning module """
def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)

def l1_loss(pred, true):
    return torch.mean(torch.abs(pred - true))

def resolve_loss_fn(loss_fn):
    """Allow selecting a loss by config string or callable."""
    if callable(loss_fn):
        return loss_fn
    if isinstance(loss_fn, str):
        registry = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
        }
        if loss_fn not in registry:
            raise ValueError(
                f"Unsupported loss_fn '{loss_fn}'. Choose from: {list(registry.keys())}"
            )
        return registry[loss_fn]
    raise TypeError(f"loss_fn must be callable or str, got {type(loss_fn)}")

class Noise_MLP_Cond_Memory_Module(pl.LightningModule):
    def __init__(self, 
                 treatment_cond,
                 memory=3, # can increase / tune to see effect
                 dim=2, 
                 w=64, 
                 time_varying=True, 
                 conditional=True,
                 lr=1e-6,
                 sigma = 0.1, 
                 loss_fn = mse_loss,
                 metrics = ['mse_loss', 'l1_loss'],
                 implementation = "ODE", # can be SDE
                 sde_noise = 0.1,
                 clip = None,
                 naming = None,
                 ode_t_span_points: int = 10,
                 distributional = None,
                 n_components = 3,
                 n_quantiles = 3,
                 quantile_levels = (0.1, 0.5, 0.9),
                 lambda_var = 0.1,
                 ):
        if ode_t_span_points < 2:
            raise ValueError("ode_t_span_points must be >= 2 (torch.linspace endpoints).")
        super().__init__()
        self.flow_model = MLP_conditional_memory(dim=dim, 
                                              w=w, 
                                              time_varying=time_varying, 
                                              conditional=conditional, 
                                              treatment_cond=treatment_cond,
                                              memory=memory,
                                              clip=clip,
                                              distributional=distributional,
                                              n_components=n_components,
                                              n_quantiles=n_quantiles)
        if implementation == "SDE":
            self.noise_model = MLP_conditional_memory_sde_noise(dim=dim,
                                                    w=w,
                                                    time_varying=time_varying,
                                                    conditional=conditional,
                                                    treatment_cond=treatment_cond,
                                                    memory=memory,
                                                    clip=clip)
        else:
            self.noise_model = MLP_conditional_memory(dim=dim,
                                                        w=w,
                                                        time_varying=time_varying,
                                                        conditional=conditional,
                                                        treatment_cond=treatment_cond,
                                                        memory=memory,
                                                        clip=clip)
        self.automatic_optimization = False 
        self.loss_fn = resolve_loss_fn(loss_fn)
        self.save_hyperparameters()
        self.dim = dim
        self.w = w
        self.time_varying = time_varying
        self.conditional = conditional
        self.treatment_cond = treatment_cond
        self.lr = lr
        self.sigma = sigma
        self.naming = "Noise_MLP_Cond_memory_Module_"+implementation if naming is None else naming
        self.metrics = metrics
        self.implementation = implementation
        self.memory = memory
        self.sde_noise = sde_noise
        self.clip = clip
        self.ode_t_span_points = int(ode_t_span_points)
        self.distributional = distributional
        self.n_components = n_components
        self.n_quantiles = n_quantiles
        self.quantile_levels = tuple(quantile_levels) if quantile_levels else (0.1, 0.5, 0.9)
        self.lambda_var = lambda_var

        if self.memory > 1:
            self.naming += "_Memory_"+str(self.memory)
        if self.distributional:
            self.naming += "_" + self.distributional
            
    def __convert_tensor__(self, tensor):
        return tensor.to(torch.float32)

    def __x_processing__(self, x0, x1, t0, t1):
        # squeeze xs (prevent mismatch)
        x0 = x0.squeeze(0)
        x1 = x1.squeeze(0)
        t0 = t0.squeeze()
        t1 = t1.squeeze()

        t = torch.rand(x0.shape[0],1).to(x0.device)
        mu_t = x0 * (1 - t) + x1 * t
        data_t_diff = (t1 - t0).unsqueeze(1)
        x = mu_t + self.sigma * torch.randn(x0.shape[0], self.dim).to(x0.device)

        ut = (x1 - x0) / (data_t_diff + 1e-4)
        t_model = t * data_t_diff + t0.unsqueeze(1)
        futuretime = t1 - t_model

        return x, ut, t_model, futuretime, t
    
    def training_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (list of output): x0_values, x0_classes, x1_values, times_x0, times_x1
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        flow_opt, noise_opt = self.optimizers()

        x0, x0_class, x1, x0_time, x1_time = batch
        x0, x0_class, x1, x0_time, x1_time = self.__convert_tensor__(x0), self.__convert_tensor__(x0_class), self.__convert_tensor__(x1), self.__convert_tensor__(x0_time), self.__convert_tensor__(x1_time)


        x, ut, t_model, futuretime, t = self.__x_processing__(x0, x1, x0_time, x1_time)
        

        if len(x0_class.shape) == 3:
            x0_class = x0_class.squeeze()

        in_tensor = torch.cat([x, x0_class, t_model], dim=-1)
        output = self.flow_model.forward_train(in_tensor)

        loss = self._compute_flow_loss(output, x1, futuretime, t, in_tensor)

        # Noise model training (only for non-distributional or SDE)
        if self.distributional is None:
            if self.implementation == "SDE":
                xt = output
                sde_noise = self.noise_model.forward_train(in_tensor)
                variance = torch.sqrt(t*(1-t))*sde_noise
                noise = torch.randn_like(xt[:,:self.dim]) * variance
                uncertainty = (xt[:,:self.dim].clone().detach() + noise)
                noise_loss = self.loss_fn(uncertainty, x1)
            else:
                xt = output
                uncertainty = torch.abs(xt[:,:self.dim].clone().detach() - x1)
                noise_loss = self.loss_fn(self.noise_model.forward_train(in_tensor)[:,:self.dim], uncertainty)
        else:
            noise_loss = torch.tensor(0.0, device=x0.device)

        flow_opt.zero_grad()
        self.manual_backward(loss)
        flow_opt.step()
        
        if self.distributional is None:
            noise_opt.zero_grad()
            self.manual_backward(noise_loss)
            noise_opt.step()
        
        self.log('train_loss', loss)
        self.log('noise_loss', noise_loss)
        return loss + noise_loss

    def _compute_flow_loss(self, output, x1_target, futuretime, t, in_tensor):
        """Compute flow model loss based on distributional mode."""
        if self.distributional == "gaussian_nll":
            mu, log_var, _cond, time_remaining = output
            loss_coord = gaussian_nll_loss(mu, log_var, x1_target[:, :self.dim])
            loss_time = mse_loss(time_remaining.squeeze(-1), futuretime.squeeze(-1))
            return loss_coord + loss_time

        elif self.distributional == "mdn":
            pi, mu, sigma, _cond, time_remaining = output
            loss_coord = mdn_loss(pi, mu, sigma, x1_target[:, :self.dim])
            loss_time = mse_loss(time_remaining.squeeze(-1), futuretime.squeeze(-1))
            return loss_coord + loss_time

        elif self.distributional == "quantile":
            quantile_preds, _cond, time_remaining = output
            loss_coord = quantile_loss(quantile_preds, x1_target[:, :self.dim],
                                       quantiles=self.quantile_levels)
            loss_time = mse_loss(time_remaining.squeeze(-1), futuretime.squeeze(-1))
            return loss_coord + loss_time

        elif self.distributional == "l1_variance":
            pred_x1 = output[:, :self.dim]
            pred_time = output[:, -1]
            loss_coord = l1_variance_loss(pred_x1, x1_target[:, :self.dim],
                                          lambda_var=self.lambda_var)
            loss_time = mse_loss(pred_time, futuretime.squeeze(-1))
            return loss_coord + loss_time

        else:
            xt = output
            if self.implementation == "SDE":
                sde_noise = self.noise_model.forward_train(in_tensor)
                variance = torch.sqrt(t*(1-t))*sde_noise
                noise = torch.randn_like(xt[:,:self.dim]) * variance
                loss = self.loss_fn(xt[:,:self.dim] + noise.clone().detach(), x1_target) + \
                       self.loss_fn(xt[:,-1], futuretime.squeeze(-1))
            else:
                loss = self.loss_fn(xt[:,:self.dim], x1_target) + \
                       self.loss_fn(xt[:,-1], futuretime.squeeze(-1))
            return loss

    def configure_optimizers(self):
        print("configuring optimizers")
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.lr)
        self.noise_optimizer = torch.optim.Adam(self.noise_model.parameters(), lr=self.lr)
        return [self.flow_optimizer, self.noise_optimizer]
    
    def validation_step(self, batch, batch_idx):
        """validation_step

        Args:
            batch (_type_): batch size of 1 (since uneven)
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        loss, pairs, metricD, noise_loss, noise_pair = self.test_func_step(batch, batch_idx, mode='val')
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('noise_val_loss', noise_loss, on_epoch=True, on_step=False, sync_dist=True)
        for key, value in metricD.items():
            self.log(key+"_val", value, on_epoch=True, on_step=False, sync_dist=True)
        # return total_loss, traj_pairs
        return {'val_loss':loss, 'traj_pairs':pairs}

    def test_step(self, batch, batch_idx):
        loss, pairs, metricD, noise_loss, noise_pair = self.test_func_step(batch, batch_idx, mode='test')
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('noise_test_loss', noise_loss, on_epoch=True, on_step=False, sync_dist=True)
        for key, value in metricD.items():
            self.log(key+"_test", value, on_epoch=True, on_step=False, sync_dist=True)
        # return total_loss, traj_pairs
        return {'test_loss':loss, 'traj_pairs':pairs}
    
    def test_func_step(self, batch, batch_idx, mode='none'):
        """assuming each is one patient/batch"""
        total_loss = []
        traj_pairs = []

        total_noise_loss = []
        noise_pairs = []

        x0_values, x0_classes, x1_values, times_x0, times_x1 = batch
        times_x0 = times_x0.squeeze()
        times_x1 = times_x1.squeeze()

        # print(x0_values.shape)
        # print(x1_values.shape)
        full_traj = torch.cat([x0_values[0,0,:self.dim].unsqueeze(0), 
                               x1_values[0,:,:self.dim]], 
                               dim=0)
        full_time = torch.cat([times_x0[0].unsqueeze(0), times_x1], dim=0)
        ind_loss, pred_traj, noise_mse, noise_pred = self.test_trajectory(batch)
        total_loss.append(ind_loss)
        traj_pairs.append([full_traj, pred_traj])
        noise_pairs.append([full_traj, noise_pred])
        total_noise_loss.append(noise_mse)

        full_traj = full_traj.detach().cpu().numpy()
        pred_traj = pred_traj.detach().cpu().numpy()
        full_time = full_time.detach().cpu().numpy()

        # graph
        fig = plot_3d_path_ind_noise(pred_traj, 
                               full_traj,
                                noise_pred, 
                               t_span=full_time,
                               title="{}_trajectory_patient_{}".format(mode, batch_idx))
        dim_fig = plot_trajectory_dimensions(
            pred_traj,
            full_traj,
            t_span=full_time,
            title="{}_trajectory_dimensions_patient_{}".format(mode, batch_idx),
        )
        if self.logger:
            # may cause problem if wandb disabled
            self.logger.experiment.log(
                {
                    "{}_trajectory_patient_{}".format(mode, batch_idx): wandb.Image(fig),
                    "{}_trajectory_dimensions_patient_{}".format(mode, batch_idx): wandb.Image(dim_fig),
                }
            )
        
        plt.close(fig)
        plt.close(dim_fig)

        # metrics
        metricD = metrics_calculation(pred_traj, full_traj, metrics=self.metrics)
        return np.mean(total_loss), traj_pairs, metricD, np.mean(total_noise_loss), noise_pairs

    def test_trajectory(self,pt_tensor):
        if self.implementation == "ODE":
            return self.test_trajectory_ode(pt_tensor)
        elif self.implementation == "SDE":
            return self.test_trajectory_sde(pt_tensor)
    
    def test_trajectory_ode(self,pt_tensor):
        print(f"Using ODE with {self.ode_t_span_points} points per interval.")
        """test_trajectory

        Args:
            pt_tensor (numpy.array): (x0_values, x0_classes, x1_values, times_x0, times_x1),


        Returns:
            mse_all, total_pred_tensor: _description_
        """
        node = NeuralODE(
            torch_wrapper_tv(self.flow_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        node_noise = NeuralODE(
            torch_wrapper_tv(self.noise_model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        total_pred = []
        noise_pred = []
        mse = []
        noise_mse = []
        # t_max = 0

        x0_values, x0_classes, x1_values, times_x0, times_x1 = pt_tensor
        # squeeze all
        x0_values = x0_values.squeeze(0)
        x1_values = x1_values.squeeze(0)
        times_x0 = times_x0.squeeze()
        times_x1 = times_x1.squeeze()
        x0_classes = x0_classes.squeeze()

        if len(x0_classes.shape) == 1:
            x0_classes = x0_classes.unsqueeze(1)

        total_pred.append(x0_values[0].unsqueeze(0))
        len_path = x0_values.shape[0]
        assert len_path == x1_values.shape[0]

        time_history = x0_classes[0][-(self.memory*self.dim):]

        for i in range(len_path): 
            time_span = self.__convert_tensor__(
                torch.linspace(times_x0[i], times_x1[i], self.ode_t_span_points)
            ).to(x0_values.device)

            new_x_classes = torch.cat([x0_classes[i][:-(self.memory*self.dim)].unsqueeze(0), time_history.unsqueeze(0)], dim=1)
            with torch.no_grad():
                # get last pred, if none then use startpt
                if i == 0:
                    testpt = torch.cat([x0_values[i].unsqueeze(0),new_x_classes],dim=1)
                else: # incorporate last prediction
                    testpt = torch.cat([pred_traj, new_x_classes], dim=1)
                # print(testpt.shape)
                traj = node.trajectory(
                    testpt,
                    t_span=time_span, 
                )
                # add noise prediction
                noise_traj = node_noise.trajectory(
                    testpt,
                    t_span=time_span, 
                )

            pred_traj = traj[-1,:,:self.dim]
            noise_traj = noise_traj[-1,:,:self.dim]
            total_pred.append(pred_traj)
            noise_pred.append(noise_traj)
            
            ground_truth_coords = x1_values[i]
            mse_traj = self.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy()
            mse.append(mse_traj)
            uncertainty_traj = ground_truth_coords - pred_traj
            noise_mse_traj = self.loss_fn(noise_traj, uncertainty_traj).detach().cpu().numpy()
            noise_mse.append(noise_mse_traj)
            
            # history update
            flattened_coords = pred_traj.flatten()
            time_history = torch.cat([time_history[self.dim:].unsqueeze(0), flattened_coords.unsqueeze(0)], dim=1).squeeze()

        mse_all = np.mean(mse)
        total_pred_tensor = torch.stack(total_pred).squeeze(1)
        noise_pred = torch.stack(noise_pred).squeeze(1)
        return mse_all, total_pred_tensor, noise_mse, noise_pred

    
    def test_trajectory_sde(self,pt_tensor):
        """test_trajectory

        Args:
            pt_tensor (numpy.array): (x0_values, x0_classes, x1_values, times_x0, times_x1),


        Returns:
            mse_all, total_pred_tensor: _description_
        """
        sde = SDE_func_solver(self.flow_model, noise=self.noise_model)
        total_pred = []
        mse = []
        noise_pred = []
        noise_mse = []
        t_max = 0

        x0_values, x0_classes, x1_values, times_x0, times_x1 = pt_tensor
        # squeeze all
        x0_values = x0_values.squeeze(0)
        x1_values = x1_values.squeeze(0)
        times_x0 = times_x0.squeeze()
        times_x1 = times_x1.squeeze()
        x0_classes = x0_classes.squeeze()

        if len(x0_classes.shape) == 1:
            x0_classes = x0_classes.unsqueeze(1)



        total_pred.append(x0_values[0].unsqueeze(0))
        len_path = x0_values.shape[0]
        assert len_path == x1_values.shape[0]

        time_history = x0_classes[0][-(self.memory*self.dim):]

        for i in range(len_path): 

            time_span = self.__convert_tensor__(
                torch.linspace(times_x0[i], times_x1[i], self.ode_t_span_points)
            ).to(x0_values.device)

            new_x_classes = torch.cat([x0_classes[i][:-(self.memory*self.dim)].unsqueeze(0), time_history.unsqueeze(0)], dim=1)
            with torch.no_grad():
                # get last pred, if none then use startpt
                if i == 0:
                    testpt = torch.cat([x0_values[i].unsqueeze(0),new_x_classes],dim=1)
                else: # incorporate last prediction
                    testpt = torch.cat([pred_traj, new_x_classes], dim=1)
                traj, noise_traj = self._sde_solver(sde, testpt, time_span)

            pred_traj = traj[-1,:,:self.dim]
            noise_traj = noise_traj[-1,:,:self.dim]

            total_pred.append(pred_traj)
            noise_pred.append(noise_traj)

            ground_truth_coords = x1_values[i]
            calculated_mse = self.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy()
            mse.append(calculated_mse)
            noise_mse.append(calculated_mse)

            # history update
            flattened_coords = pred_traj.flatten()
            time_history = torch.cat([time_history[self.dim:].unsqueeze(0), flattened_coords.unsqueeze(0)], dim=1).squeeze()
            
            

        mse_all = np.mean(mse)
        noise_mse_all = np.mean(noise_mse)
        total_pred_tensor = torch.stack(total_pred).squeeze(1)
        noise_pred_tensor = torch.stack(noise_pred).squeeze(1)
        return mse_all, total_pred_tensor, noise_mse_all, noise_pred_tensor


    def _sde_solver(self, sde, initial_state, time_span):
        dt = time_span[1] - time_span[0]  # Time step
        current_state = initial_state
        trajectory = [current_state]
        noise_trajectory = []

        for t in time_span[1:]:
            drift = sde.f(t, current_state)
            diffusion = sde.g(t, current_state)
            noise = torch.randn_like(current_state) * torch.sqrt(dt)
            current_state = current_state + drift * dt + diffusion * noise # @NEED this or not?
            trajectory.append(current_state)
            pred_diff = diffusion * noise
            noise_trajectory.append(pred_diff)

        return torch.stack(trajectory), torch.stack(noise_trajectory)
        


    
