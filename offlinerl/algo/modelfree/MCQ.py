import copy

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from loguru import logger
from torch.distributions import Normal, kl_divergence

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import to_torch
from offlinerl.utils.net.common import Net, MLP
from offlinerl.utils.net.continuous import Critic
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

import os

class VAE(torch.nn.Module):
    def __init__(self, state_dim, action_dim, vae_features, vae_layers, max_action=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = 2 * self.action_dim
        self.max_action = max_action

        self.encoder = MLP(self.state_dim + self.action_dim, 2 * self.latent_dim, vae_features, vae_layers, hidden_activation='relu')
        self.decoder = MLP(self.state_dim + self.latent_dim, self.action_dim, vae_features, vae_layers, hidden_activation='relu')
        self.noise = MLP(self.state_dim + self.action_dim, self.action_dim, vae_features, vae_layers, hidden_activation='relu')

    def encode(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        mu, logstd = torch.chunk(self.encoder(state_action), 2, dim=-1)
        logstd = torch.clamp(logstd, -4, 15)
        std = torch.exp(logstd)
        return Normal(mu, std)

    def decode(self, state, z=None):
        if z is None:
            param = next(self.parameters())
            z = torch.randn((*state.shape[:-1], self.latent_dim)).to(param)
            z = torch.clamp(z, -0.5, 0.5)

        action = self.decoder(torch.cat([state, z], dim=-1))
        action = self.max_action * torch.tanh(action)

        return action
    
    def decode_multiple(self, state, z=None, num=10):
        if z is None:
            param = next(self.parameters())
            z = torch.randn((num, *state.shape[:-1], self.latent_dim)).to(param)
            z = torch.clamp(z, -0.5, 0.5)
        state = state.repeat((num,1,1))

        action = self.decoder(torch.cat([state, z], dim=-1))
        action = self.max_action * torch.tanh(action)
        # shape: (num, batch size, state shape+action shape)
        return action

    def forward(self, state, action):
        dist = self.encode(state, action)
        z = dist.rsample()
        action = self.decode(state, z)
        return dist, action

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape, get_env_action_range
        obs_shape, action_shape = get_env_shape(args['task'])
        max_action, _ = get_env_action_range(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    
    net_a = Net(layer_num = args['layer_num'], 
                     state_shape = obs_shape,
                     hidden_layer_size = args['hidden_layer_size'])
    
    actor = TanhGaussianPolicy(preprocess_net = net_a,
                                action_shape = action_shape,
                                hidden_layer_size = args['hidden_layer_size'],
                                conditioned_sigma = True,
                              ).to(args['device'])
    
    actor_optim = optim.Adam(actor.parameters(), lr=args['actor_lr'])

    vae = VAE(obs_shape, action_shape, args['vae_features'], args['vae_layers'], max_action).to(args['device'])
    vae_optim = torch.optim.Adam(vae.parameters(), lr=args['vae_lr'])
    
    net_c1 = Net(layer_num = args['layer_num'],
                  state_shape = obs_shape,  
                  action_shape = action_shape,
                  concat = True, 
                  spectral_normalization = False,
                  hidden_layer_size = args['hidden_layer_size'])
    critic1 = Critic(preprocess_net = net_c1,  
                     hidden_layer_size = args['hidden_layer_size'],
                    ).to(args['device'])
    critic1_optim = optim.Adam(critic1.parameters(), lr=args['critic_lr'])
    
    net_c2 = Net(layer_num = args['layer_num'],
                  state_shape = obs_shape,  
                  action_shape = action_shape,
                  concat = True, 
                  spectral_normalization = False,
                  hidden_layer_size = args['hidden_layer_size'])
    critic2 = Critic(preprocess_net = net_c2, 
                     hidden_layer_size = args['hidden_layer_size'],
                    ).to(args['device'])
    critic2_optim = optim.Adam(critic2.parameters(), lr=args['critic_lr'])
    
    if args["use_automatic_entropy_tuning"]:
        if args["target_entropy"]:
            target_entropy = args["target_entropy"]
        else:
            target_entropy = -np.prod(args["action_shape"]).item() 
        log_alpha = torch.zeros(1,requires_grad=True, device=args['device'])
        alpha_optimizer = optim.Adam(
            [log_alpha],
            lr=args["actor_lr"],
        )
        
    nets =  {
        "vae" : {"net" : vae, "opt" : vae_optim},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "critic1" : {"net" : critic1, "opt" : critic1_optim},
        "critic2" : {"net" : critic2, "opt" : critic2_optim},
        "log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer, "target_entropy": target_entropy},
        
    }
        
    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.seed = args['seed']
        
        self.actor = algo_init["actor"]["net"]
        self.actor_opt = algo_init["actor"]["opt"]

        self.vae = algo_init['vae']['net']
        self.vae_optim = algo_init['vae']['opt']
        
        self.critic1 = algo_init["critic1"]["net"]
        self.critic1_opt = algo_init["critic1"]["opt"]
        self.critic2 = algo_init["critic2"]["net"]
        self.critic2_opt = algo_init["critic2"]["opt"]
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        if args["use_automatic_entropy_tuning"]:
            self.log_alpha = algo_init["log_alpha"]["net"]
            self.alpha_opt = algo_init["log_alpha"]["opt"]
            self.target_entropy = algo_init["log_alpha"]["target_entropy"]
            
        self.critic_criterion = nn.MSELoss()

        self.lam = args['lam']
        self.num = args['num']
        self.normalize = args['normalize'] # ought to be True or False
        
        self._n_train_steps_total = 0
        self._current_epoch = 0

        from offlinerl.utils.env import get_env_action_range
        action_max, _ = get_env_action_range(args['task'])
        self.act_max = action_max

        if not os.path.exists("{}".format(args['log_dir'])):
            os.makedirs("{}".format(args['log_dir']))
        self.task = args['task'][5:]
        
    def forward(self, obs, reparameterize=True, return_log_prob=True):
        log_prob = None
        tanh_normal = self.actor(obs,reparameterize=reparameterize,)

        if return_log_prob:
            if reparameterize is True:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
            log_prob = tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            if reparameterize is True:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
        return action, log_prob
    
    def _get_tensor_values(self, obs, vae=None, network=None):
        obs_repeat = obs.repeat((self.num,1,1)).reshape(-1, obs.shape[-1])
        if vae is None:
            repeat_actions, _ = self.forward(obs_repeat, reparameterize=True, return_log_prob=True)
            preds = network(obs_repeat, repeat_actions)
        else:
            # pesudo_action = vae.decode_multiple(obs, num=self.num)
            repeat_actions = vae.decode(obs_repeat)
            repeat_actions = repeat_actions.reshape(self.num*self.args["batch_size"], -1)
            preds = network(obs_repeat, repeat_actions)
            preds = preds.reshape(self.num, obs.shape[0], 1)
            preds = torch.max(preds, dim=0)[0]
            preds = preds.clamp(min=0).repeat((self.num,1,1)).reshape(-1,1)
        return preds, repeat_actions.view(self.num, self.args["batch_size"], -1)
        
    def _train(self, batch, tblogger):
        self._current_epoch += 1
        batch = to_torch(batch, torch.float, device=self.args["device"])
        rewards = batch.rew
        terminals = batch.done
        obs = batch.obs
        actions = batch.act
        next_obs = batch.obs_next
        
        ## normalization
        obs = (obs - self.mean)/(self.std + 1e-6)
        next_obs = (next_obs - self.mean)/(self.std + 1e-6)

        # train vae
        dist, _action = self.vae(obs, actions)
        kl_loss = kl_divergence(dist, Normal(0, 1)).sum(dim=-1).mean()
        recon_loss = ((actions - _action) ** 2).sum(dim=-1).mean()
        vae_loss = kl_loss + recon_loss

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

        """
        Policy and Alpha Loss
        """
        new_obs_actions, log_pi = self.forward(obs)
        
        if self.args["use_automatic_entropy_tuning"]:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        
        """
        QF Loss
        """
        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)
        
        new_next_actions,new_log_pi= self.forward(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        
        target_q_values = torch.min(
            self.critic1_target(next_obs, new_next_actions),
            self.critic2_target(next_obs, new_next_actions),
        )
        
        target_q_values = target_q_values - alpha * new_log_pi
        q_target = self.args["reward_scale"] * rewards + (1. - terminals) * self.args["discount"] * target_q_values.detach()
        
        def weight(diff):
            return torch.where(diff>=0.05, 0, 1)
        
        pesudo_next_action = self.vae.decode(next_obs)
        if self.task in ['hopper-medium-v2', 'hopper-medium-expert-v2', 'hopper-expert-v2', 'halfcheetah-medium-expert-v2', 'halfcheetah-expert-v2']:
            next_action_diff = torch.mean((new_next_actions - pesudo_next_action)**2, dim=-1, keepdim=True)
            bellman_weight = weight(next_action_diff)
        
        ## OOD Q1
        q1_ood_curr_pred, q1_ood_curr_act = self._get_tensor_values(obs, network=self.critic1)
        q1_ood_next_pred, q1_ood_next_act = self._get_tensor_values(next_obs, network=self.critic1)
        q1_ood_pred = torch.cat([q1_ood_curr_pred, q1_ood_next_pred],0)

        pesudo_q1_curr_target, q1_curr_act = self._get_tensor_values(obs, network=self.critic1, vae=self.vae)
        pesudo_q1_next_target, q1_next_act = self._get_tensor_values(next_obs, network=self.critic1, vae=self.vae)
        pesudo_q1_target = torch.cat([pesudo_q1_curr_target, pesudo_q1_next_target],0)
        pesudo_q1_target = pesudo_q1_target.detach()

        assert pesudo_q1_target.shape[0] == q1_ood_pred.shape[0]

        ## OOD Q2
        q2_ood_curr_pred, q2_ood_curr_act = self._get_tensor_values(obs, network=self.critic2)
        q2_ood_next_pred, q2_ood_next_act = self._get_tensor_values(next_obs, network=self.critic2)
        q2_ood_pred = torch.cat([q2_ood_curr_pred, q2_ood_next_pred],0)

        pesudo_q2_curr_target, q2_curr_act = self._get_tensor_values(obs, network=self.critic2, vae=self.vae)
        pesudo_q2_next_target, q2_next_act = self._get_tensor_values(next_obs, network=self.critic2, vae=self.vae)
        pesudo_q2_target = torch.cat([pesudo_q2_curr_target, pesudo_q2_next_target])
        pesudo_q2_target = pesudo_q2_target.detach()

        pesudo_q_target = torch.min(pesudo_q1_target, pesudo_q2_target)

        assert pesudo_q2_target.shape[0] == q2_ood_pred.shape[0]

        # stop vanilla pessimistic estimate becoming large
        qf1_deviation = q1_ood_pred - pesudo_q_target
        
        # can also be disabled
        if self.task != 'walker2d-medium-replay-v2':
            qf1_deviation[qf1_deviation <= 0] = 0
        
        qf2_deviation = q2_ood_pred - pesudo_q_target
        
        # can also be disabled
        if self.task != 'walker2d-medium-replay-v2':
            qf2_deviation[qf2_deviation <= 0] = 0
        
        if self.task in ['hopper-medium-v2', 'hopper-medium-expert-v2', 'hopper-expert-v2', 'halfcheetah-medium-expert-v2', 'halfcheetah-expert-v2']:
            # to ensure policy deviation between learned policy and behavior policy is not large (e.g., the assumption in Proposition 5)
            q1_curr_diff = torch.mean((q1_ood_curr_act - q1_curr_act)**2, dim=-1, keepdim=True)
            q1_next_diff = torch.mean((q1_ood_next_act - q1_next_act)**2, dim=-1, keepdim=True)
            q2_curr_diff = torch.mean((q2_ood_curr_act - q2_curr_act)**2, dim=-1, keepdim=True)
            q2_next_diff = torch.mean((q2_ood_next_act - q2_next_act)**2, dim=-1, keepdim=True)
            q1_diff = torch.cat([q1_cur_diff, q1_next_diff],0)
            q2_diff = torch.cat([q2_cur_diff, q2_next_diff],0)
            q1_weight = 1-weight(q1_diff).view(-1, 1)
            q2_weight = 1-weight(q2_diff).view(-1, 1)
            
            qf1_loss = self.lam * (bellman_weight * (q1_pred - q_target)**2).mean() + (1-self.lam) * (q1_weight * qf1_deviation**2).mean()
            qf2_loss = self.lam * (bellman_weight * (q2_pred - q_target)**2).mean() + (1-self.lam) * (q2_weight * qf2_deviation**2).mean()
        else:          
            qf1_ood_loss = torch.mean(qf1_deviation**2)
            qf2_ood_loss = torch.mean(qf2_deviation**2)

            qf1_loss = self.lam * self.critic_criterion(q1_pred, q_target) + (1-self.lam) * qf1_ood_loss
            qf2_loss = self.lam * self.critic_criterion(q2_pred, q_target) + (1-self.lam) * qf2_ood_loss

        """
        Update critic networks
        """
        self.critic1_opt.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        qf2_loss.backward()
        self.critic2_opt.step()

        if self._current_epoch % 500 == 0:
            tblogger.log('train/q1', q1_pred.mean(), self._current_epoch)
            tblogger.log('train/q target', q_target.mean(), self._current_epoch)
            tblogger.log('train/q1 ood', q1_ood_pred.mean(), self._current_epoch)
            tblogger.log('train/q ood target', pesudo_q_target.mean(), self._current_epoch)
            tblogger.log('train/q1 critic loss', qf1_loss, self._current_epoch)
            # tblogger.log('train/q2 critic loss', qf2_loss, self._current_epoch)

        q_new_actions = torch.min(
            self.critic1(obs, new_obs_actions),
            self.critic2(obs, new_obs_actions),
        )

        policy_loss = (alpha*log_pi - q_new_actions).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        if self._current_epoch % 500 == 0:
            tblogger.log('train/actor q', q_new_actions.mean(), self._current_epoch)
            tblogger.log('train/actor loss', q_new_actions.mean(), self._current_epoch)

        """
        Soft Updates target network
        """
        self._sync_weight(self.critic1_target, self.critic1, self.args["soft_target_tau"])
        self._sync_weight(self.critic2_target, self.critic2, self.args["soft_target_tau"])
        
        self._n_train_steps_total += 1

        
    def get_model(self):
        return self.actor
    
    #def save_model(self, model_save_path):
    #    torch.save(self.actor, model_save_path)
        
    def get_policy(self):
        return self.actor
    
    def train(self, train_buffer, val_buffer, callback_fn, tblogger):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        if self.normalize:
            sur_obs = torch.from_numpy(train_buffer.obs).to(self.args["device"])
            self.mean = torch.mean(sur_obs, dim=0, keepdim=True)
            self.std = torch.std(sur_obs, dim=0, keepdim=True)
        else:
            self.mean = 0
            self.std = 1

        for epoch in range(1,self.args["max_epoch"]+1):
            for step in range(1,self.args["steps_per_epoch"]+1):
                train_data = train_buffer.sample(self.args["batch_size"])
                self._train(train_data, tblogger)
               
            if self.normalize:
                mean = self.mean.cpu().numpy()
                std = self.std.cpu().numpy()
            else:
                mean = self.mean
                std = self.std
            
            res = callback_fn(self.get_policy(), epoch, mean, std)
            
            if epoch % 5 == 0:

                self.log_res(epoch, res)

                ## record statistics
                tblogger.log('eval/d4rl_score', res["score"], epoch)
                tblogger.log('eval/return', res["Reward_Mean_Env"], epoch)
                tblogger.log('eval/episode length', res["Length_Mean_Env"], epoch)
            
        return self.get_policy()
