import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from utils import OUNoise, ExperienceMemory

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


criterion = nn.MSELoss()

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).type(dtype)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        #print(type(nb_actions))
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        self.init_weights(init_w)
        #self.dense1_bn = nn.BatchNorm1d((1,hidden2))
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        #self.fc1.weight.data.uniform_(-init_w, init_w)
        #self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        #torch.nn.init.xavier_uniform_(self.fc1.weight)
        #torch.nn.init.xavier_uniform_(self.fc2.weight)
        #torch.nn.init.xavier_uniform_(self.fc3.weight)

    
    def forward(self, x):
        #print("input:",x)
        out = self.fc1(x)
        out = self.relu(out)
        #print("  fc1:",out)
        out = self.fc2(out)
        out = self.relu(out)
        #print("    fc2",out)
        out = self.fc3(out)
        out = self.tanh(out)
        #print("       fc3",out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        #print("nb_actions:",nb_actions)
        self.fc1 = nn.Linear(nb_states, hidden1)
        
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        #self.fc1.weight.data.uniform_(-init_w, init_w)
        #self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        #print("out size:",x.size(1), self.fc1.weight.size(1))
        #print("a size:",a.size(1), self.fc1.weight.size(1))
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

class Agent(object):
    def __init__( self, task, hp ):
                   
        self.task = task
        self.nb_states = task.state_size
        self.nb_actions = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        
        # why not use bits to represent continous action space as discrete actions :)
        self.action_bits = 8 # np.floor( np.log2(self.action_range) + 1 )
        self.action_size = (self.nb_actions*self.action_bits) #.astype(np.int)
        self.action_factor = self.action_high / 2**self.action_bits
        
        self.use_cuda = 1 if hp['USE_CUDA'] is True else 0
        
        if int(hp['SEED']) > 0:
            self.seed(hp['SEED'])
        
        self.buffer_size = hp['EXP_BUFFER_SIZE']
        self.batch_size = hp['EXP_BATCH_SIZE']

        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':hp['HIDDEN1'], 
            'hidden2':hp['HIDDEN2'], 
            'init_w':hp['INIT_W']
        }

        self.actor = Actor(self.nb_states, self.action_size, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.action_size, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=hp['ACTOR_LR'])

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=hp['CRITIC_LR'])
            
        self.hard_copy( self.actor, self.actor_target ) 
        self.hard_copy( self.critic, self.critic_target )
        
        # Create experience memory buffer
        self.memory = ExperienceMemory(self.buffer_size, self.batch_size)
        
        # init the process of life ... ..
        self.random_process = OUNoise(size=self.nb_actions, theta=hp['OU_THETA'], mu=hp['OU_MU'], sigma=hp['OU_SIGMA'])
        self.ou_decay = hp['OU_DECAY']

        # Hyper-parameters
        #self.batch_size = hp.BATCH_SIZE
        self.tau = hp['TAU']
        self.discount = hp['DISCOUNT']
        #self.depsilon = 1.0 / args.epsilon

        # 
        #self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # nvidia
        if hp['USE_CUDA']: 
            self.cuda()
            
    def hard_copy(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - self.tau) + param.data * self.tau )


    def update_policy(self):
        
        # Get Sample batches
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.batch_samples(self.batch_size)
        
        #state_batch = state_batch / 360
        #next_state_batch = next_state_batch / 360
        #print(action_batch)

        ###########################################
        # Prepare for the target q batch
        with torch.no_grad(): # no grad calc
            next_actions = []
            for action in self.actor_target(to_tensor(next_state_batch)):
                #print(action)
                action = to_numpy(action)
                next_actions.append(np.array(self.action_transform(action)))
                
            next_q_values = self.critic_target([to_tensor(next_state_batch),to_tensor(np.array(next_actions))])

            # Q_targets = (rewards + self.gamma * Q_targets_next.reshape(len(experiences)) * (1 - dones))
            target_q_batch = to_tensor(reward_batch) + \
                self.discount*to_tensor(1 - terminal_batch.astype(np.float))*next_q_values

        ############################################
        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        #print("vloss:",value_loss)
        value_loss.backward()
        self.critic_optim.step()

        ##############################################
        # Actor update
        self.actor.zero_grad()
        
        next_actions = []
        for action in self.actor_target(to_tensor(state_batch)):
            #print(action)
            action = to_numpy(action)
            next_actions.append(np.array(self.action_transform(action)))
            
            
        policy_loss = -self.critic([
            to_tensor(state_batch),
            to_tensor(np.array(next_actions))
        ])

        policy_loss = policy_loss.mean()
        #print("ploss:",policy_loss)
        policy_loss.backward()
        self.actor_optim.step()

        ###############################################
        # Target update
        self.soft_update( self.actor, self.actor_target )
        self.soft_update( self.critic, self.critic_target )
        
        return None, None #value_loss.detach().squeeze().numpy() ,policy_loss.detach().squeeze().numpy()

        
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.is_training = False

    def action_transform(self,action):
        # this dependes on our output activation function
        action[action <= 0.] = 0
        action[action > 0.] = 1
        action = np.array(np.split(action,self.nb_actions)).astype(np.bool)
        action = np.packbits(action).astype(np.float)#, axis=-1)
        action = action*self.action_factor
        return action
        
    def cuda(self):
        self.use_cuda = 1
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
        
        
    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.a_t = action
        self.observe( reward, next_state, done )

        # If we got our minibatch of experience memories..
        # learn from them and slowly change belief :)
        aloss = None
        ploss = None
        if len(self.memory) > self.batch_size:           
            aloss, ploss = self.update_policy()
        return aloss,ploss

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.add(self.s_t, self.a_t, r_t, s_t1, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(0.,900.,self.nb_actions)
        self.a_t = action
        return action
    
    def act(self, s_t, i_episode=0, decay_epsilon=True):
        
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        
        #action = (action * self.action_range) + self.action_low
        action = self.action_transform( action )
        #np.packbits(a, axis=-1)
        
        if(self.ou_decay != 0):
            decay = 1 - (i_episode*self.ou_decay)
            #print(action, decay)
            action += self.is_training*decay*self.random_process.sample()
            
        self.a_t = action        
        return action

    def reset(self):
        self.s_t = self.task.reset()
        self.random_process.reset()
        return self.s_t

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)