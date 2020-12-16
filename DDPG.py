import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import airsim_env

# Hyperparameters
lr_mu = 0.0005
lr_q = 0.001
gamma = 0.99
batch_size = 256
buffer_limit = 50000
tau = 0.005  # for target network soft update


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))  # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_sa = nn.Linear(9 + 4, 128)
        self.fc_a = nn.Linear(128, 64)
        self.fc_q = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        h1 = F.relu(self.fc_sa(cat))
        h2 = F.relu(self.fc_a(h1))
        q = F.relu(self.fc_q(h2))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def main():
    env = airsim_env.windENV()
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            a = mu(torch.from_numpy(s).float())
            a = a.detach().numpy()
            a = (a + np.random.normal(0, 0.1, size=4)).clip(-0.6, 0.4)
            s_prime, r, done, info = env.step(a)
            memory.put((s, a, r / 100.0, s_prime, done))
            score += r
            s = s_prime

            if memory.size() > 1000:
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

        if n_epi % 1000 == 0 and n_epi != 0:
            utils.save_models(mu, q, n_epi)


def evaluate():
    env = airsim_env.windENV()

    q = QNet()
    mu = MuNet()

    mu.load_state_dict(torch.load('./Models/' + '2000_actor.pt'))
    q.load_state_dict(torch.load('./Models/' + '2000_critic.pt'))
    score = 0.0

    for n_epi in range(5):
        s = env.reset()
        done = False

        while not done:
            a = mu(torch.from_numpy(s).float())
            a = a.detach().numpy()
            a = (a + np.random.normal(0, 0.1, size=4)).clip(-0.6, 0.4)
            s_prime, r, done, info = env.step(a)
            score += r
            s = s_prime

        print("episode: {}, reward: {:.1f}".format(n_epi, score))
        score = 0.0

if __name__ == '__main__':
    # main()
    evaluate()
