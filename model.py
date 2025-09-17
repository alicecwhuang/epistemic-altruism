import random
import numpy as np
import pandas as pd
import ast


# #### Perfect communication model in BSM

# In[3]:


class Agent():
    def __init__(self, model, id, reliability):
        self.model = model
        self.id = id
        self.r = reliability
        self.peers = [] # list of all scientists
        self.urn = np.array([]) # probability [nature, agent 1, agent 2,...]
        self.belief = 0 # current belief
        self.choice = None
        self.n_success = 0
    def choose(self):
        prob = self.urn/sum(self.urn)
        return np.random.choice(len(self.urn), size=1, p=prob)[0]
    def experiment(self):
        return 1 if random.random() < self.r else 0
    def update(self):
        self.choice = int(self.choose())
        if self.choice == 0:
            self.belief = self.experiment()
        else:
            self.belief = self.peers[self.choice - 1].belief
        self.n_success += self.belief
    def reinforce(self):
        if self.belief:
            self.urn[self.choice] += 1


# In[4]:


class Model():
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        """lb: lowest reliability amongst agents"""
        self.n = n
        self.round_per_gen = round_per_gen
        self.gen_per_run = gen_per_run
        self.agents = [] # list of agents
        self.Rs = [round(random.random(), 2) for i in range(n)] # varying reliability
        """Create agents"""
        for i in range(n):
            self.agents.append(Agent(self, i, self.Rs[i]))
        self.init_agents()
    def init_agents(self):
        for a in self.agents:
            a.peers = self.agents
            a.urn = np.array([1] * (self.n + 1))
            a.urn[a.id+1] = 0
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls)
        for i in ls:
            a = self.agents[i]
            a.update()
            a.reinforce()
    def run(self):
        for i in range(self.gen_per_run):
            for a in self.agents:
                a.belief = a.experiment()
            for i in range(self.round_per_gen):
                self.play()


# #### BH Model (My Version)

# In[13]:


class BHAgent(Agent):
    def __init__(self, model, id, reliability, n_votes=3):
        super().__init__(model, id, reliability)
        self.n_votes = n_votes
    def choose(self):
        choice = np.array([0]*len(self.urn))
        for i in range(self.n_votes):
            prob = self.urn/sum(self.urn)
            pick = np.random.choice(range(len(self.urn)), p=prob)
            choice[pick] += 1
            self.urn[pick] -= 1
        self.urn = self.urn + choice
        return choice
    def update(self):
        self.choice = self.choose()
        votes = 0
        for i in range(self.choice[0]):
            votes += self.experiment()
        votes += sum(self.choice[1:] * np.array([a.belief for a in self.model.agents]))
        if votes > (sum(self.choice) * 0.5):
            self.belief = 1
        else:
            self.belief = 0
        self.n_success += self.belief
    def reinforce(self):
        if self.belief:
            self.urn = self.urn + self.choice


# In[14]:


class BHModel(Model):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        """lb: lowest reliability amongst agents"""
        super().__init__(n, round_per_gen, gen_per_run)
        self.agents = []
        for i in range(n):
            self.agents.append(BHAgent(self, i, self.Rs[i]))
        self.init_agents()


# #### Add Group Reward (Consensus) to BH Model

# In[50]:


class CoopWeightAgent(BHAgent):
    "Strength of reinforcement dependent on success of group"
    "Only reinforce if there is individual success"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes)
        self.w = w # weight
    def reinforce(self, s):
        if self.belief:
            self.urn = self.urn + self.choice * self.w * s


# In[51]:


class CoopBonusAgent(BHAgent):
    "First reinforce based on individual results"
    "BONUS reinforcement dependent on success of group"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes)
        self.w = w # weight
    def reinforce(self, s):
        if self.belief:
            self.urn = self.urn + self.choice
        self.urn = self.urn + self.choice * self.w * s


# In[52]:


class CoopRewardModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, weight=1, v=1):
        super().__init__(n, round_per_gen, gen_per_run)
        self.w = weight
        self.agents = []
        if v == 1:
            for i in range(n):
                self.agents.append(CoopWeightAgent(self, i, self.Rs[i]))
        elif v == 2:
            for i in range(n):
                self.agents.append(CoopBonusAgent(self, i, self.Rs[i]))
        self.init_agents()
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls)
        for i in ls:
            a = self.agents[i]
            a.update()
        s = sum([a.belief for a in self.agents]) # Number of successful agents this round
        for a in self.agents:
            a.reinforce(s)


# #### Add Competition to BH Model

# In[53]:


class CompetitionModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, weight=1, v=1):
        super().__init__(n, round_per_gen, gen_per_run)
        self.w = weight
        self.agents = []
        if v == 1:
            for i in range(n):
                self.agents.append(CoopWeightAgent(self, i, self.Rs[i]))
        elif v == 2:
            for i in range(n):
                self.agents.append(CoopBonusAgent(self, i, self.Rs[i]))
        self.init_agents()
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls)
        for i in ls:
            a = self.agents[i]
            a.update()
        s = self.n - sum([a.belief for a in self.agents]) # Number of unsuccessful agents this round
        for a in self.agents:
            a.reinforce(s)


# #### Compare Models


# columns = ['reliability',
#            'matrix_BSM', 'success_BSM', 
#            'matrix_BH', 'success_BH', 
#            'matrix_CoopWeight_F', 'success_CoopWeight_F', 
#            'matrix_CompWeight_F', 'success_CompWeight_F'
#           ]
# df = pd.DataFrame(columns=columns)
# n = 5

# Rs = pd.read_csv('Rs.csv')
# Rs = Rs['reliability'].apply(ast.literal_eval)

# for i in range(len(Rs)):
#     models = [Model(n=n), 
#               BHModel(n=n), 
#               CoopRewardModel(n=n, weight=1, v=1)
#               CompetitionModel(n=n, weight=1, v=1)]
#     for m in models:
#         for a in m.agents:
#             a.r = Rs[i][a.id]
#     data = []
#     data.append(Rs[i])
#     for m in models:
#         m.run()
#         data.append([a.urn for a in m.agents])
#         data.append([a.n_success for a in m.agents])
#     df = pd.concat([df, pd.DataFrame([data], columns=columns)], ignore_index=True)

# df.to_csv('data.csv', index=False)






"""Test Baseline"""

columns = ['matrix_BH', 'success_BH', 
           'matrix_CoopWeight', 'success_CoopWeight', 
           'matrix_CompWeight', 'success_CompWeight']
df = pd.DataFrame(columns=columns)
n = 5

for i in range(200):
    models = [BHModel(n=n), 
              CoopRewardModel(n=n, weight=1, v=1),
              CompetitionModel(n=n, weight=1, v=1)]
    for m in models:
        for a in m.agents:
            a.r = 0.5
    data = []
    for m in models:
        m.run()
        data.append([a.urn for a in m.agents])
        data.append([a.n_success for a in m.agents])
    df = pd.concat([df, pd.DataFrame([data], columns=columns)], ignore_index=True)

df.to_csv('baseline.csv', index=False)

