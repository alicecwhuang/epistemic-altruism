import random
import numpy as np
import pandas as pd
import scipy.stats
import ast
from concurrent.futures import ProcessPoolExecutor


# ### First Phase (5 agents, 3 balls fixed)

# #### Perfect communication model in BSM

# In[2]:


class Agent():
    def __init__(self, model, id, reliability):
        self.model = model
        self.id = id
        self.r = reliability
        self.peers = [] # list of all scientists
        self.urn = np.array([]) # probability [nature, agent 1, agent 2,...]
        self.belief = 0 # current belief
        self.choice = None
        self.n_success = 0 # accumulted number of successes
    def choose(self):
        """randomly draw one ball"""
        prob = self.urn/sum(self.urn)
        return np.random.choice(len(self.urn), size=1, p=prob)[0]
    def experiment(self):
        """consult nature, chance of success = reliability"""
        return 1 if random.random() < self.r else 0
    def update(self):
        """update belief based on choice"""
        self.choice = int(self.choose())
        if self.choice == 0:
            self.belief = self.experiment()
        else:
            self.belief = self.peers[self.choice - 1].belief
        self.n_success += self.belief
    def reinforce(self):
        if self.belief:
            self.urn[self.choice] += 1


# In[3]:


class Model():
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        self.n = n
        self.round_per_gen = round_per_gen
        self.gen_per_run = gen_per_run
        self.agents = [] # list of agents
        self.Rs = [round(random.random(), 2) for i in range(n)] # varying reliability
        """create agents"""
        for i in range(n):
            self.agents.append(Agent(self, i, self.Rs[i]))
        self.init_agents()
    def init_agents(self):
        """update agent parameters"""
        for a in self.agents:
            a.peers = self.agents
            a.urn = np.array([1] * (self.n + 1))
            a.urn[a.id+1] = 0
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls) # Agents update in random order
        for i in ls:
            a = self.agents[i]
            a.update()
            a.reinforce()
    def run(self):
        for i in range(self.gen_per_run):
            for a in self.agents:
                a.belief = a.experiment() # Initiate agent beliefs by consulting nature
            for i in range(self.round_per_gen):
                self.play()

# #### BH Model (My Version)

# In[63]:


class BHAgent(Agent):
    def __init__(self, model, id, reliability, n_votes=3):
        super().__init__(model, id, reliability)
        self.n_votes = n_votes
        self.results = None
    def choose(self):
        choice = np.array([0]*len(self.urn)) # Array recording balls drawn
        for i in range(self.n_votes):
            prob = self.urn/sum(self.urn) # Prob proportionate to num of balls
            pick = np.random.choice(range(len(self.urn)), p=prob)
            choice[pick] += 1
            self.urn[pick] -= 1
        self.urn = self.urn + choice # Return balls back to urn
        return choice
    def update(self):
        """update belief by majority vote"""
        self.choice = self.choose()
        ex_results = np.array([sum([self.experiment() for i in range(self.choice[0])])])
        peer_results = self.choice[1:] * np.array([a.belief for a in self.model.agents])
        self.results = np.concatenate((ex_results, peer_results), axis=0)
        votes = sum(self.results)
        if votes > (sum(self.choice) * 0.5):
            self.belief = 1
        elif votes < (sum(self.choice) * 0.5):
            self.belief = 0
        else:
            self.belief = random.choice([0, 1])
        self.n_success += self.belief
    def reinforce(self):
        if self.belief:
            self.urn = self.urn + self.choice


# In[64]:


class BHModel(Model):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        super().__init__(n, round_per_gen, gen_per_run)
        self.agents = []
        for i in range(n):
            self.agents.append(BHAgent(self, i, self.Rs[i]))
        self.init_agents()


# #### Add Group Reward (Consensus) to BH Model

# In[24]:


class CoopWeightAgent(BHAgent):
    "Strength of reinforcement dependent on success of group"
    "Only reinforce if there is individual success"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes)
        self.w = w # weight
    def reinforce(self, s):
        if self.belief:
            self.urn = self.urn + self.choice * self.w * s


# In[25]:


class CoopBonusAgent(BHAgent):
    "First reinforce based on individual results"
    "BONUS reinforcement dependent on success of group"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes)
        self.w = w # weight
    def reinforce(self, s):
        if self.belief:
            self.urn = self.urn + self.choice
        self.urn = self.urn + self.choice * self.w * s # group reinforcement regardless of belief


# In[26]:


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

# In[27]:


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


# ### Fine-grained reward

# In[75]:


class fgBHAgent(BHAgent):
    def __init__(self, model, id, reliability, n_votes=3):
        super().__init__(model, id, reliability)
        self.n_votes = n_votes
    def reinforce(self):
        if self.belief:
            self.urn = self.urn + self.results

# In[76]:


class fgBHModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        super().__init__(n, round_per_gen, gen_per_run)
        self.agents = []
        for i in range(n):
            self.agents.append(fgBHAgent(self, i, self.Rs[i]))
        self.init_agents()


# In[77]:


class fgCoopWeightAgent(CoopWeightAgent):
    "Strength of reinforcement dependent on success of group"
    "Only reinforce if there is individual success"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes, w)
    def reinforce(self, s):
        if self.belief:
            self.urn = self.urn + self.results * self.w * s


# In[81]:


class fgCoopRewardModel(CoopRewardModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, weight=1):
        super().__init__(n, round_per_gen, gen_per_run, weight)
        self.agents = []
        for i in range(n):
            self.agents.append(fgCoopWeightAgent(self, i, self.Rs[i]))
        self.init_agents()


# In[82]:


class fgCompetitionModel(fgCoopRewardModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, weight=1):
        super().__init__(n, round_per_gen, gen_per_run, weight)
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls)
        for i in ls:
            a = self.agents[i]
            a.update()
        s = self.n - sum([a.belief for a in self.agents]) # Number of unsuccessful agents this round
        for a in self.agents:
            a.reinforce(s)


# #### Test fine-grained reward models

# In[83]:


columns = ['reliability', 
           'matrix_fgBH', 'success_fgBH', 
           'matrix_fgCoopWeight_F', 'success_fgCoopWeight_F',
           'matrix_fgCompWeight_F', 'success_fgCompWeight_F'
          ]
df = pd.DataFrame(columns=columns)
n = 5

Rs = pd.read_csv('Rs.csv')
Rs = Rs['reliability'].apply(ast.literal_eval)


def run_simulation(i):
    models = [fgBHModel(n=n),
              fgCoopRewardModel(n=n, weight=1), 
              fgCompetitionModel(n=n, weight=1)]
    
    for m in models:
        for a in m.agents:
            a.r = Rs[i][a.id]
    
    data = []
    data.append(Rs[i])
    for m in models:
        m.run()
        data.append([a.urn for a in m.agents])
        data.append([a.n_success for a in m.agents])
    
    return data

# Run in parallel
with ProcessPoolExecutor() as executor:
    Results = list(executor.map(run_simulation, range(10)))

# Build final DataFrame
df = pd.DataFrame(Results, columns=columns)

df.to_csv('test.csv', index=False)


