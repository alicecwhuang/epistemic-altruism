import random
import numpy as np
import pandas as pd
import scipy.stats
import ast


# ### First Phase (5 agents, 3 balls fixed)

# #### Perfect communication model in BSM

class Agent():
    def __init__(self, model, id, reliability):
        self.model = model
        self.id = id
        self.r = reliability
        self.peers = [] # list of all scientists
        self.urn = np.array([]) # probability [nature, agent 1, agent 2,...]
        self.belief = 0 # current belief
        self.choice = None
        self.n_success = 0 # accumulated number of successes
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


# In[13]:


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



class BHModel(Model):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        super().__init__(n, round_per_gen, gen_per_run)
        self.agents = []
        for i in range(n):
            self.agents.append(BHAgent(self, i, self.Rs[i]))
        self.init_agents()


# ### Fine-grained reward

# In[24]:


class fgBHAgent(BHAgent):
    def __init__(self, model, id, reliability, n_votes=3):
        super().__init__(model, id, reliability)
        self.n_votes = n_votes
    def reinforce(self):
        """Only reinforce for agents that gave the correct answer this round""" 
        """(Instead of everyone asked this round)"""
        if self.belief:
            self.urn = self.urn + self.results


# In[25]:


class fgBHModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        super().__init__(n, round_per_gen, gen_per_run)
        self.agents = []
        for i in range(n):
            self.agents.append(fgBHAgent(self, i, self.Rs[i]))
        self.init_agents()


class TwoUrnAgent(fgBHAgent):
    def __init__(self, model, id, reliability, payoff=3, cost=0.5, n_votes=None):
        super().__init__(model, id, reliability)
        self.Q_urn = np.array([]) # First urn determines how many consultations
        self.p = payoff
        self.c = cost
    def choose_quantity(self):
        """randomly draw one ball from the quantity urn to determine how many balls to draw from second urn"""
        prob = self.Q_urn/sum(self.Q_urn)
        self.n_votes = np.random.choice(len(self.Q_urn), size=1, p=prob)[0] + 1
    def choose(self):
        """choose who to consult (no repeat)"""
        choice = np.array([0]*len(self.urn))
        prob = self.urn/sum(self.urn) # Prob proportionate to num of balls
        choice[np.random.choice(range(len(self.urn)), size=self.n_votes, replace=False, p=prob)] = 1
        return choice
    def reinforce_Q(self):
        current = self.Q_urn[self.n_votes - 1]
        adjusted = current + (self.belief * self.p - self.n_votes * self.c)
        self.Q_urn[self.n_votes - 1] = max(adjusted, 1)


# In[200]:


class TwoUrnModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, payoff=3, cost=0.5, min_R=0, max_R=1):
        super().__init__(n, round_per_gen, gen_per_run)
        self.Rs = np.round(np.random.uniform(min_R, max_R, size=n), 2)
        self.agents = []
        self.p = payoff
        self.c = cost
        for i in range(n):
            self.agents.append(TwoUrnAgent(self, i, self.Rs[i], payoff, cost))
        self.init_agents()
    def init_agents(self):
        """update agent parameters"""
        for a in self.agents:
            a.peers = self.agents
            a.urn = np.array([1] * (self.n + 1))
            a.urn[a.id+1] = 0
            a.Q_urn = np.array([1] * (self.n))
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls) # Agents update in random order
        for i in ls:
            a = self.agents[i]
            a.choose_quantity()
            a.update()
            a.reinforce_Q()
            a.reinforce()


df = pd.DataFrame(columns=['reliability', 'matrix', 'Q_matrix', 'success'])
for i in range(1):
    m = TwoUrnModel(n=5, payoff=3, cost=0.2, min_R=0, max_R=1)
    m.run()
    df.loc[i] = [m.Rs, [a.urn for a in m.agents], [a.n_success for a in m.agents], [a.Q_urn for a in m.agents]]



