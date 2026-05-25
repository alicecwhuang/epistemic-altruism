import random
import numpy as np
import pandas as pd
import scipy.stats
import ast
import concurrent.futures
import sys
import os
import itertools


# ### First Phase (5 agents, 3 balls fixed)

# #### Perfect communication model in BSM

class Agent():
    """Perfect communication model from Barrett, Skyrms, Mohseni"""
    def __init__(self, model, id, reliability):
        self.model = model
        self.id = id
        self.r = reliability
        self.peers = [] # list of all scientists
        self.urn = np.array([]) # probability [nature, agent 1, agent 2,...]
        self.belief = 0 # current belief
        self.choice = None
        self.n_success = 0 # accumulated number of successes
        self.end_success = 0 # end of run success once networks have stabilized 
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
        if self.model.final:
            self.end_success += self.belief
    def reinforce(self):
        if self.belief:
            self.urn[self.choice] += 1

class Model():
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, last_n_round=1000):
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
        self.final = False # Record end of run success rate
        self.switch = np.floor(gen_per_run - last_n_round / round_per_gen)
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
            if i == self.switch: # start recording end of run success
                self.final = True
            for a in self.agents:
                a.belief = a.experiment() # Initiate agent beliefs by consulting nature
            for j in range(self.round_per_gen):
                self.play()

# #### BH Model (My Version)


class BHAgent(Agent):
    """Binary model from Bruner and Holman"""
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
        if self.model.final:
            self.end_success += self.belief
    def reinforce(self):
        if self.belief:
            self.urn = self.urn + self.choice

def majority_vote(p):
    """compute the probability of success if all agents only consult nature and take a majority vote"""
    n = len(p)
    # Create a DP table where dp[i][j] is the probability of getting j heads with the first i coins
    dp = np.zeros((n + 1, n + 1))
    dp[0][0] = 1  # Base case: 0 coins, 0 heads has probability 1
    for i in range(1, n + 1):
        for j in range(i + 1):
            # If the i-th coin is tails
            dp[i][j] += dp[i - 1][j] * (1 - p[i - 1])
            # If the i-th coin is heads
            if j > 0:
                dp[i][j] += dp[i - 1][j - 1] * p[i - 1]
    # Calculate the total probability of getting more than half heads
    probability = 0
    for j in range(n + 1):
        if j == n / 2:
            probability += dp[n][j] / 2 
        elif j > (n / 2):
            probability += dp[n][j]
    return probability
    
def optimal(p):
    result = []
    for i in range(len(p)):
        result.append(majority_vote(sorted(p, reverse=True)[:i+1]))
    idx = result.index(max(result))
    return idx + 1, max(result)

class BHModel(Model):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        super().__init__(n, round_per_gen, gen_per_run)
        o = optimal(self.Rs)
        self.optimal_n = o[0]
        self.optimal_success = o[1]
        self.agents = []
        for i in range(n):
            self.agents.append(BHAgent(self, i, self.Rs[i]))
        self.init_agents()
    


class WeightAgent(BHAgent):
    "Strength of reinforcement dependent on success of group"
    "Only reinforce if there is individual success"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes)
        self.w = w # weight
    def reinforce(self, s):
        if self.belief:
            self.urn = self.urn + self.choice * self.w * s

class BonusAgent(BHAgent):
    "First reinforce based on individual results"
    "BONUS reinforcement dependent on success of group"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes)
        self.w = w # weight
    def reinforce(self, s):
        if self.belief:
            self.urn = self.urn + self.choice
        self.urn = self.urn + self.choice * self.w * s # group reinforcement regardless of belief

class CooperationModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, n_votes=3, w=1, v=1):
        super().__init__(n, round_per_gen, gen_per_run)
        self.w = w
        self.agents = []
        if v == 1:
            for i in range(n):
                self.agents.append(WeightAgent(self, i, self.Rs[i], n_votes, w))
        elif v == 2:
            for i in range(n):
                self.agents.append(BonusAgent(self, i, self.Rs[i], n_votes, w))
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

class CompetitionModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, n_votes=3, w=1, v=1):
        super().__init__(n, round_per_gen, gen_per_run)
        self.w = w
        self.agents = []
        if v == 1:
            for i in range(n):
                self.agents.append(WeightAgent(self, i, self.Rs[i], n_votes, w))
        elif v == 2:
            for i in range(n):
                self.agents.append(BonusAgent(self, i, self.Rs[i], n_votes, w))
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


class fgBHAgent(BHAgent):
    def __init__(self, model, id, reliability, n_votes=3):
        super().__init__(model, id, reliability)
        self.n_votes = n_votes
    def reinforce(self):
        """Only reinforce for agents that gave the correct answer this round""" 
        """(Instead of everyone asked this round)"""
        if self.belief:
            self.urn = self.urn + self.results   

class fgBHModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200):
        """n: total number of agents"""
        super().__init__(n, round_per_gen, gen_per_run)
        self.agents = []
        for i in range(n):
            self.agents.append(fgBHAgent(self, i, self.Rs[i]))
        self.init_agents()


class fgWeightAgent(WeightAgent):
    "Strength of reinforcement dependent on success of group"
    "Only reinforce if there is individual success"
    def __init__(self, model, id, reliability, n_votes=3, w=1):
        super().__init__(model, id, reliability, n_votes, w)
    def reinforce(self, s):
        """Only reinforce for agents that gave the correct answer this round""" 
        """(Instead of everyone asked this round)"""
        """s: the number of other agents who also succeeded (cooperative) / failed (competitive)"""
        if self.belief:
            self.urn = self.urn + self.results * self.w * s


class fgCooperationModel(CooperationModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, w=1):
        super().__init__(n, round_per_gen, gen_per_run, w)
        self.agents = []
        for i in range(n):
            self.agents.append(fgWeightAgent(self, i, self.Rs[i]))
        self.init_agents()

class fgCompetitionModel(fgCooperationModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, w=1):
        super().__init__(n, round_per_gen, gen_per_run, w)
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls)
        for i in ls:
            a = self.agents[i]
            a.update()
        s = self.n - sum([a.belief for a in self.agents]) # Number of unsuccessful agents this round
        for a in self.agents:
            a.reinforce(s)        

class TwoUrnAgent(fgWeightAgent):
    def __init__(self, model, id, reliability, n_votes=None, w=1, payoff=3, cost=0.5):
        super().__init__(model, id, reliability, n_votes, w)
        self.Q_urn = np.array([], dtype=float) # First urn determines how many consultations
        self.p = payoff # Payoff needs to be larger than (cost * number of agents)
        self.c = cost
    def choose_quantity(self):
        """randomly draw one ball from the quantity urn to determine how many balls to draw from second urn"""
        prob = self.Q_urn/sum(self.Q_urn)
        self.n_votes = np.random.choice(len(self.Q_urn), size=1, p=prob)[0] + 1
    def choose(self):
        """choose who to consult (no repeat!)"""
        choice = np.array([0]*len(self.urn))
        prob = self.urn/sum(self.urn) # Prob proportional to num of balls
        choice[np.random.choice(range(len(self.urn)), size=self.n_votes, replace=False, p=prob)] = 1
        return choice
    def reinforce_Q(self):
        strength = self.p - self.n_votes * self.c
        if (self.belief) & (strength > 0): # Make sure no punishment
            self.Q_urn[self.n_votes - 1] += (strength)

class TwoUrnModel(BHModel):
    def __init__(self, n=5, round_per_gen=100, gen_per_run=200, n_votes=None, w=1, payoff=3, cost=0.5, R_range=[0, 1]):
        super().__init__(n, round_per_gen, gen_per_run)
        self.Rs = np.round(np.random.uniform(R_range[0], R_range[1], size=n), 2)
        o = optimal(self.Rs)
        self.optimal_n = o[0]
        self.optimal_success = o[1]
        self.agents = []
        self.p = payoff
        self.c = cost
        for i in range(n):
            self.agents.append(TwoUrnAgent(self, i, self.Rs[i], n_votes, w, payoff, cost))
        self.init_agents()
    def init_agents(self):
        """update agent parameters"""
        for a in self.agents:
            a.peers = self.agents
            a.urn = np.array([1] * (self.n + 1), dtype=float)
            a.urn[a.id+1] = 0
            a.Q_urn = np.array([1] * (self.n), dtype=float)
    def play(self):
        ls = list(range(self.n))
        random.shuffle(ls) # Agents update in random order
        for i in ls:
            """All agents update beliefs first"""
            a = self.agents[i]
            a.choose_quantity()
            a.update()
        s = self.n - sum([a.belief for a in self.agents]) # Number of unsuccessful agents this round
        for a in self.agents:
            """All agents reinforce with competition"""
            a.reinforce_Q()
            a.reinforce(s)


def simulation(params):
    np.random.seed()
    n, payoff, cost, R_range, game = params
    m = TwoUrnModel(n=n, payoff=payoff, cost=cost, R_range=R_range, round_per_gen=game[0], gen_per_run=game[1])
    m.run()
    return {
        'game': game,
        'n': n,
        'payoff': payoff,
        'cost': cost,
        'R_range': R_range,
        'reliability': m.Rs,
        'optimal_n': m.optimal_n,
        'optimal_success': m.optimal_success,
        'matrix': [a.urn for a in m.agents],
        'Q_matrix': [np.round(a.Q_urn) for a in m.agents],
        'success': [a.n_success for a in m.agents],
        'end_success': [a.end_success for a in m.agents]
    }

if __name__ == "__main__":
    # Parameter lists
    n_values = [5, 10, 15]
    payoff_values = [1, 2, 3]
    cost_values = [0, 0.1, 0.2]
    R_range_values = [[0.7, 0.7], [0.2, 0.8]]
    game_values = [[1, 2000000], [100, 20000], [1000, 2000]]

    # Generate all combinations
    combinations = list(itertools.product(n_values, payoff_values, cost_values, R_range_values, game_values))

    # Get SLURM task ID
    task_id = int(sys.argv[1])
    params = combinations[task_id]

    os.makedirs("results", exist_ok=True)

    # Run simulations in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulation, params) for _ in range(200)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Save DataFrame
    df = pd.DataFrame(results)
    filename = f"results/result_{task_id}.csv"
    df.to_csv(filename, index=False)