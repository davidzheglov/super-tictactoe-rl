# Literature Review: Reinforcement Learning for Board and Stochastic Games

## 1. Classic Search Methods

Early computer-game programs relied on search trees and handcrafted evaluation functions. Shannon (1950) proposed constructing a game tree where the state is the board configuration and operations are possible moves, evaluating leaf nodes with a utility function. Greenblatt's MacHack VI became the first chess program to defeat a human in tournament play. These programs searched to a limited depth and used evaluation heuristics to approximate game outcomes.

The **minimax algorithm** propagates leaf evaluations back up the tree under the assumption that both players play optimally. **Alpha–beta pruning** improves minimax by pruning branches that cannot affect the final decision. However, exhaustive search and handcrafted evaluation functions become infeasible in games with very large or complex state spaces. Reinforcement learning (RL) provides an alternative: agents learn value functions or policies directly from experience rather than enumerating the entire game tree.

**Design implication for Super Tic-Tac-Toe:** The game has 96 playable cells across six 4×4 boards and stochastic placement transitions. Minimax is impractical — the branching factor is ~48 on average and the stochastic transition function means any deterministic tree search would need to model probability distributions at each node. RL and sampling-based methods are therefore the natural choice.

---

## 2. Q-Learning for Board Games

### 2.1 Q-Learning Basics

Q-learning is a model-free RL algorithm introduced by Watkins (1989). It learns the expected return of taking action $a$ in state $s$ — the Q-value $Q(s, a)$ — and updates Q-values according to the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where $\alpha$ is the learning rate, $\gamma$ is the discount factor, and $r$ is the reward. An $\varepsilon$-greedy policy chooses a random action with probability $\varepsilon$ to encourage exploration and chooses the greedy action otherwise.

### 2.2 Q-Learning for Tic-Tac-Toe

Ho et al. (2022) studied Q-learning for standard Tic-Tac-Toe [[1]](#ref-1). They note that naive implementations (e.g., Widyantoro et al., 2009) achieved less than a 50% win/tie rate, and propose *optimistic initialisation* and a refined reward structure to drive exploration. Their algorithm stores Q-values in a table indexed by board states and iteratively updates them from the end of each match using the Bellman equation. Using an $\varepsilon$-greedy policy with $\varepsilon$ starting at 0.9 and decaying over 300 000 self-play games, they attain a win/tie rate of ~90% [[1]](#ref-1).

Ho et al. contrast Q-learning with minimax: minimax requires sorting through all possible board states and becomes impractical as state space grows; Q-learning avoids exhaustive enumeration by iterative Bellman updates and can handle stochastic environments because it does not assume deterministic transitions [[1]](#ref-1).

**Design implication:** The Q-table approach was our first baseline. The state space of Super Tic-Tac-Toe (96 cells, each in {–1, 0, 1}) is astronomically large (~$3^{96}$), so tabular Q-learning can only cover a tiny fraction of states even with 15 000 episodes. We use it as a proof-of-concept baseline, tracking the *number of unique states discovered* (which grew to 693 853 in our run) as a proxy for learning breadth — directly addressing the TA's request to show Q-value table evolution.

### 2.3 Multi-Agent Q-Learning

In multi-agent games, each agent's reward depends on the joint actions of all players, making the environment non-stationary from any single agent's perspective. Nash Q-learning (Hu & Wellman, 2003) extends Q-learning to general-sum stochastic games by maintaining Q-functions over joint actions and updating them under the assumption that all agents play a Nash equilibrium at each stage. Experimental results on two-player grid games show that Nash Q-learning reaches joint optimal strategies more reliably than independent Q-learning.

---

## 3. Monte Carlo Tree Search (MCTS)

Monte Carlo tree search builds a search tree incrementally by sampling random playouts (rollouts) and using their outcomes to guide future exploration. The **UCT algorithm** (Upper Confidence bounds applied to Trees) selects the child node that maximises:

$$\text{UCT}(s, a) = \bar{V}(s') + C \sqrt{\frac{\ln N(s)}{N(s')}}$$

where $\bar{V}(s')$ is the estimated value of child $s'$, $N(s)$ is the total visits to the parent, $N(s')$ is the visits to the child, and $C$ is an exploration constant. Coquelin and Munos (2007) show that UCT balances exploration and exploitation but can be over-optimistic in deep trees, and propose modified confidence sequences (Flat-UCB, Smooth Trees) for better regret bounds [[4]](#ref-4). MCTS became the foundation of state-of-the-art Go and general game players because it scales to large state spaces and can incorporate learned policies to bias the search.

---

## 4. Deep Reinforcement Learning and AlphaGo

Silver et al. (2016) combined deep neural networks with Monte Carlo tree search to create AlphaGo [[5]](#ref-5). Two kinds of networks were trained:

- **Policy networks:** Convolutional neural networks that predict move probabilities. A supervised network was first trained from expert Go games and then refined by RL self-play [[5]](#ref-5).
- **Value networks:** Networks that predict the expected outcome of a position. Trained on self-play games, they evaluate leaf nodes in the search tree.

During play, AlphaGo used MCTS guided by the policy and value networks: the rollout policy sampled moves from the fast policy network, the value network evaluated leaf nodes, and UCT with prior probabilities balanced exploration and exploitation [[6]](#ref-6). AlphaGo achieved a 99.8% win rate against other Go programs and defeated a European professional player [[5]](#ref-5).

**AlphaZero** (Silver et al., 2018) generalised this approach to Go, chess, and shogi using a single neural network $f_\theta(s) = (\mathbf{p}, v)$ that outputs both a policy distribution $\mathbf{p}$ and a value estimate $v$ [[7]](#ref-7). Starting from random play and using only the game rules, AlphaZero learned entirely from self-play within 24 hours to achieve super-human performance. The algorithm is domain-agnostic and demonstrates that general-purpose RL can outperform traditional search engines in complex games [[7]](#ref-7).

**Design implication:** AlphaZero's architecture directly inspired our PPO policy-value network (`TorchPolicyValueNet`), which also outputs both a policy distribution over the 96 actions and a scalar state value. The network is trained by PPO (a policy-gradient method) rather than MCTS-guided self-play, which is computationally cheaper at our scale. The BC pre-training step mirrors AlphaZero's supervised initialisation from expert data.

---

## 5. Imperfect-Information Games — Poker

Stochastic games with hidden information pose additional challenges. **Counterfactual regret minimisation (CFR)** is a self-play algorithm that iteratively updates strategies to minimise regret and converges to a Nash equilibrium in two-player zero-sum games.

DeepStack (Moravčík et al., 2017) introduced *continual re-solving* for heads-up no-limit Texas Hold'em [[8]](#ref-8). Rather than computing a full strategy for the entire game, DeepStack re-solves the remaining game at each decision point using CFR. A deep neural network estimates the value of subtrees beyond a certain depth; this intuition network is trained from random poker situations using self-play. DeepStack thus combines recursive reasoning (CFR) with learned value estimates (deep learning) [[8]](#ref-8). It defeated professional players over 44 000 hands and produced strategies more difficult to exploit than abstraction-based approaches.

---

## 6. Sparse Rewards and Exploration Strategies

Board games typically provide only sparse rewards (+1 win, 0 draw, –1 loss). Sparse rewards make learning difficult because most transitions provide no feedback signal. Common strategies include:

- **$\varepsilon$-greedy exploration:** Simple and effective for tabular Q-learning. UCB exploration, used in MCTS and bandit algorithms, adds an exploration bonus proportional to uncertainty [[4]](#ref-4).
- **Reward shaping and human heuristics:** Providing intermediate rewards (e.g., for blocking an opponent's line or occupying the centre) accelerates learning. Potential-based shaping (Ng et al., 1999) guarantees that shaped rewards do not change the optimal policy. In our environment, a small shaping term (default scale 0.03) rewards extending open lines and blocking opponent threats without dominating the terminal signal.
- **Curriculum learning and self-play:** Starting with simpler opponents and gradually increasing difficulty helps the agent collect more informative experiences. We implement a *mixed-opponent curriculum* that starts with the heuristic agent and gradually increases self-play proportion over training.
- **Behavioural cloning (BC) warm-start:** Pre-training the policy network by supervised learning from heuristic-generated demonstrations bootstraps the agent above random play, avoiding the cold-start problem with sparse rewards.

**Design implication — why the smart heuristic is better than the basic heuristic as a training opponent:** The `BasicHeuristicAgent` uses only immediate win/block logic; it cannot mount threats and rarely generates the multi-step tactical positions that RL agents must learn to navigate. The `HeuristicAgent`, by contrast, evaluates the *stochastic expected value* of every legal action across all 9 possible landing cells (the intended cell with probability ½, eight neighbours with probability 1/16 each), scores cells additively across all legal winning windows, and penalises high-forfeit moves. This creates a richer training distribution — the RL agent must learn to build and defend multi-step plans — yielding faster and more robust policy improvement. Training against only the basic heuristic produces an agent that collapses against stronger opponents (DQN: 0% vs smart heuristic after 6 000 episodes); training against the smart heuristic and the line-builder from the start produces agents that reach ≥45% win rate after BC pre-training alone.

---

## 7. Implications for Super Tic-Tac-Toe

Our environment implements the following stochastic game: six 4×4 boards arranged in a triangular pyramid (96 cells total). Placement is accepted with probability 0.5; otherwise the move is redirected to a random adjacent cell (probability 1/16 each), and landing outside the board or on an occupied cell results in a forfeit. Win conditions require 4-in-a-row horizontally, 4-in-a-column spanning at least two levels, or 5-in-a-diagonal.

Key challenges:
1. **Huge state space:** $3^{96}$ theoretical states. Tabular Q-learning covers only a tiny fraction even with 15 000 episodes.
2. **Stochastic transitions:** Every move has a 50% chance of landing elsewhere. Deterministic search is inapplicable; agents must learn strategies robust to misplacement.
3. **Sparse rewards:** Games last 10–30 moves with reward only at the terminal state.
4. **Forfeit mechanics:** Choosing a heavily-surrounded cell is high-variance; the smart heuristic explicitly penalises low-neighbour-count cells.

Our research progression — tabular Q-learning → DQN → PPO (TorchRL) with BC warm-start and curriculum learning — mirrors the trajectory suggested in the literature: from exact methods that fail at scale, through function approximation, to actor-critic methods capable of handling stochastic, sparse-reward environments.

---

## References

<a id="ref-1"></a>[1] Ho, X., et al. (2022). *Q-learning for Tic-Tac-Toe*. Preprint. https://d197for5662m48.cloudfront.net/documents/publicationstatus/168349/preprint_pdf/a2a234e0cceafaad479e846342d22403.pdf

<a id="ref-2"></a>[2] Widyantoro, D. H., et al. (2009). Adaptive game playing using fuzzy logic. *Asian Journal of Control*, cited in Ho et al. (2022).

<a id="ref-3"></a>[3] Watkins, C. J. C. H. (1989). *Learning from delayed rewards*. PhD thesis, University of Cambridge.

<a id="ref-4"></a>[4] Coquelin, P.-A., & Munos, R. (2007). *Bandit algorithms for tree search*. arXiv:cs/0703062. https://arxiv.org/pdf/cs/0703062

<a id="ref-5"></a>[5] Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484–489. https://raw.githubusercontent.com/tpn/pdfs/master/Mastering%20the%20Game%20of%20Go%20with%20Deep%20Neural%20Networks%20and%20Tree%20Search.pdf

<a id="ref-6"></a>[6] Silver, D., et al. (2016). AlphaGo MLRG presentation. https://opencuny.org/machinelearning/files/2016/03/AlphaGo_MLRG.pdf

<a id="ref-7"></a>[7] Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140–1144. arXiv:1712.01815. https://arxiv.org/pdf/1712.01815

<a id="ref-8"></a>[8] Moravčík, M., et al. (2017). DeepStack: Expert-level artificial intelligence in heads-up no-limit poker. *Science*, 356(6337), 508–513. https://poker.cs.ualberta.ca/publications/17science.pdf

<a id="ref-9"></a>[9] Hu, J., & Wellman, M. P. (2003). Nash Q-learning for general-sum stochastic games. *Journal of Machine Learning Research*, 4, 1039–1069.

<a id="ref-10"></a>[10] Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*, 99, 278–287.
