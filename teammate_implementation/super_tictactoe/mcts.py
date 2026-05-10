import copy
import math
import numpy as np
import torch


class MCTSNode:
    def __init__(self, prior: float = 1.0):
        self.prior = prior
        self.N = 0       # visit count
        self.W = 0.0     # total value
        self.children: dict = {}  # action -> MCTSNode

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_leaf(self) -> bool:
        return not self.children

    def ucb(self, parent_N: int, c_puct: float) -> float:
        return self.Q + c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)


class MCTS:
    """
    AlphaZero-style MCTS using the ActorCritic model for policy priors and
    leaf value estimates. Handles stochastic placement naturally via
    Monte Carlo re-simulation from root each run.
    """

    def __init__(self, model, device: str = 'cpu',
                 num_simulations: int = 100, c_puct: float = 1.5):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def _get_policy_value(self, env):
        state_t = torch.FloatTensor(env._get_state()).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, value = self.model(state_t, mask_t)
        return probs[0].cpu().numpy(), value[0].item()

    def _expand(self, node: MCTSNode, env) -> float:
        """Expand leaf: create child nodes with policy priors. Returns value estimate."""
        probs, value = self._get_policy_value(env)
        for action in np.where(env.get_action_mask())[0]:
            node.children[int(action)] = MCTSNode(prior=float(probs[action]))
        return value

    def run(self, env) -> np.ndarray:
        """Run simulations from current env state. Returns visit-count policy over 144 actions."""
        root = MCTSNode()
        root_env = copy.deepcopy(env)
        self._expand(root, root_env)

        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(root_env)
            path = [node]

            # Selection: follow UCB until reaching an unexpanded leaf or terminal.
            # Validate against sim_env's action mask — stochastic drift from earlier
            # moves can occupy a cell that was open when the node was first expanded.
            while not node.is_leaf() and not sim_env.done:
                action_mask = sim_env.get_action_mask()
                valid = {a: c for a, c in node.children.items() if action_mask[a]}
                if not valid:
                    break
                action = max(valid.keys(), key=lambda a: valid[a].ucb(node.N, self.c_puct))
                sim_env.step(action)
                node = node.children[action]
                path.append(node)

            # Evaluation
            if sim_env.done:
                # Value from the perspective of the player whose turn it now is
                # (they just lost — the previous player won)
                value = -1.0 if sim_env.winner is not None else 0.0
            else:
                value = self._expand(node, sim_env)

            # Backpropagation: negate value at each level (alternating players)
            for i, n in enumerate(reversed(path)):
                n.N += 1
                n.W += value * ((-1) ** i)

        visits = np.zeros(144)
        for action, child in root.children.items():
            visits[action] = child.N
        total = visits.sum()
        return visits / total if total > 0 else visits

    def get_action(self, env, temperature: float = 0.0) -> int:
        """Return best action. temperature=0 → greedy (strongest play)."""
        visit_probs = self.run(env)
        if temperature == 0.0:
            return int(np.argmax(visit_probs))
        probs = visit_probs ** (1.0 / temperature)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))
