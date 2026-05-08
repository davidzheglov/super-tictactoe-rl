# Heuristics and Reward Shaping

This project now contains three rule-based agents. They are deliberately
separate because they answer different research questions.

## Smart Heuristic

`HeuristicAgent` is the main human-knowledge opponent. It does not score only
the clicked cell; it scores the expected value of the stochastic move:

- the intended cell lands with probability `1/2`;
- each of the eight adjacent cells lands with probability `1/16`;
- occupied or out-of-board redirected cells become forfeits.

For every possible landing cell, it enumerates all legal winning windows:

- 4-cell horizontal windows;
- 4-cell vertical windows that span at least two levels;
- 5-cell diagonals in both directions.

The cell score is additive across every window containing that cell. This is the
important part for intersection tactics: if a cell blocks an opponent's
horizontal two-in-a-row and also blocks a vertical or diagonal construction, it
receives both scores. The agent therefore prefers the shared intersection over
a plain endpoint when the board geometry supports it.

Priority order emerges from the weights:

- immediate own wins are highest;
- immediate opponent winning threats are urgent blocks;
- opponent 2-in-horizontal, 2-in-vertical, and 3-in-diagonal windows are treated
  as dangerous developing threats;
- own open lines are extended when there is no strong defensive emergency;
- high-forfeit actions are penalized, so safer cells with more empty neighbours
  are preferred.

## Line Builder

`LineBuilderAgent` uses the same stochastic landing model but changes the
weights. It heavily values its own longest open line and gives only a small
defensive weight to opponent threats. This is the "relentless builder" baseline:
useful because a trained RL agent should beat both a defensive human-like
heuristic and an aggressive line-construction heuristic.

## Basic Heuristic

`BasicHeuristicAgent` preserves the older weak baseline: win now, block an
immediate win, otherwise prefer central/corner-ish cells. It is kept for
ablation, not as the serious opponent.

## Sparse Reward Handling

The environment still gives the true terminal reward: win, loss, draw, or
illegal-move loss. On top of that, training can add a small dense shaping term:

```text
shaped_reward =
    environment_reward
  + shaping_scale * clip(potential_after - potential_before)
  - forfeit_penalty if the agent forfeited
```

The potential is:

```text
own open-line potential - defense_weight * opponent open-line potential
```

So the learner gets weak feedback for:

- extending a viable own row, spanning column, or diagonal;
- reducing opponent threats by blocking;
- avoiding stochastic forfeits.

This follows the same research logic as the reference project: use potential
growth to make sparse terminal rewards easier to learn from, but keep terminal
win/loss rewards as the objective. The shaping coefficients are intentionally
small (`0.03` by default) so the agent does not learn to optimize the heuristic
instead of the game.
