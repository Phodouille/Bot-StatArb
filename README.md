# StatArb Bot

PCA + mean-reversion statistical arbitrage bot on 40-50 liquid US tech
stocks, with an adaptive PCA window and VIX/credit-spread macro filters as
differentiating features over a classic stat arb baseline. Paper trading on
Alpaca (cash account); 4 configurable benchmark levels in one codebase.

Full architecture, contracts between modules, and global rules live in
[`CLAUDE.md`](./CLAUDE.md) — read that before writing code in any module.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your Alpaca paper keys
```

## Working with Claude Code

Four project subagents are defined in `.claude/agents/`: `data-ingestion`,
`signals`, `backtest`, `risk`. Each is scoped to its own folder and reads
`CLAUDE.md` for the shared contracts before doing anything. Run `/agents`
inside Claude Code at the repo root to confirm they're picked up.

Each agent uses `isolation: worktree`, so they can be run in parallel
safely from a single Claude Code session without stepping on each other's
files — e.g.:

```
Use the data-ingestion, signals, backtest, and risk subagents in parallel
to scaffold their respective modules per CLAUDE.md.
```

## Repo structure

See the "Structure du repo" section in `CLAUDE.md` for which module maps to
which team member.
