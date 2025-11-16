"""Secret Santa gift assignment optimization package.

This package provides a constraint-based integer programming solution for
generating optimal Secret Santa gift assignments. It uses CVXPY to maximize
assignment novelty while respecting multiple real-world constraints including
couple relationships, family dynamics, historical assignments, and custom
restrictions.

The main algorithm is implemented in the `run` module. Use it to:
- Load configuration from TOML files
- Process participant and historical data
- Solve optimization problems with multiple constraints
- Generate gift assignments and personalized messages

Example:
    Run the Secret Santa algorithm::

        $ uv run python -m algorithm_logical_exchange_xmas.run

Modules:
    run: Main algorithm implementation with optimization and output generation.
"""
