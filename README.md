# Secret Santa Gift Assignment Optimizer

A constraint-based Secret Santa gift assignment system that uses integer programming to generate optimal gift assignments. The algorithm ensures fair and diverse gift exchanges while respecting relationship constraints, preventing repeats from previous years, and maintaining family dynamics.

## Features

- **Constraint-Based Optimization**: Uses CVXPY to solve an integer programming problem that maximizes gift assignment novelty
- **Multiple Constraint Types**:
  - No self-gifting
  - Equal distribution (everyone gives and receives the same number of gifts)
  - Couple constraints (partners can't gift each other or the same people)
  - Family limits (configurable limits on within-family gifting)
  - Historical prevention (no repeat gifts from last year)
  - Cycle prevention (if A->B, then B cannot->A)
  - Manual disallow lists (custom restrictions)
- **Reproducible Results**: Seeded random number generation ensures consistent outputs
- **Automated Output**: Generates CSV assignments and personalized Markdown messages with email addresses
- **Data Validation**: Comprehensive assertions ensure configuration and results meet all constraints

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd algorithm_logical_exchange_xmas
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

   Or using pip:

   ```bash
   pip install -e .
   ```

3. Install development dependencies (optional, for testing and linting):

   ```bash
   uv sync --dev
   ```

4. Set up pre-commit hooks (optional, but recommended):
   ```bash
   pre-commit install
   ```

## Configuration

Create a `local_config.toml` file in the project root with the following structure:

```toml
seed = 42  # Random seed for reproducibility

[algorithm]
eligible_people = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
gifts_per_person = 2  # Number of gifts each person gives and receives

# Define couples (they can't gift each other or the same people)
couples = [
    ["Alice", "Bob"],
    ["Charlie", "Diana"]
]

# Define families (all eligible people must be assigned to a family)
families = [
    ["Alice", "Bob", "Charlie"],  # Family 1
    ["Diana", "Eve", "Frank"]     # Family 2
]

# Family constraints
max_gifts_to_family = 1   # Max gifts a person can give to their own family
max_gifts_from_family = 1 # Max gifts a person can receive from their own family
max_couple_overlap = 0    # Max people both members of a couple can gift

# Message template for gift assignments
message = """Hi {giver}!

You're giving gifts to:
1. {gift1}
2. {gift2}

Happy Secret Santa!"""

[emails]
# Email addresses for each participant
Alice = "alice@example.com"
Bob = "bob@example.com"
Charlie = "charlie@example.com"
Diana = "diana@example.com"
Eve = "eve@example.com"
Frank = "frank@example.com"

[manual_disallows]
# Optional: Manually prevent specific gift assignments
# Alice = ["Eve"]  # Alice cannot gift to Eve
```

## Input Data

The algorithm requires two CSV files in the `data/input/` directory:

### 1. Last Year's Gifts (`data/input/ly_gifts.csv`)

Records gift assignments from the previous year to prevent repeats.

```csv
giver,gift1,gift2
Alice,Charlie,Diana
Bob,Eve,Frank
Charlie,Alice,Bob
Diana,Frank,Eve
Eve,Bob,Charlie
Frank,Diana,Alice
```

**Required columns:**

- `giver`: Person who gave gifts (must be unique)
- `gift1`: First gift recipient
- `gift2`: Second gift recipient

### 2. This Year's Signups (`data/input/ty_signup.csv`)

Indicates who is participating this year.

```csv
person,is_secret_santa
Alice,true
Bob,true
Charlie,true
Diana,false
Eve,true
Frank,true
```

**Required columns:**

- `person`: Participant name
- `is_secret_santa`: Boolean indicating participation (`true`/`false`)

## Usage

Run the Secret Santa assignment algorithm:

```bash
python src/algorithm_logical_exchange_xmas/run.py
```

Or using the module:

```bash
python -m algorithm_logical_exchange_xmas.run
```

The algorithm will:

1. Load and validate configuration from `local_config.toml`
2. Read input data files
3. Build and solve the optimization problem
4. Generate output files in `data/output/`

## Output

The algorithm generates two files in `data/output/`:

### 1. Assignments CSV (`data/output/assignments.csv`)

A structured file with all gift assignments:

```csv
giver,gift1,gift2,gift1_ly,gift2_ly
Alice,Eve,Frank,Charlie,Diana
Bob,Charlie,Alice,Eve,Frank
Charlie,Frank,Diana,Alice,Bob
...
```

### 2. Secret Santa Messages (`data/output/secret_santa_messages.md`)

Personalized Markdown messages for each participant with their assignments and email addresses, ready to be sent:

```markdown
# Alice

email: alice@example.com

Hi Alice!

You're giving gifts to:

1. Eve
2. Frank

Happy Secret Santa!

# Bob

email: bob@example.com

...
```

## How It Works

The algorithm uses **Constraint Programming** (specifically, integer programming) to find an optimal gift assignment that satisfies all constraints while maximizing "novelty" (a random matrix that ensures diverse assignments).

### Mathematical Formulation

**Decision Variable:**

- `X[i,j]` in {0,1}: Binary variable where 1 means person `i` gives to person `j`

**Objective:**

- Maximize novelty: `sum(X[i,j] * novelty[i,j])` where `novelty` is a random matrix

**Constraints:**

1. **No self-gifting**: `X[i,i] = 0` for all `i`
2. **Equal distribution**: Each person gives/receives exactly `gifts_per_person` gifts
3. **Historical**: If person `i` gifted person `j` last year, then `X[i,j] = 0`
4. **Couples**:
   - `X[i,j] = 0` if `i` and `j` are partners
   - Couples can't overlap gifts beyond `max_couple_overlap`
5. **Family limits**: Each person gives/receives at most `max_gifts_to_family`/`max_gifts_from_family` within their family
6. **Cycle prevention**: `X[i,j] + X[j,i] <= 1` (if A->B, then B cannot->A)
7. **Manual disallows**: `X[i,j] = 0` for manually specified pairs

### Solver

The problem is solved using [CVXPY](https://www.cvxpy.org/), a Python library for convex optimization, with a mixed-integer programming solver.

## Development

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src/algorithm_logical_exchange_xmas --cov-report=html
```

### Code Quality

This project uses comprehensive code quality tools:

- **Linting & Formatting**: Ruff with extensive rule sets
- **Type Checking**: MyPy with strict type hints
- **Security**: Bandit for security issue detection
- **Secret Detection**: detect-secrets to prevent credential leaks
- **Pre-commit Hooks**: Automated checks before each commit

Run linting:

```bash
ruff check .
```

Run formatting:

```bash
ruff format .
```

Run type checking:

```bash
mypy src
```

### Project Structure

```
algorithm_logical_exchange_xmas/
├── src/
│   └── algorithm_logical_exchange_xmas/
│       ├── __init__.py
│       └── run.py              # Main algorithm implementation
├── tests/                      # Comprehensive test suite
│   ├── conftest.py            # Test fixtures
│   ├── test_config.py         # Configuration validation tests
│   ├── test_constraints.py    # Constraint creation tests
│   ├── test_data_loading.py   # Data loading tests
│   ├── test_optimization.py   # Optimization solver tests
│   └── test_output.py         # Output generation tests
├── data/
│   ├── input/                 # Input CSV files
│   └── output/                # Generated assignments and messages
├── local_config.toml          # Configuration file (create this)
├── pyproject.toml             # Project configuration
└── .pre-commit-config.yaml    # Pre-commit hooks configuration
```

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code follows style guidelines: `ruff check . && ruff format .`
3. Type hints are correct: `mypy src`
4. Pre-commit hooks pass: `pre-commit run --all-files`

## License

[Add your license here]

## Acknowledgments

Built with:

- [CVXPY](https://www.cvxpy.org/) - Convex optimization library
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [DuckDB](https://duckdb.org/) - SQL query engine
- [Ruff](https://github.com/astral-sh/ruff) - Fast Python linter

---

**Note**: This is a fun project for optimizing Secret Santa gift exchanges. The algorithm ensures fairness and novelty while respecting real-world relationship constraints. Adjust the configuration parameters to match your group's preferences!
