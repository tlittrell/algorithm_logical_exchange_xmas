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
current_year = 2025  # Year for which assignments are being generated

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

The algorithm requires two data sources:

### 1. Secret Santa Database (`data/secret_santa_db.csv`)

A historical database containing gift assignments from all previous years. The algorithm queries this database for the previous year's data (based on `current_year - 1` from config) to prevent repeat assignments.

```csv
giver,gift1,gift2,year
Alice,Charlie,Diana,2024
Bob,Eve,Frank,2024
Charlie,Alice,Bob,2024
Diana,Frank,Eve,2024
Eve,Bob,Charlie,2024
Frank,Diana,Alice,2024
```

**Required columns:**

- `giver`: Person who gave gifts (must be unique per year)
- `gift1`: First gift recipient
- `gift2`: Second gift recipient
- `year`: Year of the assignment (integer)

**Note:** The database is automatically updated with new assignments after running the algorithm. You can re-run the algorithm for the same year, and it will replace the existing entries for that year.

### 2. Signups Database (`data/signups.csv`)

A historical database containing signup information for all years. The algorithm queries this database for the current year's participants (based on `current_year` from config) to determine who is participating.

```csv
person,is_secret_santa,is_stockings,year
Alice,true,true,2025
Bob,true,true,2025
Charlie,true,false,2025
Diana,false,true,2025
Eve,true,true,2025
Frank,true,false,2025
```

**Required columns:**

- `person`: Participant name
- `is_secret_santa`: Boolean indicating Secret Santa participation (`true`/`false`)
- `is_stockings`: Boolean indicating stockings participation (`true`/`false`)
- `year`: Year of signup (integer)

**Note:** Only participants with `is_secret_santa = true` for the current year will be included in the gift assignment algorithm. The `is_stockings` column is available for future features.

### 3. Gift Preferences Database (`data/gift_preferences.csv`)

A historical database containing gift preferences for all years. The algorithm queries this database for the current year's preferences to apply constraints on gift assignments.

```csv
person,gift,preference_type,year
Ariel,Mark,disallow,2025
Ariel,Thomas,disallow,2025
Ariel,Graham,disallow,2025
```

**Required columns:**

- `person`: Person who has the preference (the giver)
- `gift`: Person who would receive the gift (the recipient)
- `preference_type`: Type of preference (currently only `disallow` is supported)
- `year`: Year the preference applies to (integer)

**Note:** Only preferences with `preference_type = 'disallow'` for the current year will be applied. The preference type column is designed to support future enhancement with additional preference types (e.g., `prefer`, `require`). Each person-gift pair should only appear once per year.

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
2. Query the database for previous year's assignments (year = `current_year - 1`)
3. Read this year's signup data (participants with `is_secret_santa = true`)
4. Load this year's gift preferences (with `preference_type = 'disallow'`)
5. Build and solve the optimization problem with all constraints
6. Update the database with new assignments for `current_year`
7. Generate personalized messages in `data/output/`

**Re-running the algorithm:** If you need to regenerate assignments for the same year (e.g., if constraints changed), simply run the algorithm again. It will automatically replace the existing entries for `current_year` in the database.

## Output

The algorithm produces the following outputs:

### 1. Database Update (`data/secret_santa_db.csv`)

The Secret Santa database is automatically updated with new assignments for the current year. Each row includes the giver, recipients, and year:

```csv
giver,gift1,gift2,year
Alice,Charlie,Diana,2024
Bob,Eve,Frank,2024
...
Alice,Eve,Frank,2025
Bob,Charlie,Alice,2025
...
```

This database serves as both an output (for the current year) and an input (for future years), maintaining a complete historical record of all gift assignments.

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
