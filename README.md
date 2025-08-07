# 🎰 Advanced Multi-Armed Bandit Algorithms

## 🏗️ Implementation Details

### Project Structure

```
bandit_algos/
├── agents/                  # Bandit algorithm implementations
│   ├── __init__.py
│   ├── base_agent.py        # Abstract base class
│   ├── epsilon.py           # ε-Greedy variants
│   ├── ucb.py               # UCB variants
│   ├── thompson.py          # Thompson Sampling
│   ├── gradient_bandit.py   # Gradient-based methods
│   └── llm_agent.py         # LLM-powered bandit agent
├── environments/            # Bandit environments
│   ├── __init__.py
│   ├── base_environment.py  # Abstract base class
│   ├── bernoulli_bandit.py  # Binary rewards
│   └── gaussian_bandit.py   # Continuous rewards
├── configs/                 # Configuration files
│   ├── agent/               # Agent configurations
│   ├── env/                 # Environment configurations
│   └── experiment/          # Experiment configurations
├── utils/                   # Utility functions
│   ├── confidence.py        # Confidence interval calculations
│   └── seeder.py            # Random seeding utilities
├── plots/                   # Visualization tools
│   └── plot_utils.py        # Plotting functions
├── tests/                   # Test suite
├── main.py                  # Main entry point
└── README.md                # This file
```

## 🚀 Installation

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bandit-algorithms.git
   cd bandit-algorithms
   ```

2. Create and activate a virtual environment (recommended):
   ```
    python3.12 -m venv ~/venv
    source ~/venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install vllm==0.9.2
   pip install --upgrade wandb
   pip install transformers==4.53.2
   ```

## 🛠️ Usage

### Basic Example

```python
from environments.bernoulli_bandit import BernoulliBandit
from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent

# Create environment with 5 arms
env = BernoulliBandit(n_arms=5, probs=[0.1, 0.2, 0.3, 0.4, 0.5])

# Initialize agents
agents = [
    EpsilonGreedyAgent(n_actions=5, epsilon=0.1, name="ε-Greedy (ε=0.1)"),
    UCBAgent(n_actions=5, name="UCB1"),
    ThompsonSamplingAgent(n_actions=5, name="Thompson Sampling")
]

# Run simulation
n_steps = 1000
n_trials = 100

for agent in agents:
    total_reward = 0
    for _ in range(n_trials):
        env.reset()
        agent.reset()
        for t in range(n_steps):
            action = agent.get_action()
            reward = env.pull(action)
            agent.update(action, reward)
            total_reward += reward
    
    avg_reward = total_reward / (n_steps * n_trials)
    print(f"{agent.name}: Average reward = {avg_reward:.4f}")
```

### Running Experiments

Use the provided `main.py` script to run comprehensive experiments:

```bash
python main.py --config configs/experiment/default.yaml
```

### Configuration

The behavior of the experiments can be customized using YAML configuration files. Here's an example configuration:

```yaml
# configs/experiment/default.yaml
experiment:
  n_trials: 100
  n_steps: 1000
  confidence_level: 0.95
  plot_results: true
  save_plots: true
  output_dir: ./results

environment:
  type: bernoulli
  n_arms: 10
  probs: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

agents:
  - type: epsilon_greedy
    epsilon: 0.1
    name: "ε-Greedy (ε=0.1)"
  
  - type: ucb
    c: 2.0
    name: "UCB1 (c=2.0)"
  
  - type: thompson
    alpha: 1.0
    beta: 1.0
    name: "Thompson Sampling"
```

## 📊 Advanced Features

### 1. Delayed Feedback

The `delayed_feedback` module provides support for scenarios where rewards are not immediately observed:

```python
from delayed_feedback import DelayedFeedbackBandit

# Create environment with delayed feedback (average delay of 10 steps)
env = DelayedFeedbackBandit(
    base_bandit=BernoulliBandit(n_arms=5),
    delay_distribution=lambda: np.random.poisson(10)
)
```

### 2. Non-stationary Bandits

To handle environments where reward distributions change over time:

```python
from environments.non_stationary_bandit import NonStationaryBandit

# Create environment with drifting probabilities
env = NonStationaryBandit(
    n_arms=5,
    initial_probs=[0.1, 0.2, 0.3, 0.4, 0.5],
    drift_scale=0.01  # Scale of random walk
)
```

### 3. Contextual Bandits (Beta)

For contextual bandit problems where actions depend on observed features:

```python
from agents.contextual_bandit import LinUCB

# Initialize LinUCB agent with 10-dimensional context
agent = LinUCB(n_actions=5, context_dim=10, alpha=1.0)

# In each step:
context = get_context()  # Your context vector
action = agent.get_action(context)
reward = env.pull(action)
agent.update(context, action, reward)
```

## 📈 Performance Benchmarks

### Regret Comparison

![Regret Comparison](https://example.com/regret_comparison.png)

### Computational Efficiency

| Algorithm      | Time per step (ms) | Memory (MB) |
|----------------|-------------------|-------------|
| ε-Greedy      | 0.05              | 2.1         |
| UCB1          | 0.07              | 2.3         |
| Thompson      | 0.12              | 2.5         |
| Gradient      | 0.15              | 2.8         |
| LinUCB        | 1.20              | 15.2        |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Run linter:
   ```bash
   black .
   flake8
   ```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press.
2. Slivkins, A. (2019). *Introduction to Multi-Armed Bandits*. Foundations and Trends in Machine Learning.
3. Bubeck, S., & Cesa-Bianchi, N. (2012). *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems*. Foundations and Trends in Machine Learning.
4. Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). *A Tutorial on Thompson Sampling*. Foundations and Trends in Machine Learning.
5. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
