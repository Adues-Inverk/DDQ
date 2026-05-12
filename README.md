# Deep Dialog - Dialog System and Reinforcement Learning Framework

A dialog simulation framework for building, training, and evaluating dialog agents using reinforcement learning, specifically Deep Q-Networks (DQN). This system includes rule-based agents, learning-based agents, and various user simulators for dialog research.

## Project Overview

This project implements a complete dialog system pipeline including:
- **Dialog Manager**: Orchestrates conversations between agents and user simulators
- **Agents**: Multiple agent implementations (rule-based, DQN-based)
- **User Simulators**: Rule-based and model-based user simulators
- **NLU (Natural Language Understanding)**: LSTM-based semantic parsing
- **NLG (Natural Language Generation)**: LSTM-based response generation
- **Reinforcement Learning**: DQN training with experience replay

## Project Structure

```
Source/
├── run.py                          # Main entry point for dialog simulation
├── draw_learning_curve.py          # Visualization utilities
├── deep_dialog/
│   ├── dialog_config.py           # Global configuration
│   ├── agents/                    # Dialog agent implementations
│   │   ├── agent.py              # Base agent class
│   │   ├── agent_baselines.py    # Rule-based agents (Inform, Request, Random, Echo)
│   │   ├── agent_cmd.py          # Command-line agent for human interaction
│   │   └── agent_dqn.py          # Deep Q-Network agent
│   ├── usersims/                 # User simulator implementations
│   │   ├── usersim.py            # Base user simulator
│   │   ├── usersim_rule.py       # Rule-based user simulator
│   │   ├── usersim_model.py      # Model-based user simulator
│   │   └── user_model.py         # User goal tracking
│   ├── dialog_system/            # Dialog management
│   │   ├── dialog_manager.py     # Main dialog orchestrator
│   │   ├── state_tracker.py      # Dialog state management
│   │   ├── dict_reader.py        # Dictionary utilities
│   │   ├── kb_helper.py          # Knowledge base utilities
│   │   └── utils.py              # Helper functions
│   ├── nlu/                      # Natural Language Understanding
│   │   ├── nlu.py               # NLU interface
│   │   ├── lstm.py              # LSTM model
│   │   ├── bi_lstm.py           # Bidirectional LSTM
│   │   ├── seq_seq.py           # Sequence-to-sequence models
│   │   └── utils.py             # NLU utilities
│   ├── nlg/                      # Natural Language Generation
│   │   ├── nlg.py               # NLG interface
│   │   ├── lstm_decoder_tanh.py  # LSTM decoder
│   │   ├── decoder.py           # Base decoder
│   │   └── utils.py             # NLG utilities
│   ├── qlearning/                # Reinforcement Learning
│   │   ├── dqn.py               # Base DQN implementation
│   │   ├── dqn_torch.py         # PyTorch DQN implementation
│   │   └── utils.py             # RL utilities
│   ├── data/                     # Data files
│   │   ├── dicts.v3.p/.json     # Movie domain dictionaries
│   │   ├── movie_kb.1k/v2.p/.json # Movie knowledge base
│   │   ├── dia_acts.txt         # Dialog act set
│   │   ├── slot_set.txt         # Slot set definitions
│   │   ├── user_goals_*.p       # User goal templates
│   │   └── dia_act_nl_pairs.v6.json # Dialog act - NL mappings
│   ├── models/                   # Pre-trained models
│   │   ├── nlg/                 # Pre-trained NLG models
│   │   └── nlu/                 # Pre-trained NLU models
│   └── checkpoints/              # Saved agent models and performance records
```

## Quick Start

### Prerequisites
- Python 3.x
- PyTorch
- NumPy

### Basic Usage

Run a simple dialog simulation with a rule-based agent and user simulator:

```bash
python run.py --episodes 100 --agt 1 --usr 1
```

### Common Commands

**Train a DQN agent:**
```bash
python run.py \
  --episodes 500 \
  --agt 9 \
  --usr 1 \
  --warm_start 1 \
  --warm_start_epochs 100 \
  --write_model_dir ./deep_dialog/checkpoints/
```

**Run a trained DQN model:**
```bash
python run.py \
  --episodes 100 \
  --agt 9 \
  --usr 1 \
  --trained_model_path ./deep_dialog/checkpoints/trained_model.pkl \
  --predict_mode True
```

**Interactive dialog (command-line input):**
```bash
python run.py --agt 0 --usr 1
```

## Configuration Parameters

### Dialog Parameters
- `--max_turn`: Maximum turns per dialog (default: 20, 0=no limit)
- `--episodes`: Number of episodes to run (default: 1)
- `--run_mode`: 0=NL only, 1=Dialog act only, 2=both (default: 0)
- `--act_level`: 0=dialog act level, 1=NL level (default: 0)

### Agent Selection (`--agt`)
- `0`: Command-line input (human)
- `1`: Inform Agent (rule-based)
- `2`: Request All Agent (rule-based)
- `3`: Random Agent
- `4`: Echo Agent
- `5`: Request Basics Agent (rule-based)
- `9`: DQN Agent (reinforcement learning)

### User Simulator Selection (`--usr`)
- `0`: Human user
- `1`: Rule-based simulator (default)
- `2`: Model-based simulator

### Error Simulation
- `--slot_err_prob`: Slot error probability (default: 0.05)
- `--slot_err_mode`: 0=slot_val only, 1=three error types (default: 0)
- `--intent_err_prob`: Intent error probability (default: 0.05)

### DQN Agent Parameters
- `--dqn_hidden_size`: Hidden layer size (default: 60)
- `--batch_size`: Training batch size (default: 16)
- `--gamma`: Discount factor (default: 0.9)
- `--epsilon`: Exploration rate (default: 0)
- `--experience_replay_pool_size`: Replay buffer size (default: 5000)
- `--warm_start`: Enable warm start with rule-based policy (default: 1)
- `--warm_start_epochs`: Warm start episodes (default: 100)
- `--planning_steps`: Planning horizon (default: 4)
- `--train_world_model`: Train world model during training (default: 1)
- `--grounded`: Use environment instead of world model for planning (default: 0)

### Data Paths
- `--dict_path`: Dictionary file (default: `./deep_dialog/data/dicts.v3.p`)
- `--movie_kb_path`: Movie KB file (default: `./deep_dialog/data/movie_kb.1k.p`)
- `--act_set`: Dialog act set (default: `./deep_dialog/data/dia_acts.txt`)
- `--slot_set`: Slot set (default: `./deep_dialog/data/slot_set.txt`)
- `--goal_file_path`: User goals file (default: `./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p`)
- `--diaact_nl_pairs`: Dialog act - NL pairs (default: `./deep_dialog/data/dia_act_nl_pairs.v6.json`)
- `--nlg_model_path`: Pre-trained NLG model path
- `--nlu_model_path`: Pre-trained NLU model path

### Model Saving
- `--write_model_dir`: Directory to save models (default: `./deep_dialog/checkpoints/`)
- `--save_check_point`: Save model every N episodes (default: 10)

## Key Components

### Dialog Manager
Orchestrates conversations between an agent and user simulator, managing state and dialog flow.

### Agents
- **Rule-based Agents**: Use handcrafted policies for specific tasks
- **DQN Agent**: Uses Deep Q-Learning with experience replay for adaptive behavior

### User Simulators
- **Rule-based**: Follows predefined dialog patterns based on user goals
- **Model-based**: Learned user behavior model for more realistic simulation

### State Representation
The dialog state includes:
- User utterance
- Agent action history
- Slot-value pairs
- Dialog context

### NLU/NLG Pipeline
- **NLU**: Converts natural language to semantic representations (dialog acts)
- **NLG**: Converts dialog acts back to natural language responses

## Output

After running episodes, the system generates:
- `agt_X_performance_records.json`: Success rate, average reward, average turns per episode
- `agt_X_BEST_EPOCH_CURRENT_EPOCH_SUCCESS_RATE.pkl`: Saved model checkpoint

## Performance Metrics
- **Success Rate**: Percentage of successful dialogs
- **Average Reward**: Mean reward across episodes
- **Average Turns**: Mean dialog length

## References

This project implements concepts from:
- Deep Reinforcement Learning for Dialog Management
- User Simulation for Dialog System Evaluation
- LSTM-based NLU and NLG components
- Deep Dyna-Q learning with world models for planning

## Notes

- Ensure all data files are present before running
- Pre-trained NLG and NLU models are required for NL-level dialogs
- DQN training is computationally intensive; GPU is recommended
- World model training can be enabled/disabled via `--train_world_model`
