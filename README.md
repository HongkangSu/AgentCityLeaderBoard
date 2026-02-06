# AgentCity LeaderBoard

A multi-agent framework for continuous construction and evaluation of spatiotemporal benchmarks on top of LibCity.

## Project Structure

```
LeaderBoard/
├── app.py                    # Flask backend server
├── requirements.txt          # Python dependencies
├── static/
│   ├── index.html           # Home page - AgentCity introduction
│   ├── quickstart.html      # Quick Start guide
│   └── leaderboard.html     # LeaderBoard page with task/dataset navigation
└── result/                  # Results directory (task/dataset/model results)
    ├── traffic_state_pred/
    ├── eta/
    └── traj_loc_pred/
```

## Pages

### 1. Home Page (`/`)
- AgentCity project overview
- Key features showcase
- Architecture explanation
- Three-stage process visualization

### 2. Quick Start (`/quickstart`)
- List of 8 migrated models
- Installation instructions
- Environment setup guide
- Model verification steps

### 3. LeaderBoard (`/leaderboard`)
- Interactive leaderboard with URL-based navigation
- Task selection
- Dataset browsing
- Model performance comparison
- Supports URL parameters: `/leaderboard?task=traffic_state_pred&dataset=METR_LA`

## Running the Application

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Environment Variables
```bash
export ANTHROPIC_BASE_URL=""
export ANTHROPIC_API_KEY=""
```

### Start Server
```bash
python app.py
```

The application will be available at `http://localhost:8085`

## Features

- **LibCity Design**: Matches LibCity's blue-green gradient theme (#1b8ce9 to #0bf7bc)
- **Responsive Layout**: Mobile-friendly design
- **URL Navigation**: Direct links to tasks and datasets
- **Interactive Tables**: Sortable leaderboard with multiple metrics
- **Real-time Stats**: Dynamic task and dataset counters

## API Endpoints

- `GET /api/tasks` - List all tasks
- `GET /api/tasks/<task>/datasets` - List datasets for a task
- `GET /api/tasks/<task>/datasets/<dataset>/rankings` - Get model rankings

## Migrated Models

1. PatchSTG
2. LSTGAN
3. MLCAFormer
4. RSTIB
5. GriddedTNP
6. EAC
7. SRSNet
8. ST-SSDL

All models are available in the LibCity repository under `/model/traffic_speed_prediction`.
