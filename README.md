# Stryktips Analysis Project

This project analyzes historical Stryktips data to build predictive models and develop betting strategies.

## Project Structure

```
stryktips-analys/
├── data/               # Data files
│   ├── raw/           # Raw CSV files
│   ├── processed/     # Processed data
│   └── models/        # Saved models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
└── app/              # Web application
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Place your raw CSV files in the `data/raw/` directory
2. Run the notebooks in order:
   - 1_data_exploration.ipynb
   - 3_model_building.ipynb
   - 4_betting_strategy.ipynb

## Web Application

To run the web application:
```bash
python app/main.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 