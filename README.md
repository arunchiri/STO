# STO - Store Sales Time Series Forecasting

A machine learning project for forecasting store sales using time series analysis and deep learning models.

## Project Overview

This project predicts store sales across multiple locations, incorporating factors such as:
- Historical sales data
- Oil prices
- Store information
- Holiday/event calendars
- Transaction patterns

## Project Structure

```
STO/
├── train.py                 # Main training script
├── model.py                 # Model architecture definitions
├── data_processing.py       # Data preprocessing and feature engineering
├── sub.py                   # Submission generation script
├── dataset/                 # Raw dataset files
│   ├── train.csv            # Historical sales data
│   ├── test.csv             # Test set for predictions
│   ├── oil.csv              # Daily oil prices
│   ├── stores.csv           # Store metadata
│   ├── transactions.csv     # Transaction counts
│   ├── holidays_events.csv  # Holiday and event information
│   └── sample_submission.csv # Submission format template
├── checkpoints/             # Saved model weights
│   ├── best.pt              # Best performing model checkpoint
│   └── meta.pt              # Model metadata
├── .gitignore               # Git ignore configuration
└── .gitattributes           # Git attributes configuration
```

## Installation

### Prerequisites
- Python 3.8+
- conda (Anaconda/Miniconda)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/arunchiri/STO.git
cd STO
```

2. Create and activate conda environment:
```bash
conda create -n hrm_env python=3.9
conda activate hrm_env
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset contains:
- **train.csv**: Historical sales data with date and store information
- **test.csv**: Test set for generating predictions
- **oil.csv**: Daily crude oil prices (relevant economic indicator)
- **stores.csv**: Store metadata (location, type, cluster)
- **transactions.csv**: Transaction counts per store per date
- **holidays_events.csv**: Holiday and special event information

## Usage

### Data Processing

Prepare and process the data:
```bash
python data_processing.py
```

### Training

Train the model with custom parameters:
```bash
python train.py \
  --data_dir ./dataset \
  --epochs 25 \
  --batch_size 256 \
  --lr 0.001 \
  --patience 5
```

**Arguments:**
- `--data_dir`: Path to dataset directory (default: `./dataset`)
- `--epochs`: Number of training epochs (default: 25)
- `--batch_size`: Batch size for training (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 5)

The best model will be saved to `checkpoints/best.pt`.

### Generate Submissions

Generate predictions for the test set:
```bash
python sub.py
```

This will create submission files based on the best trained model.

## Model Architecture

The project uses a deep learning approach for time series forecasting. Key components:
- Feature engineering from date, store, and economic indicators
- Neural network model with temporal dependencies
- Early stopping to prevent overfitting

Trained model weights are stored in `checkpoints/best.pt`.

## Results

Model checkpoints are saved in the `checkpoints/` directory:
- **best.pt**: Best performing model during training
- **meta.pt**: Model metadata and configuration

## Notes

- All CSV data files are excluded from version control (see `.gitignore`)
- The `.gitattributes` has been updated to remove Git LFS tracking for CSV files
- Ensure you have sufficient disk space for the dataset files

## Future Improvements

- Hyperparameter tuning and grid search
- Ensemble methods combining multiple models
- Advanced feature engineering techniques
- Cross-validation for robust evaluation

## License

This project is for educational and research purposes.
