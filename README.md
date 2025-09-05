# Fraud Detection Project

A machine learning project for detecting fraudulent transactions using credit card data.

## Project Overview

This repository contains a fraud detection system that analyzes credit card transaction data to identify potentially fraudulent activities. The project uses machine learning algorithms to classify transactions as legitimate or fraudulent.

## Dataset

The project uses three main datasets:
- `creditcard.csv` - Main dataset with credit card transactions
- `train_data.csv` - Training dataset for model development
- `test_data.csv` - Test dataset for model evaluation

**Note**: Due to the large size of the CSV files, they are not included in the Git repository. Please ensure you have access to these datasets before running the code.

## Project Structure

```
Fraude-detection/
├── src/                    # Source code directory
│   ├── data/              # Data processing scripts
│   ├── models/            # Model training and evaluation
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore rules
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Fraude-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the required datasets in the project root directory.

## Usage

1. Place your CSV files in the project root directory
2. Run the data preprocessing scripts
3. Train the model using the training scripts
4. Evaluate the model performance

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
