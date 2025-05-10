# File: main.py

import logging

from data_input.market_feed import get_market_data
from prediction_engine.sequence_preprocessor import SequencePreprocessor

def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Fetch data (uses config/strategy_config.yaml by default)
    df = get_market_data(
        symbols=None,
        start_date="2020-01-01",
        end_date="2023-12-31",
        interval="1d",
    )

    # 2. Turn DataFrame → model-ready sequences
    #    (you’ll implement this class next)
    preprocessor = SequencePreprocessor(sequence_length=10)
    X, y = preprocessor.transform(df)

    logging.info(f"Prepared {len(X)} sequences of length {preprocessor.sequence_length}")
    # 3. Pass X into your LSTM predictor, or proceed with backtesting, etc.

if __name__ == "__main__":
    main()

