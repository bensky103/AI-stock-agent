import pytest
import pandas as pd
from data_input.market_feed import get_market_data, load_config

def test_load_config(tmp_path):
    # create a temp YAML
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("symbols:\n  - TSLA\n")
    cfg = load_config(str(cfg_file))
    assert "TSLA" in cfg["symbols"]

def test_get_market_data_single_symbol(monkeypatch, tmp_path):
    # monkeypatch yfinance.download to return a fake DataFrame
    import yfinance as yf
    
    # Create a properly formatted fake DataFrame that matches yfinance output
    dates = pd.date_range("2021-01-01", periods=2)
    fake_df = pd.DataFrame({
        'Open': [1, 2],
        'High': [1, 2],
        'Low': [1, 2],
        'Close': [1, 2],
        'Adj Close': [1, 2],
        'Volume': [100, 200],
    }, index=dates)
    
    def mock_download(**kwargs):
        # Ensure the index is named 'Date' to match yfinance output
        fake_df.index.name = 'Date'
        return fake_df
    
    monkeypatch.setattr(yf, "download", mock_download)

    # write config.yaml
    cfg_file = tmp_path / "strategy_config.yaml"
    cfg_file.write_text("symbols:\n  - FAKE\n")
    
    df = get_market_data(
        symbols=None,
        start_date="2021-01-01",
        end_date="2021-01-02",
        interval="1d",
        config_path=str(cfg_file)
    )
    
    # Verify the result
    assert isinstance(df, pd.DataFrame)
    assert ("FAKE", pd.Timestamp("2021-01-01")) in df.index
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume'])

