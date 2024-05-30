# 📈  Stock Performance Prediction

This Streamlit application allows users to build a machine learning model for predicting stock performance. The app fetches stock data from Yahoo Finance, processes the data, and uses a Random Forest Regressor to predict stock prices.

## Features

- Fetch stock data from Yahoo Finance.
- Display stock data preview.
- Set parameters for training the machine learning model.
- Train a Random Forest Regressor.
- Display performance metrics of the model.
- Download the generated model and data.- Fetch stock data from Yahoo Finance.
- Display stock data preview.
- Set parameters for training the machine learning model.
- Train a Random Forest Regressor.
- Display performance metrics of the model.
- Download the generated model and data.
- 
## Libraries Used

- **Pandas** for data wrangling
- **yfinance** for fetching stock data
- **Scikit-learn** for building the machine learning model
- **Altair** for chart creation
- **Streamlit** for user interface

## How to Use

1. **Select stock ticker**: Enter the stock ticker symbol (e.g., AAPL, MSFT).
2. **Select time period and interval**: Choose the time period and interval for fetching stock data.
3. **Set parameters**: Adjust the model parameters using the sliders and dropdowns.
4. **Train the model**: Click on "Run" to initiate the model training process.
5. **View results**: The app will display the stock data preview, performance metrics, feature importance, and prediction results.
6. **Download data**: Download the generated model and data as a ZIP file.

## Parameters

### Data Split Ratio
- Determines the proportion of data used for training versus testing.
- Adjust using the slider (default is 80% for training).

### Learning Parameters
- **Number of estimators (n_estimators)**: The number of trees in the forest.
- **Max features (max_features)**: The number of features to consider when looking for the best split.
- **Minimum samples required to split an internal node (min_samples_split)**: Higher values make the model more conservative.
- **Minimum samples required to be at a leaf node (min_samples_leaf)**: Affects the model's ability to generalize.

### General Parameters
- **Seed number (random_state)**: Ensures reproducibility of the model training.
- **Performance measure (criterion)**: Function to measure the quality of a split.
- **Bootstrap samples (bootstrap)**: Whether to use bootstrap samples when building trees.
- **Out-of-bag samples (oob_score)**: Use out-of-bag samples to estimate the R^2 on unseen data.

## Example

1. Enter `AAPL` as the stock ticker.
2. Select `1y` as the time period and `1h` as the time interval.
3. Set the data split ratio to 80%.
4. Adjust the learning and general parameters as needed.
5. Click on "Run" to train the model and view the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://www.streamlit.io/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [Altair](https://altair-viz.github.io/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
