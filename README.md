# Repository of the AU Master's Thesis: Building Energy Production and Consumption Forecasting

Summary of files:
```
  \ preprocessor.py
      Class for preprocessing the intial data
  \ statistics.py
      Class for calculating the statistics presented in Section 4.3.2: Descriptive Statistics
  \ exploratory_visualization.py
      Script to make the plots presented in Section 4.3.1: Data Visualization
  \ arima_config.py
      Script for configuring the orders of the ARIMA parameters with ACF and PACF plots
  \ preparator.py
      Class for data preparation (for the models)
  \ error_metric_calculator.py
      Class for error metric calculation
  \ config.toml
      TOML config file for model tuning and training
  
  Model classes:
  \ persistence_model.py
  \ arima_model.py
  \ svr_model.py
  \ lstm_model.py
  
  Execute GridSearch and training:
  \ run_grid_search.py
  \ run_models.py
  
  Experiment evaluation plots:
  \ baseline_model_evaluation.py
  \ gridsearch_result_evaluation.py
  \ final_model_evlauation.py
  ```
  
  

