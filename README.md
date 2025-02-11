# Regression & Forecasting Agent

An AI-powered agent that automates the end-to-end process of building and selecting regression and forecasting models, with interactive capabilities and comprehensive model analysis.

## 🌟 Features

- **Multiple Problem Types Support**: 
  - Optimized for continuous value prediction (regression)
  - Time series forecasting with automatic temporal feature handling
- **Intelligent Data Processing**: Automatically handles data preprocessing and validation
- **Smart Feature Engineering**: 
  - AI-powered categorical feature handling and encoding
  - Automated target leakage detection and mitigation
  - Interactive feature selection and removal
  - Temporal feature generation for forecasting
- **Automated Model Pipeline**:
  - Data validation and cleaning
  - Feature preprocessing
  - Intelligent data splitting (including time-aware splits for forecasting)
  - Model training with multiple algorithms
  - Model selection and evaluation
- **Interactive Interface**: Review and modify AI suggestions at each step
- **Flexible Data Input**: Supports various data formats
- **Visual Progress Tracking**: Clear feedback on each step

## 🚀 Getting Started

### Prerequisites

- Python 3.9-3.11
- OpenAI API key

### Platform-Specific Requirements

#### macOS
If you're using macOS, you'll need to install OpenMP support:

```bash
brew install libomp
```

#### Ubuntu/Debian
Install OpenMP support using:

```
bash
sudo apt-get install libomp-dev
```

#### Windows
No additional steps required - OpenMP support is included with the standard installation.


### Installation

1. Clone the repository:

```
git clone git@github.com:stepfnAI/regression_agent.git
cd regression_agent
```

2. Create and activate a virtual environment:

```bash
pip install virtualenv                # Install virtualenv if not already installed
virtualenv venv                       # Create virtual environment
source venv/bin/activate             # Linux/Mac
# OR
.\venv\Scripts\activate               # Windows
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

### Running the Application

```bash
# Windows
streamlit run .\examples\main_app.py

# Linux/Mac
streamlit run ./examples/main_app.py
```


## 🔄 Workflow

1. **Data Upload**
   - Upload your dataset
   - Initial data preview
   - Reset functionality available

2. **Problem Type Selection**
   - Choose between regression and forecasting
   - Configure problem-specific parameters
   - Automatic validation of requirements for each type

3. **Data Validation**
   - Automatic field mapping (CUST_ID, REVENUE, TARGET, TIMESTAMP, etc.)
   - Data type validation and suggestions
   - Numeric target validation
   - Temporal consistency checks for forecasting

4. **Feature Preprocessing**
   - Automatic categorical feature detection
   - Intelligent encoding selection
   - Interactive preprocessing options

5. **Data Splitting**
   - Smart train/validation/inference split
   - Temporal awareness for time-series data
   - Configurable validation window
   - Time-based validation for forecasting problems

6. **Model Training**
   - Multiple regression models (XGBoost, LightGBM, CatBoost)
   - Default hyperparameters for regression (Note: Automated hyperparameter tuning coming soon)
   - Progress tracking

7. **Model Selection**
   - Comprehensive model evaluation (R², MSE, MAE)
   - AI-powered model recommendations
   - Interactive model selection interface

8. **Model Inference**
   - Batch prediction capabilities
   - Prediction statistics and insights
   - Results export

## 🔜 Upcoming Features

- **Automated Hyperparameter Optimization**: 
  - Intelligent hyperparameter tuning for regression models
  - Bayesian optimization support
  - Cross-validation integration
- **Enhanced Model Explainability**
- **Additional Model Support**
- **Advanced Feature Engineering Options**

## 🛠️ Key Components

- **SFNMappingAgent**: Handles field mapping
- **SFNDataTypeSuggestionAgent**: Manages data type validation
- **SFNCategoricalFeatureAgent**: Processes categorical features
- **SFNDataSplittingAgent**: Manages data splitting
- **SFNModelTrainingAgent**: Handles regression model training
- **SFNModelSelectionAgent**: Provides model recommendations
- **StreamlitView**: Manages user interface
- **SFNSessionManager**: Handles application state

## 📦 Dependencies

- sfn_blueprint==0.5.2
- sfn-llm-client==0.1.0
- lightgbm>=4.1.0
- catboost>=1.2.2

## 🔒 Security

- Secure API key handling
- Input validation
- Safe data processing
- Environment variable management

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact
puneet@stepfunction.ai