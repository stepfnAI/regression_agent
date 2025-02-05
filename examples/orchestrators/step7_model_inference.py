from sfn_blueprint import Task
from regression_agent.utils.model_manager import ModelManager
import pandas as pd
from typing import Dict
import numpy as np

class ModelInference:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.model_manager = ModelManager()
        
    def execute(self):
        """Execute model inference step"""
        # try:
        # Get analysis type and data
        is_forecasting = self.session.get('is_forecasting', False)
        selected_model = self.session.get('selected_model')
        self.mappings = self.session.get('field_mappings')
        print('>>>selected_model', selected_model)
        if not selected_model:
            self.view.show_message(
                "❌ No selected model found. Please complete model selection first.",
                "error"
            )
            return False
            
        # Get split info which contains inference data
        split_info = self.session.get('split_info')
        infer_df = split_info.get('infer_df')
        print('>>>infer_df', infer_df)
        if split_info is None or 'infer_df' not in split_info:
            self.view.show_message("❌ No inference data found in split info.", "error")
            return False
            
        # Get appropriate target column based on analysis type
        target_column = self.mappings.get('forecasting_field' if is_forecasting else 'target')
        
        # Get predictions
        with self.view.display_spinner('Generating predictions...'):
            predictions = self._generate_predictions(
                selected_model, 
                infer_df, 
                target_column,
                is_forecasting
            )
        
        if predictions is None:
            self.view.show_message("❌ Failed to generate predictions.", "error")
            return False
            
        # Save predictions
        self.session.set('predictions', predictions)
        
        # Display results
        self._display_predictions(predictions, infer_df, is_forecasting)
        
        # Save summary
        self._save_step_summary(predictions, infer_df, is_forecasting)
        # Always show results if available (moved outside the if block)
        results_df = self.session.get('inference_results')
        if results_df is not None:
            # Show summary
            summary = self.session.get('step_7_summary')
            if summary:
                self.view.show_message(summary, "success")
            
            # Display full results
            self.view.display_subheader("📊 Detailed Results")
            self.view.display_dataframe(results_df)
            
            # Add download button
            self.view.create_download_button(
                label="📥 Download Predictions as CSV",
                data=results_df.to_csv(index=False),
                file_name="prediction_results.csv",
                mime_type='text/csv'
            )
        
        # Return True only if step is complete
        return self.session.get('step_7_complete', False)
        
        # except Exception as e:
        #     self.view.show_message(f"Error in model inference: {str(e)}", "error")
        #     return False
            
    def _generate_predictions(self, model_info: Dict, infer_df: pd.DataFrame, 
                            target_column: str, is_forecasting: bool) -> pd.Series:
        """Generate predictions using the selected model"""
    # try:
        model = model_info.get('model')
        features = model_info.get('training_features', [])
        model_name = model_info.get('model_name', '').lower()
        mappings = self.session.get('field_mappings')
        
        print('>>> Type of model:', type(model))
        print('>>> Model:', model)
        print('>>> Features:', features)
        
        if is_forecasting:
            if model_name == 'sarimax':
                try:
                    # Get exogenous features
                    exog_features = [f for f in features 
                                   if f != mappings.get('date') 
                                   and f != target_column]
                    
                    # Get start and end dates from inference data
                    date_col = mappings.get('date')
                    start_date = pd.to_datetime(infer_df[date_col].min())
                    end_date = pd.to_datetime(infer_df[date_col].max())
                    
                    print('>>> Forecasting from:', start_date)
                    print('>>> Forecasting to:', end_date)
                    
                    # Get the required number of periods from model
                    nobs = model.nobs  # Number of observations used in training
                    print('>>> Training observations:', nobs)
                    
                    # Prepare exog data just for prediction period
                    exog_data = infer_df[exog_features].astype(float)
                    print('>>> Exog data shape:', exog_data.shape)
                    
                    # Get forecast for specific dates
                    predictions = model.get_forecast(
                        steps=len(infer_df),
                        exog=exog_data
                    ).predicted_mean
                    
                    # Ensure index matches inference dates
                    predictions.index = pd.to_datetime(infer_df[date_col])
                    
                except Exception as e:
                    print(f">>> SARIMAX error: {str(e)}")
                    print(f">>> Available methods: {dir(model)}")
                    raise
            elif model_name == 'prophet':
                # Prophet specific handling
                date_col = mappings.get('date')
                forecast_df = pd.DataFrame()
                forecast_df['ds'] = pd.to_datetime(infer_df[date_col])
                forecast_df['y'] = np.nan
                forecast = model.predict(forecast_df)
                predictions = forecast['yhat'].values
                print('>>>forecast', forecast)
                print('>>>forecast_df', forecast_df)
            else:
                # For other forecasting models (XGBoost, LightGBM)
                predictions = model.predict(infer_df[features])
        else:
            # Standard regression prediction
            predictions = model.predict(infer_df[features])
        
        print('>>>predictions', predictions)
        return predictions
        
    # except Exception as e:
    #     self.view.show_message(f"Prediction error: {str(e)}", "error")
    #     return None
            
    def _display_predictions(self, predictions, infer_df: pd.DataFrame, 
                           is_forecasting: bool):
        """Display prediction results"""
        self.view.display_subheader("Prediction Results")
        
        # Create results DataFrame
        results_df = infer_df.copy()
        
        if is_forecasting:
            # For forecasting, predictions come with a datetime index
            date_col = self.session.get('field_mappings').get('date')
            
            # Convert predictions to a Series with datetime index if not already
            if not isinstance(predictions, pd.Series):
                predictions = pd.Series(predictions, index=pd.to_datetime(infer_df[date_col]))
            
            # Merge predictions with results using the date
            results_df['Predicted'] = predictions.values
            
            # For display, show date and prediction
            display_df = results_df[[date_col, 'Predicted']].head(10)
            msg = "**Sample Forecasts** (First 10 periods):\n"
        else:
            # For regression, predictions are a simple array
            results_df['Predicted'] = predictions
            
            # Show basic stats
            msg = "**Prediction Statistics:**\n"
            msg += f"- Mean Prediction: {predictions.mean():.2f}\n"
            msg += f"- Min Prediction: {predictions.min():.2f}\n"
            msg += f"- Max Prediction: {predictions.max():.2f}\n\n"
            msg += "**Sample Predictions** (First 10 records):\n"
            display_df = results_df.head(10)
        
        self.session.set('inference_results', results_df)
        print('>>>results_df', results_df.head(3))
        
        self.view.show_message(msg, "info")
        self.view.display_dataframe(display_df)
        
    def _save_step_summary(self, predictions: pd.Series, infer_df: pd.DataFrame, 
                          is_forecasting: bool):
        """Save step summary with prediction details"""
        summary = "✅ Model Inference Complete\n\n"
        
        # Add analysis type
        summary += f"**Analysis Type:** {'Forecasting' if is_forecasting else 'Regression'}\n\n"
        
        # Add prediction stats
        summary += "**Prediction Statistics:**\n"
        summary += f"- Total Predictions: {len(predictions)}\n"
        summary += f"- Mean Value: {predictions.mean():.2f}\n"
        summary += f"- Range: {predictions.min():.2f} to {predictions.max():.2f}\n"
        
        if is_forecasting:
            # Add forecasting-specific info
            date_col = self.session.get('field_mappings').get('date')
            summary += f"\n**Forecast Period:**\n"
            summary += f"- From: {infer_df[date_col].min()}\n"
            summary += f"- To: {infer_df[date_col].max()}\n"
        
        self.session.set('step_7_summary', summary) 