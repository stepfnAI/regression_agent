from sfn_blueprint import Task, SFNValidateAndRetryAgent
from regression_agent.agents.model_selection_agent import SFNModelSelectionAgent
from regression_agent.utils.model_manager import ModelManager
from regression_agent.config.model_config import DEFAULT_LLM_PROVIDER
from typing import Dict

class ModelSelection:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.selection_agent = SFNModelSelectionAgent()
        self.model_manager = ModelManager()
        
    def execute(self):
        """Execute model selection step"""
        try:
            # Get analysis type and model results
            is_forecasting = self.session.get('is_forecasting', False)
            model_results = self.session.get('model_results')
            
            if not model_results:
                self.view.show_message(
                    "❌ No model results found. Please complete model training first.",
                    "error"
                )
                return False
            
            # Format model results for selection
            selection_info = self._format_model_results(model_results, is_forecasting)
            
            # Initialize selection agent with appropriate type
            selection_agent = SFNModelSelectionAgent(
                analysis_type="forecasting" if is_forecasting else "regression"
            )
            
            # Get model recommendation
            with self.view.display_spinner('🤖 AI is analyzing model performance...'):
                task_data = {
                    'selection_info': selection_info,
                    'custom_instructions': ""
                }
                
                recommendation = selection_agent.execute_task(
                    Task("Select best model", data=task_data)
                )
            
            # Display recommendation
            self._display_recommendation(recommendation, is_forecasting)
            
            # Save selected model
            selected_model = recommendation.get('selected_model')
            if selected_model and selected_model in model_results:
                self.session.set('selected_model', model_results[selected_model])
                self.session.set('model_selection_details', recommendation)
                self._save_step_summary(recommendation, is_forecasting)
                self.session.set('step_6_complete', True)
                return True
            else:
                self.view.show_message(
                    "❌ Selected model not found in results.",
                    "error"
                )
                return False
                
        except Exception as e:
            self.view.show_message(f"Error in model selection: {str(e)}", "error")
            return False
            
    def _format_model_results(self, model_results: Dict, is_forecasting: bool) -> str:
        """Format model results for LLM consumption"""
        formatted_info = "Model Performance Summary:\n\n"
        
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            formatted_info += f"Model: {model_name}\n"
            
            if is_forecasting:
                formatted_info += f"- MAPE: {metrics.get('mape', 'N/A')}\n"
                formatted_info += f"- RMSE: {metrics.get('rmse', 'N/A')}\n"
                formatted_info += f"- MAE: {metrics.get('mae', 'N/A')}\n"
            else:
                formatted_info += f"- R² Score: {metrics.get('r2', 'N/A')}\n"
                formatted_info += f"- MSE: {metrics.get('mse', 'N/A')}\n"
                formatted_info += f"- MAE: {metrics.get('mae', 'N/A')}\n"
            
            # Add records info
            records = results.get('records_info', {})
            formatted_info += f"- Training Records: {records.get('used_train', 0)}/{records.get('total_train', 0)}\n"
            formatted_info += f"- Validation Records: {records.get('used_valid', 0)}/{records.get('total_valid', 0)}\n\n"
            
        return formatted_info
            
    def _display_recommendation(self, recommendation: Dict, is_forecasting: bool):
        """Display model selection recommendation"""
        self.view.display_subheader("Model Selection Results")
        
        # Show selected model and explanation
        selected_model = recommendation.get('selected_model', 'Unknown')
        explanation = recommendation.get('explanation', '')
        
        msg = f"🏆 **Selected Model: {selected_model}**\n\n"
        msg += f"**Explanation:**\n{explanation}\n\n"
        
        # Show model rankings
        rankings = recommendation.get('model_rankings', [])
        if rankings:
            msg += "**Model Rankings:**\n"
            for rank in rankings:
                strengths = ", ".join(rank.get('key_strengths', []))
                msg += f"{rank['rank']}. {rank['model']}: {strengths}\n"
        
        self.view.show_message(msg, "success")
        
    def _save_step_summary(self, recommendation: Dict, is_forecasting: bool):
        """Save step summary with selection details"""
        summary = "✅ Model Selection Complete\n\n"
        
        # Add analysis type
        summary += f"**Analysis Type:** {'Forecasting' if is_forecasting else 'Regression'}\n\n"
        
        # Add selected model
        selected_model = recommendation.get('selected_model', 'Unknown')
        summary += f"**Selected Model:** {selected_model}\n\n"
        
        # Add key strengths
        for rank in recommendation.get('model_rankings', []):
            if rank['model'] == selected_model:
                strengths = rank.get('key_strengths', [])
                if strengths:
                    summary += "**Key Strengths:**\n"
                    for strength in strengths:
                        summary += f"- {strength}\n"
                break
        
        # Add comparison summary
        comparison = recommendation.get('comparison_summary')
        if comparison:
            summary += f"\n**Model Comparison:**\n{comparison}\n"
        
        self.session.set('step_6_summary', summary) 