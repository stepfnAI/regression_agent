from sfn_blueprint import Task, SFNValidateAndRetryAgent
from regression_agent.agents.categorical_feature_agent import SFNCategoricalFeatureAgent
from regression_agent.agents.leakage_detection_agent import SFNLeakageDetectionAgent
from regression_agent.config.model_config import DEFAULT_LLM_PROVIDER

class FeaturePreprocessing:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.categorical_agent = SFNCategoricalFeatureAgent()
        self.leakage_detector = SFNLeakageDetectionAgent()
        
    def execute(self):
        """Execute preprocessing step"""
        try:
            # Get analysis type
            is_forecasting = self.session.get('is_forecasting', False)
            
            # Handle categorical features first
            cat_result = self._handle_categorical_features()
            if cat_result is None:  # Still waiting for confirmation
                return False
            if not cat_result:  # Error occurred
                return False
            
            # Only do leakage detection for regression
            if not is_forecasting:
                leakage_result = self._handle_leakage_detection()
                if leakage_result is None:  # Still waiting for user input
                    return False
                if not leakage_result:  # Error occurred
                    return False
            
            # Save preprocessing summary
            self._save_step_summary()
            self.session.set('step_3_complete', True)
            return True
            
        except Exception as e:
            self.view.show_message(f"Error in preprocessing: {str(e)}", "error")
            return False
        
    def _handle_leakage_detection(self):
        """Handle leakage detection analysis"""
        try:
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            target_col = mappings.get('target')
            
            # Check if leakage detection was already completed
            if self.session.get('leakage_detection_complete'):
                return True

            # If we've already analyzed but waiting for confirmation
            if self.session.get('leakage_analysis_done'):
                leakage_analysis = self.session.get('leakage_analysis')
            else:
                # Do the initial analysis
                with self.view.display_spinner('🔍 Analyzing potential target leakage...'):
                    task_data = {
                        'df': df,
                        'target_column': target_col
                    }
                    
                    leakage_analysis = self.leakage_detector.execute_task(
                        Task("Detect target leakage", data=task_data)
                    )
                    # Store the analysis results
                    self.session.set('leakage_analysis', leakage_analysis)
                    self.session.set('leakage_analysis_done', True)
            
            # Get all features that need attention
            severe_features = leakage_analysis['severe_leakage']
            review_features = [rec['feature'] for rec in leakage_analysis['recommendations']['review']]
            
            if severe_features or review_features:
                warning_msg = "⚠️ Target Leakage Analysis:\n\n"
                
                # Add high-risk features with details
                if severe_features:
                    warning_msg += f"{len(severe_features)} High-risk features detected ⚠️\n"
                    for rec in leakage_analysis['recommendations']['remove']:
                        warning_msg += f"- {rec['feature']}: {rec['reason']}\n"
                    warning_msg += "\n"
                
                # Add potential concern features with details
                if review_features:
                    warning_msg += f"{len(review_features)} Features with potential concerns detected ℹ️\n"
                    for rec in leakage_analysis['recommendations']['review']:
                        warning_msg += f"- {rec['feature']}: {rec['reason']}\n"
                
                self.view.show_message(warning_msg, "error")
                
                # Get mapped columns to exclude from selection
                mapped_columns = set(mappings.values())
                
                # Filter out mapped columns from available features
                available_features = [col for col in df.columns if col not in mapped_columns]
                default_selected = [f for f in (severe_features + review_features) if f not in mapped_columns]
                
                # Let user choose features to remove with all flagged features pre-selected
                features_to_remove = self.view.multiselect(
                    "Select features to remove (high-risk and potential concern features are pre-selected):",
                    options=available_features,
                    default=default_selected
                )
                
                if self.view.display_button("Confirm Feature Removal"):
                    if features_to_remove:
                        # Store removed features in session
                        self.session.set('removed_features', features_to_remove)
                        # Remove selected features from DataFrame
                        df = df.drop(columns=features_to_remove)
                        self.session.set('df', df)
                        
                        self.view.show_message(
                            f"✅ Removed {len(features_to_remove)} features",
                            "success"
                        )
                    # Mark leakage detection as complete
                    self.session.set('leakage_detection_complete', True)
                    self.session.set('leakage_analysis_done', False)  # Reset for next time
                    return True
                
                # Don't return False, just return None to keep the UI state
                return None
            else:
                self.view.show_message("✅ No target leakage detected", "success")
                self.session.set('leakage_detection_complete', True)
                return True
            
        except Exception as e:
            self.view.show_message(f"Error in leakage detection: {str(e)}", "error")
            return False
        
    def _handle_categorical_features(self):
        """Handle categorical feature processing"""
        try:
            # Check if already completed
            if self.session.get('categorical_features_complete'):
                return True
            
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            
            with self.view.display_spinner('🤖 AI is analyzing categorical features...'):
                task_data = {
                    'df': df,
                    'mappings': mappings
                }
                
                result = self.categorical_agent.execute_task(Task("Analyze categorical features", data=task_data))
                
                modified_df = result['df']
                feature_info = result['feature_info']
            
            # Display modified data and feature information
            self.view.display_subheader("Categorical Feature Processing")
            self.view.display_dataframe(modified_df.head())
            
            # Show encoding summary
            self.view.display_subheader("Encoding Summary")
            summary_msg = "Applied Encodings:\n"
            for feature, info in feature_info.items():
                summary_msg += f"- {feature}: **{info['encoding_type']}**\n"
                if 'cardinality' in info:
                    summary_msg += f"  - Unique values: {info['cardinality']}\n"
            self.view.show_message(summary_msg, "info")
            
            if self.view.display_button("Confirm Feature Processing"):
                self.session.set('df', modified_df)
                self.session.set('feature_info', feature_info)
                self.session.set('categorical_features_complete', True)
                return True
                
            return None  # Return None to keep UI state when button not clicked
            
        except Exception as e:
            self.view.show_message(f"Error in categorical feature processing: {str(e)}", "error")
            return False
        
    def _save_step_summary(self):
        """Save preprocessing summary"""
        is_forecasting = self.session.get('is_forecasting', False)
        summary = "✅ Step 3 Complete\n\n"
        
        # Add categorical features info
        cat_features = self.session.get('categorical_features', {})
        if cat_features:
            summary += "**Categorical Features:**\n"
            for col, info in cat_features.items():
                summary += f"- {col}: {len(info['unique_values'])} unique values\n"
        
        # Add leakage info only for regression
        if not is_forecasting:
            leakage_info = self.session.get('leakage_info', {})
            if leakage_info:
                summary += "\n**Leakage Detection:**\n"
                for feature, risk in leakage_info.items():
                    summary += f"- {feature}: {risk} risk\n"
        
        self.session.set('step_3_summary', summary) 