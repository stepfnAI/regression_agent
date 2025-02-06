from sfn_blueprint import Task, SFNValidateAndRetryAgent
from regression_agent.agents.mapping_agent import SFNMappingAgent as MappingAgent
from regression_agent.config.model_config import DEFAULT_LLM_PROVIDER
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

class DataValidation:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.mapping_agent = MappingAgent()
        self.excluded_features = {}  # Will store features excluded from model training
        
    def execute(self):
        """Execute the data validation step"""
        # Check if data is available
        df = self.session.get('df')
        if df is None:
            self.view.show_message("❌ No data found. Please upload data first.", "error")
            return False
        
        # Get mapping status
        mapping_complete = self.session.get('mapping_complete', False)
        mapping_confirmed = self.session.get('mapping_confirmed', False)
        data_types_validated = self.session.get('data_types_complete', False)
        
        # First handle mapping if not confirmed
        if not mapping_confirmed:
            if not self._handle_mapping():  # Call mapping agent first
                return False
            return False  # Return to refresh UI after mapping
            
        # If mapping confirmed but data types not validated
        if mapping_confirmed and not data_types_validated:
            if not self._handle_data_types():
                return False
                
        # If data types validated, proceed to analysis type selection
        if data_types_validated and not mapping_complete:
            return self._get_analysis_type(list(df.columns))
        
        return mapping_complete
        
    def _handle_mapping(self):
        """Handle column mapping logic"""
        try:
            df = self.session.get('df')
            mapping_confirmed = self.session.get('mapping_confirmed', False)
            
            # Get AI suggestions for mapping only if not already confirmed
            if not mapping_confirmed and not self.session.get('suggested_mappings'):
                with self.view.display_spinner('🤖 AI is mapping critical fields...'):
                    mapping_task = Task("Map columns", data=df)
                    validation_task = Task("Validate field mapping", data=df)
                    
                    validate_and_retry_agent = SFNValidateAndRetryAgent(
                        llm_provider=DEFAULT_LLM_PROVIDER,
                        for_agent='field_mapper'
                    )
                    
                    mappings, validation_message, is_valid = validate_and_retry_agent.complete(
                        agent_to_validate=self.mapping_agent,
                        task=mapping_task,
                        validation_task=validation_task,
                        method_name='execute_task',
                        get_validation_params='get_validation_params',
                        max_retries=2,
                        retry_delay=3.0
                    )
                    
                    if is_valid:
                        self.session.set('suggested_mappings', mappings)
                    else:
                        self.view.show_message("❌ AI couldn't generate valid field mappings.", "error")
                        self.view.show_message(validation_message, "warning")
                        return False
            
            # Display mapping interface
            result = self._display_mapping_interface()
            
            # If mapping is confirmed, don't show the interface again
            if result:
                self.session.set('mapping_confirmed', True)
            
            return result
                
        except Exception as e:
            self.view.show_message(f"Error in mapping: {str(e)}", "error")
            return False
        
    def _display_mapping_interface(self):
        """Display interface for verifying and modifying field mappings"""
        self.view.display_subheader("AI Suggested Critical Field Mappings")
        
        # Get current mappings and available columns
        suggested_mappings = self.session.get('suggested_mappings', {})  # Default to empty dict
        current_mappings = self.session.get('field_mappings', {})  # Default to empty dict
        df = self.session.get('df')
        
        if df is None:
            self.view.show_message("❌ No data found. Please upload data first.", "error")
            return False
            
        all_columns = list(df.columns)
        mapping_confirmed = self.session.get('mapping_confirmed', False)
        
        # Check for mandatory fields in AI suggestions
        mandatory_fields = ['cust_id']
        missing_mandatory = []
        if suggested_mappings:  # Only check if we have suggestions
            missing_mandatory = [
                field for field in mandatory_fields 
                if field not in suggested_mappings or not suggested_mappings[field]
            ]
        
        if missing_mandatory or not suggested_mappings:  # If no suggestions or missing fields
            missing_fields = ', '.join(missing_mandatory) if missing_mandatory else "required fields"
            self.view.show_message(
                f"⚠️ AI couldn't find mapping for {missing_fields}. "
                "Please map these fields manually.",
                "warning"
            )
            # Automatically switch to manual mapping
            return self._handle_manual_mapping(all_columns, current_mappings)
        
        # Format message with consistent field names
        message = "🎯 AI Suggested Mappings:\n"
        field_display_names = {
            "cust_id": "Customer ID",
            "target": "Target",
            "revenue": "Revenue",
            "date": "Date",
            "prod_id": "Product ID"
        }
        
        for field, mapped_col in suggested_mappings.items():
            display_name = field_display_names.get(field, field)
            message += f"- {display_name}:  **{mapped_col or 'Not Found'}**\n"
        
        self.view.show_message(message, "info")
        self.view.display_markdown("---")
        
        # Only show options if mapping not yet confirmed
        if not mapping_confirmed:
            # Show options to proceed
            action = self.view.radio_select(
                "How would you like to proceed?",
                options=[
                    "Use AI Recommended Mappings",
                    "Select Columns Manually"
                ],
                key="mapping_choice"
            )
            
            if action == "Use AI Recommended Mappings":
                if self.view.display_button("Confirm Mappings"):
                    self.session.set('field_mappings', suggested_mappings)
                    self.session.set('mapping_confirmed', True)
                    return False
            else:  # Select Columns Manually
                return self._handle_manual_mapping(all_columns, current_mappings)
        else:
            # Get analysis type after mapping confirmation
            return self._get_analysis_type(all_columns)
        
        return False
        
    def _handle_manual_mapping(self, all_columns, current_mappings):
        """Handle manual column mapping selection"""
        # Only show mapping options if not confirmed
        required_fields = ["cust_id"]
        optional_fields = ["target", "date", "prod_id", "revenue"]
        
        modified_mappings = {}
        suggested_mappings = self.session.get('suggested_mappings', {})
        
        field_display_names = {
            "cust_id": "Customer ID",
            "target": "Target",
            "revenue": "Revenue",
            "date": "Date",
            "prod_id": "Product ID"
        }
        
        self.view.display_subheader("Required Fields")
        for field in required_fields:
            # Use suggested mapping as default if available
            default_value = suggested_mappings.get(field) or current_mappings.get(field) or ""
            modified_mappings[field] = self.view.select_box(
                f"Select column for {field_display_names[field]}",
                options=[""] + all_columns,
                default=default_value
            )
        
        self.view.display_subheader("Optional Fields")
        for field in optional_fields:
            # Use suggested mapping as default if available
            default_value = suggested_mappings.get(field) or current_mappings.get(field) or ""
            value = self.view.select_box(
                f"Select column for {field_display_names[field]} (optional)",
                options=[""] + all_columns,
                default=default_value
            )
            if value:
                modified_mappings[field] = value
        
        if self.view.display_button("Confirm Mappings"):
            target_mapped = bool(modified_mappings.get('target'))
            date_mapped = bool(modified_mappings.get('date'))
            
            if not target_mapped and not date_mapped:
                self.view.show_message(
                    "❌ Please map either target or date column to proceed.",
                    "error"
                )
                return False
            
            self.session.set('field_mappings', modified_mappings)
            self.session.set('mapping_confirmed', True)
            
        return False

    def _get_analysis_type(self, all_columns):
        """Determine analysis type based on confirmed mappings"""
        mappings = self.session.get('field_mappings')
        df = self.session.get('df')  # Get dataframe to check numeric columns
        
        # Filter numeric columns
        numeric_columns = [
            col for col in all_columns 
            if col != mappings.get('date') and 
            pd.api.types.is_numeric_dtype(df[col])
        ]
        
        # Add field display names dictionary
        field_display_names = {
            "cust_id": "Customer ID",
            "target": "Target",
            "forecasting_field": "Forecasting Field",
            "revenue": "Revenue",
            "date": "Date",
            "prod_id": "Product ID"
        }
        
        # Display confirmed mappings
        self.view.display_subheader("Confirmed Mappings")
        mapping_msg = "\n".join(
            [f"- {field_display_names[k]}: **{v}**" 
             for k, v in mappings.items()]
        )
        self.view.show_message(mapping_msg, "success")
        
        # Determine analysis type based on mappings
        target_mapped = bool(mappings.get('target'))
        date_mapped = bool(mappings.get('date'))
        
        self.view.display_subheader("Analysis Type Selection")
        
        if target_mapped and date_mapped:
            problem_type = self.view.radio_select(
                "What type of analysis would you like to perform?",
                options=[
                    "Regression",
                    "Forecasting"
                ],
                key="problem_type_choice"
            )
            
            if problem_type == "Forecasting":
                forecast_field = self.view.select_box(
                    "Select field to forecast (numeric fields only)",
                    options=["None"] + numeric_columns,
                    key="forecast_field"
                )
                
                # Only proceed if a valid field is selected
                if forecast_field == "None":
                    self.view.show_message(
                        "⚠️ Please select a field to forecast",
                        "warning"
                    )
                    return False
                
                # Add forecast periods dropdown
                forecast_periods = self.view.select_box(
                    "How many periods do you want to forecast?",
                    options=[3, 4, 5, 6],
                    key="forecast_periods"
                )
                
                mappings['forecasting_field'] = forecast_field  # New field for forecasting
                self.session.set('is_forecasting', True)
                self.session.set('forecast_periods', forecast_periods)  # Save forecast periods
            else:
                self.session.set('is_forecasting', False)
                
        elif target_mapped:
            self.view.show_message(
                "ℹ️ Proceeding with regression analysis since only target is mapped.",
                "info"
            )
            self.session.set('is_forecasting', False)
            
        else:  # only date_mapped
            self.view.show_message(
                "ℹ️ Since only date is mapped, this will be a forecasting analysis.",
                "info"
            )
            forecast_field = self.view.select_box(
                "Select field to forecast (numeric fields only)",
                options=["None"] + numeric_columns,
                key="forecast_field"
            )
            
            # Only proceed if a valid field is selected
            if forecast_field == "None":
                self.view.show_message(
                    "⚠️ Please select a field to forecast",
                    "warning"
                )
                return False
            
            # Add forecast periods dropdown
            forecast_periods = self.view.select_box(
                "How many periods do you want to forecast?",
                options=[3, 4, 5, 6],
                key="forecast_periods"
            )
            
            mappings['forecasting_field'] = forecast_field  # Use forecasting_field instead of target
            self.session.set('is_forecasting', True)
            self.session.set('forecast_periods', forecast_periods)  # Save forecast periods
        
        if self.view.display_button("✅ Proceed with Analysis"):
            self.session.set('field_mappings', mappings)
            self.session.set('mapping_complete', True)
            self.session.set('step_2_complete', True)
            self._save_step_summary()
            return True
            
        return False
        
    def _handle_data_types(self):
        """Handle data type conversions"""
        # try:
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        target_processed = self.session.get('target_classification_complete', False)
        
        with self.view.display_spinner('Converting data types...'):
            # Make a copy to avoid modifying original
            modified_df = df.copy()
            
            # 1. Convert ID column to text
            id_col = mappings.get('cust_id')
            if id_col:
                modified_df[id_col] = modified_df[id_col].astype(str)
            
            # 2. Convert date column to YYYY-MM format
            date_col = mappings.get('date')
            if date_col:
                try:
                    modified_df[date_col] = pd.to_datetime(modified_df[date_col])
                    modified_df[date_col] = modified_df[date_col].dt.strftime('%Y-%m')
                except Exception as e:
                    self.view.show_message(f"Error converting date column: {str(e)}", "error")
                    return False
            
            # 3. Check target values (if present)
            target_col = mappings.get('target')
            if target_col and not target_processed:  # Only process if not already done
                target_handling_result = self._handle_target_classification(modified_df, target_col)
                if isinstance(target_handling_result, bool) and not target_handling_result:
                    return False
                modified_df = target_handling_result
                # Update session DataFrame immediately after target classification
                self.session.set('df', modified_df)

        # Display modified data for confirmation
        self.view.display_subheader("Modified Data Types")
        self.view.display_dataframe(modified_df.head())
        
        # Show data type information
        self.view.display_subheader("Data Types Summary")
        dtypes_msg = "Updated Data Types:\n"
        for col, dtype in modified_df.dtypes.items():
            if col in mappings.values():
                dtypes_msg += f"- {col}: **{dtype}**\n"
        self.view.show_message(dtypes_msg, "info")
        
        if self.view.display_button("Confirm Data Types"):
            self.session.set('df', modified_df)  # Update again to be safe
            self.session.set('data_types_complete', True)
            return True
                
        # except Exception as e:
        #     self.view.show_message(f"Error in data type conversion: {str(e)}", "error")
        # return False
        
    def _handle_target_classification(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Handle target column validation for regression"""
        # Get target values excluding NaN/None/null
        target_values = df[target_col].dropna()
        
        # Check if target values are numeric
        if not pd.api.types.is_numeric_dtype(target_values):
            self.view.show_message(
                f"❌ Target column '{target_col}' must contain numeric values for regression. "
                "Please select another column with numeric values.",
                "error"
            )
            return False
        
        # Show value distribution statistics
        stats_msg = f"Target distribution in '{target_col}':\n"
        stats = {
            'Mean': target_values.mean(),
            'Std Dev': target_values.std(),
            'Min': target_values.min(),
            'Max': target_values.max(),
            'Median': target_values.median()
        }
        
        for stat_name, value in stats.items():
            stats_msg += f"- {stat_name}: {value:.2f}\n"
        
        # Show null count if any
        null_count = df[target_col].isna().sum()
        if null_count > 0:
            percentage = (null_count / len(df)) * 100
            stats_msg += f"- Missing/NaN: {null_count} records ({percentage:.1f}%) - Will be used for inference\n"
        
        self.view.show_message(stats_msg, "info")
        
        # No need for binary conversion, return original dataframe
        self.session.set('target_classification_complete', True)
        return df
        
    def _apply_target_encoding(self, df: pd.DataFrame, target_col: str, 
                              class_0_values: List, class_1_values: List) -> pd.DataFrame:
        """This method is not needed for regression"""
        return df
        
    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        mappings = self.session.get('field_mappings')
        df = self.session.get('df')
        field_display_names = {
            "cust_id": "Customer ID",
            "target": "Target",
            "forecasting_field": "Forecasting Field",  # Add new display name
            "revenue": "Revenue",
            "date": "Date",
            "prod_id": "Product ID"
        }
        
        summary = "✅ Step 2 Complete\n\n"
        
        # Add analysis type confirmation
        problem_type = "Forecasting" if self.session.get('is_forecasting') else "Regression"
        summary += f"**Confirmed Analysis Type:** {problem_type}\n\n"
        if problem_type == "Forecasting":
            forecast_periods = self.session.get('forecast_periods', 3)  # Default to 3 if None
            summary += f"- Forecasting Field: **{mappings.get('forecasting_field')}**\n"
            summary += f"- Time Column: **{mappings.get('date')}**\n"
            summary += f"- Forecast Periods: **{forecast_periods} months**\n\n"
        
        # Add field mappings with data types
        summary += "Field Mappings:\n"
        for field, col in mappings.items():
            if col:  # Only show mapped fields
                dtype = df[col].dtype if col else None
                display_name = field_display_names.get(field, field)
                summary += f"- {display_name}: **{col}** ({dtype})\n"
            
        self.session.set('step_2_summary', summary) 