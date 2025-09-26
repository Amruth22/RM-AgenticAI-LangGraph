"""Main Streamlit application for RM-AgenticAI-LangGraph system."""

import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# Configure page
st.set_page_config(
    page_title="🤖 AI-Powered Investment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
from config.settings import get_settings
from config.logging_config import setup_logging, get_logger
from langraph_agents.workflows.prospect_analysis_workflow import ProspectAnalysisWorkflow
from langraph_agents.state_models import WorkflowState

# Initialize
settings = get_settings()
setup_logging()
logger = get_logger("MainApp")

# Initialize workflow
@st.cache_resource
def get_workflow():
    """Initialize and cache the workflow."""
    return ProspectAnalysisWorkflow()

# Load data
@st.cache_data
def load_prospects():
    """Load prospects data."""
    try:
        df = pd.read_csv(settings.prospects_csv)
        df["label"] = df["prospect_id"] + " - " + df["name"]
        return df
    except Exception as e:
        logger.error(f"Failed to load prospects: {str(e)}")
        # Create dummy data for demo
        return pd.DataFrame([
            {
                "prospect_id": "P001",
                "name": "John Doe",
                "age": 35,
                "annual_income": 800000,
                "current_savings": 500000,
                "target_goal_amount": 2000000,
                "investment_horizon_years": 10,
                "number_of_dependents": 2,
                "investment_experience_level": "Intermediate",
                "investment_goal": "Retirement Planning",
                "label": "P001 - John Doe"
            },
            {
                "prospect_id": "P002",
                "name": "Jane Smith",
                "age": 28,
                "annual_income": 1200000,
                "current_savings": 300000,
                "target_goal_amount": 5000000,
                "investment_horizon_years": 15,
                "number_of_dependents": 0,
                "investment_experience_level": "Advanced",
                "investment_goal": "Wealth Creation",
                "label": "P002 - Jane Smith"
            }
        ])

async def analyze_prospect_async(workflow: ProspectAnalysisWorkflow, prospect_data: Dict[str, Any]) -> WorkflowState:
    """Async wrapper for prospect analysis."""
    return await workflow.analyze_prospect(prospect_data)

def run_analysis(workflow: ProspectAnalysisWorkflow, prospect_data: Dict[str, Any]) -> WorkflowState:
    """Run prospect analysis synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(analyze_prospect_async(workflow, prospect_data))

def display_analysis_results(state: WorkflowState):
    """Display comprehensive analysis results."""
    
    # Execution Summary
    st.subheader("🔄 Execution Summary")
    exec_summary = state.get_execution_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", exec_summary['total_executions'])
    with col2:
        st.metric("Completed", exec_summary['completed'])
    with col3:
        st.metric("Success Rate", f"{exec_summary['success_rate']:.1%}")
    with col4:
        st.metric("Total Time", f"{exec_summary['total_execution_time']:.2f}s")
    
    # Analysis Results
    st.subheader("📊 Analysis Results")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        if state.analysis.risk_assessment:
            st.markdown("**🎯 Risk Assessment**")
            risk = state.analysis.risk_assessment
            st.write(f"**Risk Level:** `{risk.risk_level}`")
            st.write(f"**Confidence:** {risk.confidence_score:.1%}")
            
            if risk.risk_factors:
                st.write("**Risk Factors:**")
                for factor in risk.risk_factors[:3]:
                    st.write(f"• {factor}")
        
        if state.prospect.data_quality_score:
            st.markdown("**📈 Data Quality**")
            quality_score = state.prospect.data_quality_score
            st.progress(quality_score)
            st.write(f"Quality Score: {quality_score:.1%}")
    
    with analysis_col2:
        if state.analysis.persona_classification:
            st.markdown("**👤 Persona Classification**")
            persona = state.analysis.persona_classification
            st.write(f"**Persona:** `{persona.persona_type}`")
            st.write(f"**Confidence:** {persona.confidence_score:.1%}")
            
            if persona.behavioral_insights:
                st.write("**Key Insights:**")
                for insight in persona.behavioral_insights[:3]:
                    st.write(f"• {insight}")
    
    # Product Recommendations
    if state.recommendations.recommended_products:
        st.subheader("💼 Product Recommendations")
        
        # Create recommendations dataframe
        rec_data = []
        for rec in state.recommendations.recommended_products:
            rec_data.append({
                "Product": rec.product_name,
                "Type": rec.product_type,
                "Suitability": f"{rec.suitability_score:.1%}",
                "Risk Level": rec.risk_alignment,
                "Expected Returns": rec.expected_returns or "N/A",
                "Fees": rec.fees or "N/A"
            })
        
        rec_df = pd.DataFrame(rec_data)
        st.dataframe(rec_df, use_container_width=True)
        
        # Justification
        if state.recommendations.justification_text:
            st.markdown("**🎯 Recommendation Justification**")
            st.info(state.recommendations.justification_text)
    
    # Key Insights and Action Items
    col1, col2 = st.columns(2)
    
    with col1:
        if state.key_insights:
            st.subheader("💡 Key Insights")
            for insight in state.key_insights:
                st.write(f"• {insight}")
    
    with col2:
        if state.action_items:
            st.subheader("✅ Action Items")
            for action in state.action_items:
                st.write(f"• {action}")

def display_agent_performance(state: WorkflowState):
    """Display agent performance metrics."""
    st.subheader("🤖 Agent Performance")
    
    if state.agent_executions:
        perf_data = []
        for execution in state.agent_executions:
            perf_data.append({
                "Agent": execution.agent_name,
                "Status": execution.status.title(),
                "Execution Time": f"{execution.execution_time:.2f}s" if execution.execution_time else "N/A",
                "Start Time": execution.start_time.strftime("%H:%M:%S"),
                "End Time": execution.end_time.strftime("%H:%M:%S") if execution.end_time else "Running"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)

def main():
    """Main application."""
    
    # Header
    st.title("🤖 AI-Powered Investment Analyzer")
    st.markdown("**Advanced Multi-Agent System for Financial Advisory**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Workflow info
        workflow = get_workflow()
        workflow_summary = workflow.get_workflow_summary()
        
        st.subheader("Workflow Overview")
        st.write(f"**Agents:** {len(workflow_summary['agents'])}")
        st.write(f"**Steps:** {len(workflow_summary['steps'])}")
        
        with st.expander("View Agents"):
            for agent in workflow_summary['agents']:
                st.write(f"• {agent}")
        
        # Settings
        st.subheader("Analysis Settings")
        show_performance = st.checkbox("Show Agent Performance", value=True)
        show_execution_details = st.checkbox("Show Execution Details", value=False)
    
    # Main content
    prospects_df = load_prospects()
    
    st.markdown("### 👥 Select Prospect for Analysis")
    selected_label = st.selectbox(
        "Choose a prospect to analyze:",
        ["Select a Prospect"] + list(prospects_df["label"]),
        key="prospect_selector"
    )
    
    if selected_label != "Select a Prospect":
        selected_row = prospects_df[prospects_df["label"] == selected_label].iloc[0]
        
        # Display prospect info
        st.success(f"📋 Selected Prospect: **{selected_row['name']}**")
        
        with st.expander("View Prospect Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Age:** {selected_row['age']}")
                st.write(f"**Annual Income:** ₹{selected_row['annual_income']:,}")
                st.write(f"**Current Savings:** ₹{selected_row['current_savings']:,}")
            with col2:
                st.write(f"**Target Amount:** ₹{selected_row['target_goal_amount']:,}")
                st.write(f"**Investment Horizon:** {selected_row['investment_horizon_years']} years")
                st.write(f"**Experience Level:** {selected_row['investment_experience_level']}")
        
        # Analysis button
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            
            with st.spinner("🤖 Running Multi-Agent Analysis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Convert to dict for analysis
                    prospect_data = selected_row.to_dict()
                    
                    # Update progress
                    progress_bar.progress(20)
                    status_text.text("Initializing agents...")
                    
                    # Run analysis
                    workflow = get_workflow()
                    
                    progress_bar.progress(40)
                    status_text.text("Analyzing prospect data...")
                    
                    # Execute workflow
                    result_state = run_analysis(workflow, prospect_data)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis completed!")
                    
                    # Store in session state
                    st.session_state['analysis_result'] = result_state
                    st.session_state['analysis_timestamp'] = datetime.now()
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Analysis failed: {str(e)}")
                    return
        
        # Display results if available
        if 'analysis_result' in st.session_state:
            st.markdown("---")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "🤖 Agent Performance", "💬 Chat Assistant"])
            
            with tab1:
                display_analysis_results(st.session_state['analysis_result'])
            
            with tab2:
                if show_performance:
                    display_agent_performance(st.session_state['analysis_result'])
                else:
                    st.info("Enable 'Show Agent Performance' in the sidebar to view metrics.")
            
            with tab3:
                st.subheader("💬 RM Chat Assistant")
                st.info("🚧 Interactive chat functionality coming soon!")
                
                # Placeholder for chat interface
                user_query = st.text_input("Ask a question about this analysis:")
                if user_query:
                    st.write("🤖 **Assistant:** This feature will be available in the next update.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**🤖 RM-AgenticAI-LangGraph** | "
        "Powered by LangGraph Multi-Agent System | "
        f"Built with ❤️ for Financial Advisory"
    )

if __name__ == "__main__":
    main()