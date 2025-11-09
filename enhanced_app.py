import streamlit as st
import time
from classifier.llm_classifier import classify_query
from pipelines import (
    memorization_pipeline,
    understanding_pipeline,
    problem_solving_pipeline,
    reasoning_taa_pipeline,
    reasoning_ate_pipeline,
)

# Page configuration
st.set_page_config(
    page_title="Cyber Threat Intelligence Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 4px solid #ff6b6b;
        padding-left: 1rem;
    }
    .pipeline-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .pipeline-card:hover {
        transform: translateY(-5px);
    }
    .security-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.1rem;
    }
    .analysis-result {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .pipeline-indicator {
        background: linear-gradient(45deg, #11998e, #38ef7d);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .template-btn {
        background: linear-gradient(45deg, #11998e, #38ef7d);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .template-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with pipeline information
with st.sidebar:
    st.markdown("## 🛡️ CTI Assistant")
    st.markdown("---")
    
    st.markdown("### 🔍 Available Pipelines")
    
    # Pipeline descriptions
    pipelines_info = {
        "🧠 Memorization": "Direct information retrieval and fact-based responses",
        "💭 Understanding": "Comprehension and explanation of complex concepts",
        "🔧 Problem Solving": "Analytical solutions and step-by-step reasoning",
        "🎯 Reasoning TAA": "Threat actor analysis and attribution",
        "⚡ Reasoning ATE": "Attack technique evaluation and mapping"
    }
    
    for pipeline, description in pipelines_info.items():
        with st.expander(pipeline):
            st.caption(description)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.info("""
    **Current Session:**
    - Queries Processed: 0
    - Active Pipeline: None
    - Status: Ready
    """)

# Main content
st.markdown('<div class="main-header">🛡️ Cyber Threat Intelligence Assistant</div>', unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2 = st.tabs(["🔍 Threat Analysis", "📊 Pipeline Visualizer"])

with tab1:
    # Two-column layout for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">🎯 Task / Instruction</div>', unsafe_allow_html=True)
        task = st.text_area(
            "Task / Instruction",
            height=120,
            placeholder="Describe what you want the assistant to do. Example: 'Map the CVE to CWE and justify briefly.'",
            help="Give a single instruction or role — e.g., 'Analyze the CVE', 'Classify the threat actor', or 'Calculate CVSS'."
        )
        
        # Quick task templates
        st.markdown("**🚀 Quick Templates:**")
        template_cols = st.columns(2)
        with template_cols[0]:
            if st.button("📋 CVE Analysis", use_container_width=True):
                st.session_state.task_template = "Analyze this CVE and provide mitigation recommendations."
        with template_cols[1]:
            if st.button("🎯 Threat Mapping", use_container_width=True):
                st.session_state.task_template = "Map this threat to MITRE ATT&CK framework."
        
        template_cols2 = st.columns(2)
        with template_cols2[0]:
            if st.button("📈 Risk Assessment", use_container_width=True):
                st.session_state.task_template = "Perform risk assessment and calculate CVSS score."
        with template_cols2[1]:
            if st.button("🔍 IOC Analysis", use_container_width=True):
                st.session_state.task_template = "Analyze these indicators of compromise and provide context."
        
        if 'task_template' in st.session_state:
            task = st.text_area("Task / Instruction", value=st.session_state.task_template, height=120)

    with col2:
        st.markdown('<div class="sub-header">📋 Context / Description</div>', unsafe_allow_html=True)
        context = st.text_area(
            "Context / Description",
            height=200,
            placeholder="Paste the CVE description, threat report, or security context here...",
            help="Provide the detailed input text that will be analyzed by the appropriate pipeline."
        )
    
    # Submit button with enhanced styling
    if st.button("🚀 Analyze Threat", use_container_width=True, type="primary"):
        if not task.strip() or not context.strip():
            st.warning("⚠️ Please fill in both the task/instruction and context/description fields.")
        else:
            # Create a progress container
            progress_container = st.container()
            
            with progress_container:
                # Step 1: Classification
                st.markdown("### 🔍 Step 1: Query Classification")
                classification_progress = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    classification_progress.progress(i + 1)
                    status_text.text(f"Analyzing query intent... {i+1}%")
                    time.sleep(0.01)
                
                with st.spinner("🤖 Classifying query type..."):
                    query_type = classify_query(task, context)
                
                classification_progress.empty()
                status_text.empty()
                
                # Display classification result with animation
                st.success(f"✅ Query classified as: **{query_type.replace('_', ' ').title()}**")
                
                # Step 2: Pipeline Selection Visualization
                st.markdown("### ⚡ Step 2: Pipeline Selection")
                
                # Show pipeline selection animation
                pipeline_placeholder = st.empty()
                pipelines = ["memorization", "understanding", "problem_solving", "reasoning_taa", "reasoning_ate"]
                
                for pipeline in pipelines:
                    pipeline_placeholder.markdown(
                        f'<div style="padding: 1rem; margin: 0.5rem 0; border-radius: 10px; '
                        f'background: {"#4CAF50" if pipeline == query_type else "#f0f0f0"}; '
                        f'color: {"white" if pipeline == query_type else "black"}; '
                        f'transition: all 0.5s ease;">'
                        f'🔍 Evaluating: {pipeline.replace("_", " ").title()}'
                        f'{" ✅ SELECTED" if pipeline == query_type else ""}'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.3) if pipeline != query_type else time.sleep(0.5)
                
                time.sleep(1)
                pipeline_placeholder.empty()
                
                st.markdown(f'<div class="pipeline-indicator">Active Pipeline: {query_type.replace("_", " ").title()}</div>', unsafe_allow_html=True)
                
                # Step 3: Pipeline Execution
                st.markdown("### 🚀 Step 3: Pipeline Execution")
                execution_progress = st.progress(0)
                execution_status = st.empty()
                
                for i in range(100):
                    execution_progress.progress(i + 1)
                    execution_status.text(f"Executing {query_type} pipeline... {i+1}%")
                    time.sleep(0.02)
                
                with st.spinner(f"🔄 Running {query_type.replace('_', ' ').title()} pipeline..."):
                    if query_type == "memorization":
                        response = memorization_pipeline.run(context)
                    elif query_type == "understanding":
                        response = understanding_pipeline.run(context)
                    elif query_type == "problem_solving":
                        response = problem_solving_pipeline.run(context)
                    elif query_type == "reasoning_taa":
                        response = reasoning_taa_pipeline.run(context)
                    elif query_type == "reasoning_ate":
                        response = reasoning_ate_pipeline.run(context)
                    else:
                        response = "[ERROR] Unknown query type returned by classifier."
                
                execution_progress.empty()
                execution_status.empty()
                
                # Display final results
                # st.balloons()
                # Security shield animation
                st.markdown("""
                <div style='text-align: center; padding: 1rem;'>
                    <h3 style='color: #28a745;'>🛡️ Threat Analysis Secured!</h3>
                    <p>All pipelines executed successfully</p>
                </div>
                """, unsafe_allow_html=True)

                # Lock icon completion
                st.markdown("""
                <div style='text-align: center; font-size: 3rem;'>
                    🔓→🔒
                </div>
                <h4 style='text-align: center; color: #28a745;'>Analysis Locked & Complete</h4>
                """, unsafe_allow_html=True)

                st.success("🎉 Analysis Complete!")  
                # Results section
                st.markdown("### 📊 Analysis Results")
                st.markdown(f'<div class="analysis-result">{response}</div>', unsafe_allow_html=True)
                
                # Security context badges
                st.markdown("### 🏷️ Analysis Context")
                badge_cols = st.columns(5)
                badges = ["Threat Intel", "CVE Analysis", "Risk Assessment", "IOC", "Mitigation"]
                for i, badge in enumerate(badges):
                    with badge_cols[i]:
                        st.markdown(f'<div class="security-badge">{badge}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="sub-header">🔍 Pipeline Architecture</div>', unsafe_allow_html=True)
    
    # Pipeline visualization
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 1rem 0;">
            <h3>🛡️ Context-Aware RAG Pipeline</h3>
            <p>Dynamic pipeline selection based on query classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pipeline flow diagram
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="text-align: center;">
                <div style="margin: 1rem 0; padding: 1rem; background: #e3f2fd; border-radius: 8px;">
                    <strong>Input Query + Context</strong>
                </div>
                <div style="margin: 1rem 0;">⬇️</div>
                <div style="margin: 1rem 0; padding: 1rem; background: #fff3e0; border-radius: 8px;">
                    <strong>LLM Classifier</strong>
                </div>
                <div style="margin: 1rem 0;">⬇️</div>
                <div style="margin: 1rem 0; padding: 1rem; background: #e8f5e8; border-radius: 8px;">
                    <strong>Pipeline Selection</strong>
                </div>
                <div style="margin: 1rem 0;">⬇️</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin: 1rem 0;">
                    <div style="padding: 0.5rem; background: #fce4ec; border-radius: 5px;">🧠 Memorization</div>
                    <div style="padding: 0.5rem; background: #f3e5f5; border-radius: 5px;">💭 Understanding</div>
                    <div style="padding: 0.5rem; background: #e8eaf6; border-radius: 5px;">🔧 Problem Solving</div>
                    <div style="padding: 0.5rem; background: #e0f2f1; border-radius: 5px;">🎯 Reasoning TAA</div>
                    <div style="padding: 0.5rem; background: #fff8e1; border-radius: 5px; grid-column: span 2;">⚡ Reasoning ATE</div>
                </div>
                <div style="margin: 1rem 0;">⬇️</div>
                <div style="margin: 1rem 0; padding: 1rem; background: #f1f8e9; border-radius: 8px;">
                    <strong>Context-Aware Response</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🛡️ Cyber Threat Intelligence Assistant | Context-Aware RAG System | Secure Analysis"
    "</div>",
    unsafe_allow_html=True
)