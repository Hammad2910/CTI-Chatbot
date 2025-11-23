import streamlit as st
import time
import random
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
    page_title="CTA RAG - Cognitive Task-Aware Retrieval-Augmented Generation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional cyber theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5ff 0%, #0077ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
    }
    
    .sub-header {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.3rem;
        color: #00f5ff;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #0077ff;
        padding-bottom: 0.5rem;
    }
    
    .cyber-container {
        background: rgba(8, 15, 30, 0.9);
        border: 1px solid rgba(0, 119, 255, 0.4);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 119, 255, 0.1);
    }
    
    .cyber-input {
        background: rgba(5, 10, 25, 0.8) !important;
        border: 1px solid rgba(0, 119, 255, 0.5) !important;
        border-radius: 6px !important;
        color: #e8f4ff !important;
        font-family: 'Exo 2', sans-serif !important;
        padding: 12px !important;
    }
    
    .cyber-input:focus {
        border: 1px solid #00f5ff !important;
        box-shadow: 0 0 8px rgba(0, 245, 255, 0.3) !important;
    }
    
    .pipeline-card {
        background: linear-gradient(135deg, rgba(8, 20, 40, 0.9) 0%, rgba(12, 25, 50, 0.9) 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(0, 119, 255, 0.3);
        transition: all 0.3s ease;
        font-family: 'Exo 2', sans-serif;
    }
    
    .pipeline-card:hover {
        border: 1px solid rgba(0, 245, 255, 0.6);
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .pipeline-active {
        background: linear-gradient(135deg, rgba(0, 50, 100, 0.9) 0%, rgba(0, 80, 120, 0.9) 100%);
        border: 1px solid #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
    }
    
    .cyber-button {
        background: linear-gradient(135deg, #0077ff 0%, #00aaff 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .cyber-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 119, 255, 0.4) !important;
    }
    
    .analysis-result {
        background: rgba(8, 15, 30, 0.9);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0077ff;
        margin-top: 1rem;
        color: #e8f4ff;
        font-family: 'Exo 2', sans-serif;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pipeline-indicator {
        background: linear-gradient(135deg, #0077ff, #00aaff);
        color: #fff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .agent-flow {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        position: relative;
    }
    
    .flow-node {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: rgba(0, 119, 255, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        z-index: 2;
        position: relative;
        border: 2px solid rgba(0, 245, 255, 0.5);
    }
    
    .flow-connector {
        height: 2px;
        background: linear-gradient(90deg, #0077ff, #00aaff);
        flex-grow: 1;
        position: relative;
        z-index: 1;
    }
    
    .flow-connector::after {
        content: '';
        position: absolute;
        top: -3px;
        right: 0;
        width: 8px;
        height: 8px;
        background: #00f5ff;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.2); }
    }
    
    .system-status {
        background: rgba(8, 15, 30, 0.9);
        border: 1px solid rgba(0, 119, 255, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00ff88;
        margin-right: 8px;
        animation: status-pulse 2s infinite;
    }
    
    @keyframes status-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0077ff, #00aaff);
    }
    
    /* Background styling */
    .cyber-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        background: linear-gradient(135deg, #0a0f1e 0%, #151a30 50%, #0a0f1e 100%);
    }
    
    .cyber-grid {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 119, 255, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 119, 255, 0.05) 1px, transparent 1px);
        background-size: 40px 40px;
    }
</style>

<div class="cyber-bg">
    <div class="cyber-grid"></div>
</div>
""", unsafe_allow_html=True)

# Sidebar with system information
with st.sidebar:
    st.markdown("## 🧠 CTA RAG System")
    st.markdown("---")
    
    st.markdown("### Active Pipelines")
    
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
    st.markdown("### System Status")
    
    # System status
    st.markdown("""
    <div class="system-status">
        <div><span class="status-indicator"></span>System: <strong>Online</strong></div>
        <div>Active Agents: <strong>5/5</strong></div>
        <div>Processing: <strong>Ready</strong></div>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown('<div class="main-header">CTA RAG</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-family: Exo 2, sans-serif; color: #00aaff; font-size: 1.1rem; margin-bottom: 2rem;">Cognitive Task-Aware Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

# Two-column layout for input
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">Instruction</div>', unsafe_allow_html=True)
    task = st.text_area(
        "Instruction",
        height=120,
        placeholder="Describe the task or instruction for the AI system...",
        help="Provide clear instructions for the cognitive task",
        key="instruction_input"
    )

with col2:
    st.markdown('<div class="sub-header">Context / Description</div>', unsafe_allow_html=True)
    context = st.text_area(
        "Context / Description",
        height=200,
        placeholder="Provide the context, data, or description for analysis...",
        help="Input text that will be processed by the appropriate pipeline",
        key="context_input"
    )

# Submit button
if st.button("🚀 Execute CTA RAG Analysis", use_container_width=True, type="primary"):
    if not task.strip() or not context.strip():
        st.warning("⚠️ Please fill in both the instruction and context fields.")
    else:
        # Create a progress container
        progress_container = st.container()
        
        with progress_container:
            # Step 1: Classification
            st.markdown("### Step 1: Query Classification")
            
            # Agent flow visualization
            st.markdown("""
            <div class="agent-flow">
                <div class="flow-node">Input</div>
                <div class="flow-connector"></div>
                <div class="flow-node">Classify</div>
                <div class="flow-connector"></div>
                <div class="flow-node">Process</div>
            </div>
            """, unsafe_allow_html=True)
            
            classification_progress = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                classification_progress.progress(i + 1)
                status_text.text(f"Analyzing cognitive task requirements... {i+1}%")
                time.sleep(0.01)
            
            with st.spinner("Classifying query type..."):
                query_type = classify_query(task, context)
            
            classification_progress.empty()
            status_text.empty()
            
            # Display classification result
            st.success(f"✅ Query classified as: **{query_type.replace('_', ' ').title()}**")
            
            # Step 2: Pipeline Selection
            st.markdown("### Step 2: Pipeline Selection")
            
            # Show pipeline selection
            pipeline_placeholder = st.empty()
            pipelines = ["memorization", "understanding", "problem_solving", "reasoning_taa", "reasoning_ate"]
            pipeline_names = {
                "memorization": "🧠 Memorization",
                "understanding": "💭 Understanding", 
                "problem_solving": "🔧 Problem Solving",
                "reasoning_taa": "🎯 Reasoning TAA",
                "reasoning_ate": "⚡ Reasoning ATE"
            }
            
            for pipeline in pipelines:
                is_active = pipeline == query_type
                
                pipeline_placeholder.markdown(
                    f'<div class="pipeline-card {"pipeline-active" if is_active else ""}">'
                    f'<h4 style="color: {"#00f5ff" if is_active else "#e8f4ff"}; margin: 0; font-family: Exo 2, sans-serif;">{pipeline_names[pipeline]}</h4>'
                    f'<p style="margin: 0.5rem 0 0 0; color: #b8d4ff; font-size: 0.9rem;">{"✓ Active Pipeline" if is_active else "Standby"}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                time.sleep(0.2)
            
            time.sleep(0.5)
            pipeline_placeholder.empty()
            
            st.markdown(f'<div class="pipeline-indicator">Active Pipeline: {pipeline_names[query_type]}</div>', unsafe_allow_html=True)
            
            # Step 3: Pipeline Execution
            st.markdown("### Step 3: Pipeline Execution")
            
            execution_progress = st.progress(0)
            execution_status = st.empty()
            
            for i in range(100):
                execution_progress.progress(i + 1)
                execution_status.text(f"Executing {pipeline_names[query_type]} pipeline... {i+1}%")
                time.sleep(0.02)
            
            with st.spinner(f"Running {pipeline_names[query_type]} pipeline..."):
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
            st.markdown("""
            <div style='text-align: center; padding: 1rem; background: rgba(8, 15, 30, 0.9); border-radius: 8px; border: 1px solid rgba(0, 119, 255, 0.4);'>
                <h3 style='color: #00f5ff; font-family: Exo 2, sans-serif;'>Analysis Complete</h3>
                <p style='color: #b8d4ff;'>CTA RAG system processing finished</p>
            </div>
            """, unsafe_allow_html=True)

            # Results section
            st.markdown("### Analysis Results")
            st.markdown(f'<div class="analysis-result">{response}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #00aaff; font-family: Exo 2, sans-serif; font-size: 0.9rem;'>"
    "CTA RAG System | Cognitive Task-Aware Retrieval-Augmented Generation | Advanced AI Architecture"
    "</div>",
    unsafe_allow_html=True
)