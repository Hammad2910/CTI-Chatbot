import streamlit as st
from classifier.llm_classifier import classify_query
from pipelines import (
    memorization_pipeline,
    understanding_pipeline,
    problem_solving_pipeline,
    reasoning_taa_pipeline,
    reasoning_ate_pipeline,
)

st.set_page_config(page_title="CTI Chatbot — Task + Context", layout="wide")
st.title("🧠 CTI Intelligence Assistant")

st.markdown("### Task / Instruction")
task = st.text_input(
    "Task / Instruction",
    placeholder="Describe what you want the assistant to do. Example: 'Map the CVE to CWE and justify briefly.'"
)
st.caption("Give a single instruction or role — e.g., 'Analyze the CVE', 'Classify the threat actor', or 'Calculate CVSS'.")

st.markdown("### Context / Description")
context = st.text_area(
    "Context / Description",
    height=200,
    placeholder="Paste the CVE description, MCQ, or threat report here..."
)
st.caption("Provide the detailed input text that will be analyzed by the pipeline.")

if st.button("Submit"):
    if not task.strip() or not context.strip():
        st.warning("Please fill in both the task/instruction and context/description fields.")
    else:
        with st.spinner("Classifying query..."):
            # Send both inputs separately to classifier
            query_type = classify_query(task, context)

        st.success(f"Query classified as: **{query_type}**")

        with st.spinner("Running pipeline..."):
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

        st.subheader("📄 Pipeline Output")
        st.write(response)
