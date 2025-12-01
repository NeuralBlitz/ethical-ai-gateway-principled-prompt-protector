# app_hf_space.py
import streamlit as st
from src.interface import PrincipledPromptProtector
import torch # Required for model loading context

# --- Configuration ---
MODEL_PATH = "./models/" # In HF Spaces, models are often symlinked to the root
TOKENIZER_PATH = "./models/"

st.set_page_config(
    page_title="Principled Prompt Protector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🛡️ Principled Prompt Protector")
st.markdown("---")

st.markdown("""
Welcome to the **Principled Prompt Protector (PPP)**! This tool helps you craft ethically sound prompts for Large Language Models (LLMs).
It analyzes your input for potential risks related to harm, bias, privacy, deception, and accountability, providing guidance to ensure responsible AI interaction.
""")

st.markdown("---")

# --- Initialize the Protector (Singleton pattern ensures it loads only once) ---
@st.cache_resource
def load_protector():
    try:
        protector_instance = PrincipledPromptProtector(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH)
        return protector_instance
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure model files are in '{MODEL_PATH}'")
        return None

protector = load_protector()

if protector:
    st.subheader("Enter Your LLM Prompt Below:")
    user_prompt = st.text_area("Prompt Input", "Write a short story about a future where AI helps humanity flourish ethically.", height=150)

    risk_threshold_display = st.slider(
        "Flagging Threshold (higher means stricter flagging)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust this to make the protector more or less sensitive to potential risks."
    )

    if st.button("Assess Prompt"):
        if user_prompt.strip():
            with st.spinner("Assessing prompt for ethical risks..."):
                assessment = protector.assess_prompt(user_prompt, risk_threshold=risk_threshold_display)
            
            st.markdown("---")
            st.subheader("Assessment Results:")
            
            if assessment['is_flagged']:
                st.error("🚨 Prompt Flagged for Potential Ethical Concerns 🚨")
            else:
                st.success("✅ Prompt assessed as Ethically Compliant ✅")
            
            st.write(f"**Overall Risk Score:** {assessment['overall_risk_score']:.2f}")
            st.write(f"**Suggested Guidance:** {assessment['suggested_guidance']}")
            
            st.markdown("---")
            st.subheader("Detailed Risk Breakdown:")
            
            risk_cols = st.columns(len(assessment['risk_details']))
            col_idx = 0
            for category, score in assessment['risk_details'].items():
                with risk_cols[col_idx]:
                    st.metric(label=category.replace('_score', '').replace('_', ' ').title(), value=f"{score:.2f}")
                col_idx += 1
            
            st.markdown("---")
            st.info("""
            **Understanding the Scores:**
            *   **0.00 - 0.25:** Low to negligible risk.
            *   **0.25 - 0.50:** Moderate risk, warrants review.
            *   **0.50 - 0.75:** High risk, likely requires rephrasing.
            *   **0.75 - 1.00:** Very high risk, strong recommendation to avoid.
            """)
        else:
            st.warning("Please enter a prompt to assess.")

else:
    st.error("The Principled Prompt Protector could not be loaded.")

st.markdown("---")
st.caption("Powered by NeuralBlitz's Ethical AI Gateway initiative.")
