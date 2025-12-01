import streamlit as st
from src.interface import PrincipledPromptProtector

MODEL_PATH = "./models/"
TOKENIZER_PATH = "./models/"

st.title("🛡️ Principled Prompt Protector")
st.markdown("---")

@st.cache_resource
def load_protector():
    protector_instance = PrincipledPromptProtector(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH)
    return protector_instance

protector = load_protector()

if protector:
    st.subheader("Enter Your LLM Prompt Below:")
    user_prompt = st.text_area("Prompt Input", height=150)
    risk_threshold_display = st.slider(
        "Flagging Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    if st.button("Assess Prompt"):
        if user_prompt.strip():
            with st.spinner("Assessing prompt for ethical risks..."):
                assessment = protector.assess_prompt(user_prompt, risk_threshold=risk_threshold_display)
            st.write(assessment)
        else:
            st.warning("Please enter a prompt to assess.")
else:
    st.error("The Principled Prompt Protector could not be loaded.")
