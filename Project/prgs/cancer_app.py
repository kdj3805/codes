import streamlit as st
from cancer_retrieval import generate_answer

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🧬", layout="wide")

# ==============================================================================
# SIDEBAR: PATIENT CONTEXT INGESTION
# ==============================================================================
with st.sidebar:
    st.header("📄 Patient Report")
    st.markdown("Upload or paste a medical report here. The AI will use it to personalize your answers and explain complex terms.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])
    
    # 2. Text Area (for copy-pasting)
    pasted_report = st.text_area("Or paste your report text here:", height=200)
    
    # Consolidate the patient data
    patient_context = ""
    if uploaded_file is not None:
        patient_context = uploaded_file.getvalue().decode("utf-8")
    elif pasted_report.strip():
        patient_context = pasted_report.strip()
        
    if patient_context:
        st.success("✅ Patient report loaded into AI memory!")
        with st.expander("View Loaded Report"):
            st.write(patient_context)

# ==============================================================================
# MAIN CHAT INTERFACE
# ==============================================================================
st.title("🧬 Personalized Medical RAG Assistant")
st.markdown("Ask me anything about your report or general oncology questions. I am backed by clinical research papers.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Please upload your medical report on the left, and ask me any questions you have about it."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("E.g., What does 'BRAF V600E positive' mean in my report?"):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing report and searching medical literature..."):
            try:
                # Pass BOTH the question and the patient's report to the backend
                answer, sources = generate_answer(query=user_query, patient_report=patient_context)
                
                source_text = "\n\n**Sources Used from Literature:**\n" + "\n".join([f"- {s}" for s in sources])
                full_response = answer + source_text
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")