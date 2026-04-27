import os
import re
import streamlit as st
from retrieval_save_cancer import generate_answer

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Medical RAG Assistant", page_icon="🧬", layout="wide")

# This MUST match the folder where your ingest.py saved the images
IMG_DIR = r"D:\trial\extracted_images" 

# ==============================================================================
# THE MAGIC: MULTI-MODAL RENDERER
# ==============================================================================
def render_message_with_images(text):
    """
    Scans the LLM's response for our custom image tag [IMAGE: filename.png].
    Highly forgiving of markdown bolding, backticks, and case variations.
    """
    # 1. Strip out annoying markdown the LLM might have wrapped around the tag
    clean_text = text.replace("**[", "[").replace("]**", "]")
    clean_text = clean_text.replace("`[", "[").replace("]`", "]")
    clean_text = clean_text.replace("*[", "[").replace("]*", "]")
    
    # 2. Split using IGNORECASE (Catches [Image: ...], [image: ...], etc.)
    parts = re.split(r'\[IMAGE:\s*([^\]]+)\]', clean_text, flags=re.IGNORECASE)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Normal text
            if part.strip():
                st.markdown(part)
        else:
            # It's an image filename! Clean it up aggressively.
            filename = part.strip().replace("`", "").replace('"', '').replace("'", "")
            
            # Construct the absolute path
            img_path = os.path.join(IMG_DIR, filename)
            
            # Show the user EXACTLY what the system is trying to look for
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Reference Document: {filename}", use_container_width=True)
            else:
                st.error(f"⚠️ The AI referenced an image, but it couldn't be found.")
                st.code(f"Attempted to load from: {img_path}") # Debugging helper

# ==============================================================================
# SIDEBAR: PATIENT CONTEXT INGESTION
# ==============================================================================
with st.sidebar:
    st.header("📄 Patient Report")
    st.markdown("Upload or paste a medical report here.")
    
    uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])
    pasted_report = st.text_area("Or paste your report text here:", height=200)
    
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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me any questions. I can pull charts and figures directly from the medical literature to help explain concepts!"}
    ]

# Render the entire chat history using our custom renderer so images stay visible
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message_with_images(message["content"])

# User Input
if user_query := st.chat_input("E.g., Can you show me the ABCDE criteria for melanoma?"):
    
    # Render user query
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing report and searching medical literature..."):
            try:
                # Call the backend
                answer, sources = generate_answer(query=user_query, patient_report=patient_context)
                
                # Append sources
                source_text = "\n\n**Sources Used from Literature:**\n" + "\n".join([f"- {s}" for s in sources])
                full_response = answer + source_text
                
                # Pass the final string through our custom renderer!
                render_message_with_images(full_response)
                
                with st.expander("🔍 DEBUG: View Raw LLM Output"):
                    st.text(full_response)

                # Save to session state
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")