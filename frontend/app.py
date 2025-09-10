import streamlit as st
import requests
import time

# Configuration for the FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:", layout="wide")
    
    st.title("Chat with your PDFs 📚")
    st.markdown("""
    <style>
        .st-emotion-cache-1y4p8pa {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)


    # Sidebar for PDF uploads
    with st.sidebar:
        st.header("1. Upload Documents")
        st.markdown("""
        Upload one or more PDF files. The system will process them and build a knowledge base. 
        This might take a few minutes depending on the size and number of documents.
        """)
        
        uploaded_files = st.file_uploader(
            "Choose your PDF files",
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Process Documents", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing documents... Please wait."):
                    files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                    
                    try:
                        response = requests.post(f"{BACKEND_URL}/upload/", files=files_to_upload, timeout=600)
                        if response.status_code == 202:
                            st.success(response.json().get("message", "Processing started!"))
                            with st.spinner("Indexing documents... this can take a moment."):
                                time.sleep(10) # Give backend some time to process
                            st.toast("✅ Documents are ready for querying!")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not connect to backend: {e}")
            else:
                st.warning("Please upload at least one PDF file.")

    # Main chat interface
    st.header("2. Ask a Question")
    st.markdown("Once your documents are processed, you can ask questions here.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    payload = {"query": prompt}
                    response = requests.post(f"{BACKEND_URL}/ask/", json=payload, timeout=300)
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "Sorry, I couldn't find an answer.")
                        message_placeholder.markdown(answer)
                        
                        # Display source documents
                        with st.expander("View Sources"):
                            for doc in result.get("source_documents", []):
                                st.write(f"**Source:** `{doc.get('source', 'N/A')}`")
                                st.info(doc.get("content", ""))
                                
                        full_response = answer
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"Failed to get an answer: {error_detail}")
                        full_response = f"Error: {error_detail}"

                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
                    full_response = f"Error: Could not get a response from the backend. {e}"
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()