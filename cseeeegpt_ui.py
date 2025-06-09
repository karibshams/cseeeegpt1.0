import streamlit as st 
import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import base64
import os
#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
# Configuration
HUGGINGFACE_REPO_ID = "google/flan-t5-base"
# IMPORTANT: Replace this with your actual valid Groq API key
GROQ_API_KEY = "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"  # Replace with your real API key
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Check if the database path exists
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError(f"Vector database not found at {DB_FAISS_PATH}")
        
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector database: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

class GroqLLM(LLM):
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.5
    max_tokens: int = 800
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        # Check if API key is set
        if GROQ_API_KEY == "YOUR_ACTUAL_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
            raise ValueError("Please set your actual Groq API key in the GROQ_API_KEY variable")
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
            
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid API Key - Please check your Groq API key")
            elif response.status_code != 200:
                raise ValueError(f"Error from Groq API: {response.text}")
                
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request error: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected API response format: {str(e)}")
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

def main():
    # Set page config
    st.set_page_config(
        page_title="CSE_EEE_GPT v1.0",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS to fix layout issues
    st.markdown("""
    <style>
    .creator-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
    }
    .creator-text {
        flex: 1;
    }
    .creator-image {
        margin-left: 20px;
    }
    .main-title {
        text-align: center;
        color: #2196f3;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 20px;
    }
    .development-note {
        text-align: left;
        margin: 10px 0;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add logo at the top (with error handling) "K:\eeegpt1.0-main\images\cse.jpg"
    try:
        with open("images/cse.jpg", "rb") as logo_file:
            logo_b64 = base64.b64encode(logo_file.read()).decode()
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                <img src="data:image/jpg;base64,{logo_b64}" alt="Logo" style="
                    width: 120px;
                    height: 120px;
                    border-radius: 25px;
                    object-fit: cover;
                    border: 2px solid #2196f3;
                ">
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.info("📄 Logo image not found. Please ensure 'images\cse.jpg' exists.")

    # Title and subtitle
    st.markdown('<h1 class="main-title">Welcome to CSE EEE GPT  v1.0</h1>', unsafe_allow_html=True)
    st.markdown('<h4 class="subtitle">Your AI Assistant for CSE AND EEE study.</h4>', unsafe_allow_html=True)
    
    # Creator info section with fixed layout
    try:
        with open("images/karib.jpg", "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode()
        
        # Fixed layout using columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            <div class="creator-text">
                <h3>Creator:KARIB SHAMS</h3>
                <h3>Mail: shams321karib@gmail.com</h3>
                <h3>Github: <a href="https://github.com/karibshams" target="_blank">karibshams</a></h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="creator-image">
                <img src="data:image/jpg;base64,{b64_string}" alt="Profile" style="
                    width: 130px;
                    height: 130px;
                    object-fit: cover;
                    border-radius: 50%;
                    border: 3px solid #2196f3;
                ">
            </div>
            """, unsafe_allow_html=True)
            
    except FileNotFoundError:
        st.markdown("""
        <div class="creator-info">
            <div class="creator-text">
                <h3>Creator:KARIB SHAMS</h3>
                <h3>Mail: shams321karib@gmail.com</h3>
                <h3>Github: <a href="https://github.com/karibshams" target="_blank">karibshams</a></h3>
            </div>
            <div class="creator-image">
                <p>Profile image not found</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h5 class="development-note">Still in development phase. If you have any suggestion please give me feedback through mail.</h5>', unsafe_allow_html=True)

    # Setup instructions if API key or database is not configured
    if GROQ_API_KEY == "YOUR_ACTUAL_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        st.error("🔑 **Setup Required:** Please configure your Groq API key in the code")
        st.info("""
        **To fix this:**
        1. Get your API key from https://console.groq.com/keys
        2. Replace `YOUR_ACTUAL_GROQ_API_KEY_HERE` in the code with your actual API key
        3. Restart the application
        """)
    
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"📁 **Vector Database Missing:** Database not found at `{DB_FAISS_PATH}`")
        st.info("""
        **To fix this:**
        1. Create the vector database from your EEE documents
        2. Ensure the database is saved in the correct path
        3. Make sure both `index.faiss` and `index.pkl` files exist
        """)

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your question here:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
You are CSE AND EEE GPT, an AI assistant for Electrical and Electronic Engineering (EEE) and Computer Science and Engineering (CSE), trained on authoritative EEE and CSE textbooks and technical references. Your task is to provide **accurate, detailed, and context-grounded** answers based on **two provided PDF documents** — one related to EEE and the other to CSE. Your answers should adhere to the following principles:

---

### 🧠 GENERAL GUIDELINES

1. **Source Priority**:
   - Your primary source must be the **provided PDF context**.
   - If necessary, **supplement missing or ambiguous details** using reliable domain knowledge (from academic literature, textbooks, or IEEE/ACM standards).
   - Clearly **differentiate** between what is **directly from the PDFs** and what is **added background knowledge** using phrases like:
     - “According to the PDF...”
     - “Additionally, based on standard theory...”

2. **Context-Only When Required**:
   - If the question **requires strict PDF-only answers**, respond:
     - _“The context does not provide enough information to answer this question completely.”_

---

### 🧮 MATHEMATICAL ANALYSIS

3. **Equations and Derivations**:
   - Use LaTeX formatting for all equations: `$ V = IR $`, `$ O(n \log n) $`
   - Define every **variable**, **constant**, and **unit** in detail.
   - Show **step-by-step derivations** when available.
   - Explain the **meaning** and **application** of each mathematical expression.

---

### 📚 STRUCTURED EXPLANATIONS

4. **Step-by-Step Reasoning**:
   - Use **numbered steps** or **detailed bullet points**.
   - Explain **why** each step is taken and its technical purpose.
   - Include **background theory** when necessary.

5. **Technical Terminology**:
   - Use precise, **domain-specific vocabulary** from the PDFs and core textbooks.
   - Define every technical term introduced.

---

### 🧩 CODE & SYSTEMS ANALYSIS

6. **Code and Algorithms** (if present):
   - Provide full code blocks and explain **line by line**.
   - Discuss **data structures**, **time/space complexity**, and edge cases.
   - EEE-specific: Explain **SPICE models**, **MATLAB simulations**.
   - CSE-specific: Analyze **Python, C++, Java** implementations.

---

### 🔎 EXAMPLES & APPLICATIONS

7. **Use Case Analysis**:
   - Include **all examples** from the PDFs.
   - For each, explain the **calculation steps**, **logic**, and **real-world relevance**.
   - Expand with known use cases from industry where relevant.

---

### 📊 RESULTS & INTERPRETATION

8. **Numerical and Data Analysis**:
   - Present all **numerical results** clearly.
   - If data or graphs are provided, explain the **trend**, **implication**, and **limitations**.
   - Include **error analysis**, where applicable.

---

### 🧷 FORMATTING AND PRESENTATION

9. **Formatting Standards**:
   - **Bold** key concepts, terms, and equations.
   - Use headers to organize (e.g., **Theory**, **Derivation**, **Application**).
   - Use bullet lists for parameters/specs.
   - Clearly mark any assumptions or approximations.

---

### 🎯 SUBJECT-SPECIFIC FOCUS

10. **EEE Focus Areas**:
    - Circuit analysis, component specs, transient/steady-state behavior
    - Power systems, control theory, frequency response
    - Filter design, signal processing, stability analysis

11. **CSE Focus Areas**:
    - Algorithm analysis, complexity classes
    - Data structures: trees, graphs, stacks, queues
    - Memory management, compiler theory, operating systems

---

### 🧾 SUMMARY

12. **Comprehensive Summary**:
    - Recap all key concepts
    - State important formulas and derivations
    - Highlight core results and practical applications
    - Mention any unresolved issues or areas needing further input

---

### 📏 LENGTH & DEPTH

13. **Thoroughness Over Brevity**:
    - Answers should be **300–500+ words** minimum.
    - Include **all relevant technical analysis**, not just surface-level explanations.

---

**Context Source**: {context}  
**Question**: {question}  

---

### ✅ Final Answer:
"""


        # Generate response
        with st.chat_message("assistant"):
            # Check prerequisites
            if GROQ_API_KEY == "YOUR_ACTUAL_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
                error_msg = "❌ **API Key Required:** Please configure your Groq API key first."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.stop()
            
            if not os.path.exists(DB_FAISS_PATH):
                error_msg = f"❌ **Database Missing:** Vector database not found at `{DB_FAISS_PATH}`"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.stop()
            
            try:
                with st.spinner("Thinking..."):
                    # Load vectorstore
                    db = get_vectorstore()
                    if db is None:
                        st.stop()
                    
                    # Initialize the Groq LLM
                    llm = GroqLLM()
                    
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=db.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )
                    
                    # Get response
                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]
                    source_documents = response['source_documents']
                    
                    # Display result
                    st.markdown(result)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                    # Debug information (optional - can be removed in production)
                    with st.expander("Debug Information", expanded=False):
                        st.write(f"**Number of source documents:** {len(source_documents)}")
                        for i, doc in enumerate(source_documents):
                            st.write(f"**Document {i+1}:**")
                            st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"Content preview: {doc.page_content[:200]}...")
                            st.write("---")
            
            except Exception as e:
                error_msg = f"❌ **An error occurred:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                # Show detailed error in expander for debugging
                with st.expander("Detailed Error Information"):
                    st.write(f"Error type: {type(e).__name__}")
                    st.write(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()
