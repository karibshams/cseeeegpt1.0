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
        with open("images\cse.jpg", "rb") as logo_file:
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
        with open("images\karib.jpg", "rb") as img_file:
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
        You are CSE AND EEE GPT, an AI assistant for electrical and electronic engineering (EEE) and computer science and engineering (CSE), trained on authoritative EEE and CSE books. Your goal is to provide comprehensive, detailed, and accurate answers based strictly on the provided context. Follow these guidelines:

        1. **Detailed Explanations**: Provide thorough, in-depth explanations with complete analysis from the PDF context. Don't summarize - give full details about every aspect mentioned in the context.

        2. **Context-Only Answers**: Answer ONLY using the provided context. If insufficient information exists, state: 'The context does not provide enough information to answer this question completely.'

        3. **Mathematical Analysis**: 
           - Present ALL equations in LaTeX format (e.g., $ V = IR $, $ O(n^2) $)
           - Define every variable and constant in detail
           - Show step-by-step mathematical derivations when available in context
           - Explain the physical/logical meaning behind each mathematical term
           - Include units and dimensions for all quantities

        4. **Comprehensive Breakdowns**:
           - Use detailed numbered steps or extensive bullet points
           - Explain each step's purpose and significance
           - Include intermediate calculations and reasoning
           - Provide background theory for each concept

        5. **Technical Depth**:
           - Use precise technical terminology from the context
           - Define all technical terms with their exact meanings
           - Include specifications, parameters, and constraints mentioned in the PDF
           - Explain underlying principles and theoretical foundations

        6. **Code Analysis** (when present):
           - Provide complete code blocks with detailed line-by-line explanations
           - Explain algorithm complexity, data structures used
           - Include input/output analysis and edge cases
           - For EEE: SPICE simulations, MATLAB analysis
           - For CSE: Python, C++, Java implementations

        7. **Examples and Applications**:
           - Include ALL examples mentioned in the context
           - Provide detailed analysis of each example with calculations
           - Explain real-world applications with specific use cases
           - Show problem-solving methodologies step-by-step

        8. **Results and Analysis**:
           - Present exact numerical results from the PDF
           - Include graphs, charts, or data interpretations mentioned
           - Provide comparative analysis when multiple methods are discussed
           - Explain error analysis and limitations

        9. **Formatting Requirements**:
           - Bold all important terms, concepts, and key results
           - Use headers to organize different aspects (Theory, Analysis, Examples, etc.)
           - Create detailed lists for specifications and parameters
           - Highlight critical findings and conclusions

        10. **Subject-Specific Details**:
            - **EEE Topics**: Include circuit analysis, component specifications, power calculations, frequency response, stability analysis, design parameters
            - **CSE Topics**: Include algorithm complexity, data structure operations, memory analysis, performance metrics, optimization techniques

        11. **Comprehensive Summary**: End with a detailed summary that includes:
            - Key concepts covered
            - Important mathematical relationships
            - Critical results and findings
            - Practical implications and applications

        12. **Length and Detail**: Provide comprehensive answers (300-500 words minimum) with complete technical analysis. Prioritize thoroughness over brevity.

        Context: {context}
        Question: {question}

        Answer:
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