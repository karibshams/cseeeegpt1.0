import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Setup LLM with your actual token
HF_TOKEN="hf_...qah" # Replace with your full token
HUGGINGFACE_REPO_ID = "google/flan-t5-base"  # Changed to more reliable model

def load_llm(huggingface_repo_id):
    """Load LLM with correct parameter configuration"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            max_new_tokens=256,  # Direct parameter, not in model_kwargs
            huggingfacehub_api_token=HF_TOKEN,
            model_kwargs={
                "max_length": 512,  # Only non-explicit parameters go here
                "do_sample": True,
                "repetition_penalty": 1.1
            }
        )
        return llm
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("Trying with minimal parameters...")
        
        # Fallback with minimal parameters
        try:
            llm = HuggingFaceEndpoint(
                repo_id=huggingface_repo_id,
                temperature=0.5,
                huggingfacehub_api_token=HF_TOKEN
            )
            return llm
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    """Main function with error handling"""
    try:
        print("Loading LLM...")
        llm = load_llm(HUGGINGFACE_REPO_ID)
        if not llm:
            print("Failed to load LLM. Please check your token and internet connection.")
            return
        
        print("Loading vector database...")
        # Load Database
        DB_FAISS_PATH = "vectorstore/db_faiss"
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        print("Creating QA chain...")
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )
        
        print("System ready!\n")
        
        # Interactive loop
        while True:
            try:
                user_query = input("Write Query Here (or 'quit' to exit): ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_query:
                    continue
                
                print("Processing...")
                response = qa_chain.invoke({'query': user_query})
                
                print("\n" + "="*50)
                print("RESULT:", response["result"])
                print("\n" + "-"*30)
                print("SOURCE DOCUMENTS:")
                for i, doc in enumerate(response["source_documents"], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
                print("="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Please try again.\n")
                
    except FileNotFoundError as e:
        print(f"Error: Vector database not found. Please check if 'vectorstore/db_faiss' exists.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error initializing system: {e}")

if __name__ == "__main__":
    main()