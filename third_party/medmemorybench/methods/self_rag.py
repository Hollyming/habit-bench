import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import time
import re

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks

# Load environment variables from a .env file
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


def _get_embeddings(embedding_model: str = None):
    """Create an Embeddings instance (OpenAI API or local HuggingFace model)."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "openai")
    model_name = embedding_model or os.environ.get("DEFAULT_EMBEDDING_MODEL")

    if provider in ("local", "huggingface"):
        # Use local HuggingFace model
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print(f"[SelfRAG] Using local HuggingFace embedding model: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            print("[SelfRAG] Warning: HuggingFaceEmbeddings not available, falling back to OpenAI")
            provider = "openai"

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if model_name:
        kwargs["model"] = model_name

    return OpenAIEmbeddings(**kwargs)


def _get_openai_embeddings(embedding_model: str = None):
    """Backward-compatible alias."""
    return _get_embeddings(embedding_model)


def _get_chat_openai(model_name: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1000):
    """Create a ChatOpenAI instance with custom base_url support."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


# Define all relevant classes/functions
class RetrievalResponse(BaseModel):
    response: str = Field(..., title="Determines if retrieval is necessary", description="Output only 'Yes' or 'No'.")


class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant",
                          description="Output only 'Relevant' or 'Irrelevant'.")


class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response", description="The generated response.")


class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported",
                          description="Output 'Fully supported', 'Partially supported', or 'No support'.")


class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")


# Define prompt templates
retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)

relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)

generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)

support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)



def replace_t_with_space(list_of_documents):
        """
        Replaces all tab characters ('\t') with spaces in the page content of each document

        Args:
            list_of_documents: A list of document objects, each with a 'page_content' attribute.

        Returns:
            The modified list of documents with tab characters replaced by spaces.
        """

        for doc in list_of_documents:
            doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
        return list_of_documents

def encode_documents(documents, chunk_size=4096, chunk_overlap=200, embedding_model=None):
    """
    Encodes a PDF document into a vector store using hypothetical prompt embeddings.

    Args:
        documents: The documents to be encoded.
        chunk_size: The size of each text chunk.
        chunk_overlap: The overlap between consecutive chunks.
        embedding_model: Optional embedding model name.

    Returns:
        A FAISS vector store containing the encoded book content.
    """
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = _get_openai_embeddings(embedding_model)
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore



# Define main class
class SelfRAG:
    def __init__(self, documents, temperature=0.7, top_k=3, model_name=None, embedding_model=None):
        self.vectorstore = encode_documents(documents=documents, embedding_model=embedding_model)
        self.top_k = top_k
        self.llm = _get_chat_openai(model_name or "gpt-4o-mini", temperature=temperature, max_tokens=1000)

        # Create LLMChains for each step
        self.retrieval_chain = retrieval_prompt | self.llm.with_structured_output(RetrievalResponse)
        self.relevance_chain = relevance_prompt | self.llm.with_structured_output(RelevanceResponse)
        self.generation_chain = generation_prompt | self.llm.with_structured_output(GenerationResponse)
        self.support_chain = support_prompt | self.llm.with_structured_output(SupportResponse)
        self.utility_chain = utility_prompt | self.llm.with_structured_output(UtilityResponse)
        self.start_time = time.time()
        
        
    def run(self, query):
        print(f"\nProcessing query: {query}")

        # Step 1: Determine if retrieval is necessary
        print("Step 1: Determining if retrieval is necessary...")
        
        ## Step 1.1: find the "Question" in the message
        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query =  ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query =  ''.join(match.groups())
            else:
                retrieval_query = query
        print(f"\n\nRetrieval query: {retrieval_query}\n\n")
        
        ## Step 1.2: decide if retrieval is necessary
        input_data = {"query": retrieval_query}        
        retrieval_decision = self.retrieval_chain.invoke(input_data).response.strip().lower() #retrieval
        print(f"Retrieval decision: {retrieval_decision}")

        if retrieval_decision == 'yes':
            # Step 2: Retrieve relevant documents
            print("Step 2: Retrieving relevant documents...")
            docs = self.vectorstore.similarity_search(retrieval_query, k=self.top_k)
            contexts = [doc.page_content for doc in docs]
            print(f"Retrieved {len(contexts)} documents")

            # Step 3: Evaluate relevance of retrieved documents
            print("Step 3: Evaluating relevance of retrieved documents...")
            relevant_contexts = []
            for i, context in enumerate(contexts):
                input_data = {"query": retrieval_query, "context": context}
                relevance = self.relevance_chain.invoke(input_data).response.strip().lower() #retrieval
                print(f"Document {i + 1} relevance: {relevance}")
                if relevance == 'relevant':
                    relevant_contexts.append(context)

            print(f"Number of relevant contexts: {len(relevant_contexts)}")

            # If no relevant contexts found, generate without retrieval
            memory_construction_time = time.time() - self.start_time
            if not relevant_contexts:
                print("No relevant contexts found. Generating without retrieval...")
                input_data = {"query": query, "context": "No relevant context found."}
                
                return self.generation_chain.invoke(input_data).response, "No relevant context found.", memory_construction_time, (time.time() - self.start_time - memory_construction_time)

            
            
            # Step 4: Generate response using relevant contexts
            print("Step 4: Generating responses using relevant contexts...")
            responses = []
            for i, context in enumerate(relevant_contexts):
                print(f"Generating response for context {i + 1}...")
                input_data = {"query": query, "context": context}
                response = self.generation_chain.invoke(input_data).response

                # Step 5: Assess support
                print(f"Step 5: Assessing support for response {i + 1}...")
                input_data = {"response": response, "context": context}
                support = self.support_chain.invoke(input_data).response.strip().lower()
                print(f"Support assessment: {support}")

                # Step 6: Evaluate utility
                print(f"Step 6: Evaluating utility for response {i + 1}...")
                input_data = {"query": query, "response": response}
                utility = int(self.utility_chain.invoke(input_data).response)
                print(f"Utility score: {utility}")

                responses.append((response, support, utility))

            # Select the best response based on support and utility
            print("Selecting the best response...")
            best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))
            print(f"Best response support: {best_response[1]}, utility: {best_response[2]}")
            query_time_len = time.time() - self.start_time - memory_construction_time
            return best_response[0], relevant_contexts, memory_construction_time, query_time_len
        else:
            # Generate without retrieval
            print("Generating without retrieval...")
            input_data = {"query": query, "context": "No retrieval necessary."}
            memory_construction_time = time.time() - self.start_time
            query_time_len = time.time() - self.start_time - memory_construction_time
            
            # Generate response
            return self.generation_chain.invoke(input_data).response, "No retrieval necessary.", memory_construction_time, query_time_len


# Argument parsing functions
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Self-RAG method")
    parser.add_argument('--path', type=str, default='../data/Understanding_Climate_Change.pdf',
                        help='Path to the PDF file for vector store')
    parser.add_argument('--query', type=str, default='What is the impact of climate change on the environment?',
                        help='Query to be processed')
    return parser.parse_args()


# Main entry point
if __name__ == "__main__":
    #args = parse_args()
    #rag = SelfRAG(path=args.path)
    #response = rag.run(args.query)
    #print("\nFinal response:")
    #print(response)
    
    pass
