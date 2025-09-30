# qa_chain.py
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class QAChainBuilder:
    """Builds and manages the QA chain."""
    
    def __init__(self, model_name: str = "llama3-chatqa:8b", temperature: float = 0.2):
        self.llm = Ollama(model=model_name, temperature=temperature)
    
    def build_chain(self, retriever) -> RetrievalQA:
        prompt_template = """Use the following CRISPR lab protocol information to answer the question. If you don't know the answer, say so - don't make up information.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
