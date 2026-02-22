from dotenv import load_dotenv 
from PyPDF2 import PdfReader  
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.llms import OpenAI
from langchain_classic.callbacks import get_openai_callback 
def main():

    load_dotenv()

    
    pdf = "./Marketing"

    def load_pdf(pdf):
	pass
		
	def extract_text(documents):
	    text=""
		for page in pdf.pages:
			text+=page.extract_texts(page)
		return text

    extracted_text= extract_text(pdf)
	def chunking(text:str):
		chunks= CharacterTextSplitters(separator=["\n","\n\n"] offset=50)
		docs=chunks.split_documents()
		return docs

    chunked_documents =chunking(extracted_text)

	def embed(chunked_documents):
		embedding_model = HuggingFaceInstructorEmbedding("instructor-lx")
        for doc in chunked_documents:
			embed=embedding_model.embed(doc)
			vector_indexing(embed)
			
    def vector_store:
        pass
    def vector_indexing
        pass
    def search_vector:
        pass

    if pdf is not None:
      
        pdf_reader = PdfReader(pdf)
       
        text=""
    
        
        for page in pdf_reader.pages:
    
            text+=page.extract_text()
            st.write(text)
            print(text)

		    
            text_splitter= CharacterTextSplitter(separator="/n",chunk_size=1000,chunk_overlap=200,length_function=len)
		    
            chunks=text_splitter.split_text(text)
		    
      
            
            embeddings= OpenAIEmbeddings()
		    
            knowledge_base= FAISS.from_texts(chunks,embeddings) #chunks is the data we want to embed and embeddings is the  embeddings providers.
		
        
            user_input = input("Ask a Questions: ")

            if user_input:
		
                docs_search= knowledge_base.similarity_search(user_input)
                llm= OpenAI()
                chain= load_qa_chain(llm,chain_type="
                with get_openai_callback as cb: 
                    responde = chain.run(input_documents=docs_search,question=user_input)
                    print(cb)
                    st.write("The Answer")
                    st.write(response)
			 

if __name__ == "__main__":
    main()
