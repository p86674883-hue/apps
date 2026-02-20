from dotenv import load_dotenv #retrive api values from dotenv file
import streamlit as st # provide the GUI for user 
from PyPDF2 import PdfReader  # used to read and extract text from PDF
from langchain_text_splitter import CharacterTextSplitter # use to spilt the text into multiple chunks
from langchain.openai import OpenAIEmbeddings #embedding chunks using openai embedders
from langchain_community.vectorstores import FAISS # it is a vector database that we use to perform semantic searchs
from langchain_classic.chains.question_answering import load_qa_chain
from langchaim_classic.llms import OpenAI
from langchain_classic.callbacks import get_openai_callback # used for calculating cost of usage
def main():

    # loading the api key value from dotenv file 
    load_dotenv()

    #setting the page name on streamlit
    st.set_page_config(page_title="PDF Reader")

    #Dislay the header 
    st.header("Upload your pdf  to Ask questions")

    #file_uploader will take pdf file -> (type="pdf") and storing into pdf variable
    pdf = st.file_uploader("Drag and drop", type="pdf")

    #checking if pdf exist or not
    if pdf is not None:
        #reading the pdf 
        pdf_reader = PdfReader(pdf)
        #declaring the variable to store extracted text 
        text=""
    
        #Extracting the text from pdf pages one by one with loops
         for page in pdf_reader.pages:
    
            #concating the extracted text from page
            text+=page.extract_text()
        
            #showing the text on screen
            st.write(text)

		    #With in the for loop we also chunk the text into multiple clusters
		    # first initialize/create an object of CharacterTextSpilitter
		    text_splitter= CharacterTextSpillter(separator="/n",chunk_size=1000,chunk_overlap=200,length_function=len)
		    #Now splitting the text
		    chunks=text_splitter.split_text(text)
		    #displayong the chunks
		    st.write("chunks are belows:")
		    st.write(chunks)
            #Now it's time for Embedding the chunks using OpenAIEmbeddings
		    #first, Create an initialize/Object of OpenAIEmbeddings
		    embeddings= OpenAIEmbeddings()
		    #Create a Vector Database (FAISS) and storing Vector Value in it.
		    knowledge_base= FAISS.from_texts(chunks,embeddings) #chunks is the data we want to embed and embeddings is the  embeddings providers.
		
            #-----Creating a User Interface----
		    user_input = st.text_input("Ask a Questions: ")
		    #check either user ask a question or not
		    if user_input:
		        # searching the asked question on the knowledge_base/vector Database
		        docs_search= knowledge_base.similarity_search(user_input)
                # revelant Answer to questions
			    # initialize LLM (OpenAI)
			    llm= OpenAI()
			    chain= load_qa_chain(llm,chain_type="stuff")
			    #responding to question 
			    with get_openai_callback as cb: #<-- it is the funvtion that is used to calculate the cost per questions
			        responde = chain.run(input_documents=docs_search,question=user_input)
				    print(cb)
			    st.write("The Answer")
			    st.write(response)
			
	
# checking the program are being executed directly and not as modules 

if __name__ == "__main__":
    main()


# overall description of this project 
# | | |
# v v v
# import all the requirement and dependency
# By using PyPDF module we read PDF amd extract the text from each page amd store into variable
# By using langchain Class call CharacterTextSplitter we split the text into multiple clusters
# By using OpenAIEmbeddings we vectorize every chunk and store in FAISS vector Database.


# creating a use interface to interact with Agent
# check the question either asked or not
# check all the relevent answer from database
# Agent answer the question by using all list of relevent answer and specified LLM model
# we capture the response and display it to user
# response is wrapped around by get_openai_callback to calculate the cost per questions
