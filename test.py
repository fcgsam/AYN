from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# Initialize Mistral AI (Replace with your API key)
llm = ChatMistralAI(model="mistral-large-latest", api_key="YnljT3H2jaDR0AkOLg86GhViw079Ijts")

# Load text from a document
# text = """can we use Mistral for natural language processing tasks? Mistral is a powerful language model that can be used for various NLP tasks such as text generation, summarization, and question answering. It is designed to understand and generate human-like text based on the input it receives."""
text = """SQl function is a powerful tool in SQL that allows you to perform operations on data within your database. It can be used to manipulate, analyze, and retrieve data efficiently. SQL functions can be categorized into several types, including aggregate functions, scalar functions, and window functions. Aggregate functions perform calculations on a set of values and return a single value, such as SUM, AVG, COUNT, MIN, and MAX. Scalar functions operate on individual values and return a single value, such as UPPER, LOWER, LENGTH, and ROUND. Window functions perform calculations across a set of rows related to the current row, allowing for advanced analytics without collapsing the result set.
python is a versatile programming language that can be used for various applications, including web development, data analysis, artificial intelligence, and more. It is known for its simplicity and readability, making it a popular choice among developers. Python has a rich ecosystem of libraries and frameworks that enhance its capabilities, such as Django for web development, Pandas for data manipulation, and TensorFlow for machine learning. Its extensive community support and continuous development ensure that Python remains a relevant and powerful tool in the programming world.
django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It provides a robust set of features for building web applications, including an ORM (Object-Relational Mapping) system, an admin interface, and built-in security features. Django follows the "batteries-included" philosophy, meaning it comes with many built-in tools and libraries to help developers create complex web applications quickly and efficiently. Its modular architecture allows for easy integration with other technologies and services, making it a popular choice for web development projects.
"""
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

# Create embeddings and store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

prompt = PromptTemplate.from_template(
    "Use only the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
)


stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)
retrieved_docs = vectorstore.similarity_search("how many topics", k=4)
# Ask a question
question = "Only use my document to answer: how much topics is in my document? and if there is no information about it, then answer it by yourself and also tell me that there is no information about it in the document.but this is something that I know, so you can answer it by yourself. but if there is information about it in the document, then answer it by using the document.and yess if my question is 'count the topics' or 'how many topics' like this type of questions then ans it by yourself without telling that there is no information about it in the document."

answer = stuff_chain.invoke({"context": retrieved_docs, "question": question})
print("Answer:", answer)



# YnljT3H2jaDR0AkOLg86GhViw079Ijts