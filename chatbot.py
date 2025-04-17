
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# import gradio as gr

# from dotenv import load_dotenv
# load_dotenv()


# DATA_PATH = r"data"
# CHROMA_PATH = r"chroma_db"

# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings_model,
#     persist_directory=CHROMA_PATH, 
# )

# num_results = 5
# retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# def stream_response(message, history):

#     docs = retriever.invoke(message)

#     knowledge = ""

#     for doc in docs:
#         knowledge += doc.page_content+"\n\n"


#     if message is not None:

#         partial_message = ""

#         rag_prompt = f"""
#         You are an assistent which answers questions based on knowledge which is provided to you.
#         While answering, you don't use your internal knowledge, 
#         but solely the information in the "The knowledge" section.
#         You don't mention anything to the user about the povided knowledge.

#         The question: {message}

#         Conversation history: {history}

#         The knowledge: {knowledge}

#         """

#         for response in llm.stream(rag_prompt):
#             partial_message += response.content
#             yield partial_message

# chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
#     container=False,
#     autoscroll=True,
#     scale=7),
# )

# chatbot.launch()