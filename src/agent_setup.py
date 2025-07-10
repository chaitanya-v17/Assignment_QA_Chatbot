from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


def setup_qa_chain(vector_store, openai_api_key, verbose=True):
    # Initialize LLM
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o", temperature=0.4)

    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    # Custom prompt template
    prompt_template = """
You are a helpful and accurate AI assistant. Use only the information provided in the following context to answer the user’s question.

Context:
{context}

Question:
{question}

Instructions:
- Only answer based on the given context.
- If the answer is not in the context, respond with "The answer is not available in the provided document."
- Do not make up or hallucinate any information.
- Keep the response concise, relevant, and factually accurate.
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Build Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer",
        verbose=verbose
    )

    return qa_chain


