from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from src.core import vector_store

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=1)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# def query_or_respond(state: MessagesState):
#     """Generate tool call for retrieval or respond."""
#     llm_with_tools = llm.bind_tools([retrieve])
#     response = llm_with_tools.invoke(state["messages"])
#     # MessagesState appends messages to state instead of overwriting
#     return {"messages": [response]}


# Step 2: Execute the retrieval.


# # Step 3: Generate a response using the retrieved content.
# def generate(state: MessagesState):
#     """Generate answer."""
#     # Get generated ToolMessages
#     recent_tool_messages = []
#     for message in reversed(state["messages"]):
#         if message.type == "tool":
#             recent_tool_messages.append(message)
#         else:
#             break
#     tool_messages = recent_tool_messages[::-1]

#     # Format into prompt
#     docs_content = "\n\n".join(doc.content for doc in tool_messages)
#     system_message_content = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         f"{docs_content}"
#     )
#     conversation_messages = [
#         message
#         for message in state["messages"]
#         if message.type in ("human", "system")
#         or (message.type == "ai" and not message.tool_calls)
#     ]
#     prompt = [SystemMessage(system_message_content)] + conversation_messages

#     # Run
#     response = llm.invoke(prompt)
#     return {"messages": [response]}

# graph_builder.add_node(query_or_respond)
# graph_builder.add_node(tools)
# graph_builder.add_node(generate)

# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tools"},
# )
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)

# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "hellototo"}}


class Chat():
    def __init__(self, model):
        self.llm = model

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}
    
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}