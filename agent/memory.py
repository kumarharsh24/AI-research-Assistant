from langchain_classic.memory import ConversationBufferMemory


def build_memory() -> ConversationBufferMemory:
    """Create chat memory so the agent can use prior turns."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=True,
    )
