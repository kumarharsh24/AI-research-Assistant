import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    if not api_key or api_key == "YOUR_API_KEY" or api_key == "YOUR_GEMINI_API_KEY":
        print("Set GEMINI_API_KEY in your .env file before running chat_cli.py")
        return

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
    )

    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        reply = llm.invoke(user_input)
        print(f"Bot: {reply.content}")


if __name__ == "__main__":
    main()
