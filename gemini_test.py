import os

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    if not api_key or api_key == "YOUR_API_KEY" or api_key == "YOUR_GEMINI_API_KEY":
        print("Set GEMINI_API_KEY in your .env file before running gemini_test.py")
        return

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="What is an AI agent?",
        )
    except Exception as exc:
        err = str(exc)
        if "NOT_FOUND" in err and "models/" in err:
            print(
                "Configured Gemini model is unavailable. "
                "Set GEMINI_MODEL to a supported model (for example: gemini-2.0-flash)."
            )
            return
        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            print(
                "Gemini quota exhausted. Check quota/billing, wait for reset, "
                "or use another API key."
            )
            return
        raise

    print(response.text)


if __name__ == "__main__":
    main()
