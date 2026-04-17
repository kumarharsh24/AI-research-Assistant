from core.logger import setup_logging
from ui.chat_page import render_app


def main() -> None:
    setup_logging()
    render_app()


if __name__ == "__main__":
    main()
