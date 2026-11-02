"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time
import os

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())

    temperature = float(os.getenv("AIE_TEMPERATURE", "1"))
    print(f"[requested temperature] {temperature}")

    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config={"temperature": temperature},
    )
    elapsed = time.perf_counter() - start

    print(f"[time] {elapsed:.3f} seconds")

    if response.usage_metadata is not None:
        usage = response.usage_metadata
        print(
            f"[tokens] input={usage.prompt_token_count}, "
            f"output={usage.candidates_token_count}"
        )

    return response.text

def chat_loop() -> None:
    client = genai.Client(api_key=get_api_key())
    temperature = float(os.getenv("AIE_TEMPERATURE", "1"))

    chat = client.chats.create(
        model=MODEL,
        config={"temperature": temperature},
    )

    print(f"[model] {MODEL}")
    print(f"[requested temperature] {temperature}")
    print("会話モード：終了するには /exit を入力")

    while True:
        try:
            question = input("\nあなた > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if question == "/exit":
            break
        if not question:
            continue

        try:
            start = time.perf_counter()
            response = chat.send_message(question)
            elapsed = time.perf_counter() - start
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            continue

        print(f"[time] {elapsed:.3f} seconds")

        if response.usage_metadata is not None:
            usage = response.usage_metadata
            print(
                f"[tokens] input={usage.prompt_token_count}, "
                f"output={usage.candidates_token_count}"
            )

        print(f"AI > {response.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話モード")
    args = parser.parse_args()

    if args.chat:
        chat_loop()
        return

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
