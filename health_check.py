"""NVIDIA NIM 엔드포인트 health 체크 (langchain ChatOpenAI 기반)."""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")


def main() -> int:
    if not API_KEY:
        print("❌ NVIDIA_API_KEY 가 .env 에 설정되지 않았습니다.")
        return 1

    llm = ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=1,
        top_p=1,
        max_tokens=4096,
    )

    print(f"→ base_url : {BASE_URL}")
    print(f"→ model    : {MODEL}")
    print("→ ping     : 'Reply with OK' ...")

    try:
        resp = llm.invoke("Reply with the single word: OK")
    except Exception as exc:  # 연결/인증/모델 오류 모두 잡아서 표시
        print(f"❌ health check 실패: {type(exc).__name__}: {exc}")
        return 1

    # gpt-oss 계열은 reasoning_content 를 따로 내려주기도 함
    reasoning = resp.additional_kwargs.get("reasoning_content")
    if reasoning:
        print(f"[reasoning] {reasoning}")

    print(f"✅ health OK — 응답: {resp.content!r}")

    usage = getattr(resp, "usage_metadata", None)
    if usage:
        print(f"   tokens: {usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
