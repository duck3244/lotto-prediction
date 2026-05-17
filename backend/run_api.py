#!/usr/bin/env python3
"""FastAPI 진입 스크립트.

- ``set_deterministic_env()`` 를 **TF import 이전에** 호출한다 (env_setup 모듈 책임).
- 단일 사용자 로컬 도구이므로 호스트 기본값은 ``127.0.0.1`` (LAN 노출 차단).
- ``LOTTO_API_HOST`` / ``LOTTO_API_PORT`` / ``LOTTO_API_CORS_ORIGINS`` 환경변수로
  오버라이드 가능.
"""

# IMPORTANT: TF 결정성 환경변수는 어떤 TF transitive 임포트보다 먼저 세팅되어야 한다.
from env_setup import set_deterministic_env

set_deterministic_env()

import logging

import uvicorn

from app.config import settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
