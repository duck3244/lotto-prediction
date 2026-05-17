"""FastAPI 애플리케이션 설정.

단일 사용자 MVP 기준 — 인증/멀티테넌시 없음. 환경변수로 호스트/포트/CORS 출처를
오버라이드할 수 있도록 한다. **기본 호스트는 127.0.0.1** 로 고정해 LAN 노출을 막는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_csv(env_value: str | None, default: list[str]) -> list[str]:
    if not env_value:
        return default
    return [item.strip() for item in env_value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # 바인딩 — 단일 사용자 로컬 도구이므로 외부 노출을 막는 게 안전 디폴트.
    host: str = field(default_factory=lambda: os.environ.get("LOTTO_API_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("LOTTO_API_PORT", "8000")))

    # 개발 시 Vite dev server (5173) 허용. 빌드된 정적 파일을 같은 호스트로
    # 서빙하면 CORS 자체가 불필요해진다.
    cors_origins: list[str] = field(
        default_factory=lambda: _parse_csv(
            os.environ.get("LOTTO_API_CORS_ORIGINS"),
            default=["http://localhost:5173", "http://127.0.0.1:5173"],
        )
    )

    # 기본 시퀀스/시드 (요청에서 미지정 시 사용)
    default_sequence_length: int = 10
    default_seed: int = 42

    # 최근 회차 응답 기본 개수
    default_recent_limit: int = 20
    max_recent_limit: int = 200


settings = Settings()
