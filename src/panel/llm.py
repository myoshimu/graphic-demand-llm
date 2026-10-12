"""LLM 呼び出し（Vertex AI の Gemini と Claude）：キャッシュ・台帳・IP ポリシー・予算上限（CLAUDE.md §2.4 / §6）。

- キャッシュ・台帳・予算は _PanelClient が共通に持ち、API を呼ぶ部分（_invoke）だけをモデルごとに分ける。
- Claude（decisions_log 2026-10-07）は seed を受け付けないので、seed はキャッシュキーと記録にだけ残して送らない。

- キャッシュキー：model_key, model_id, prompt_sha256, image_sha256, temperature, seed（config/models.yaml cache.key_fields）
  に thinking と media_resolution を加えたもの。ヒットしたら API を呼ばない。
- 台帳：data/cache/llm_ledger.parquet に 1 呼び出し 1 行で追記（キャッシュヒットも cache_hit=True で記録）。
- IP 画像（is_original == False）は config/ip_api_policy.yaml の allow_external_api_for_ip が true のときだけ送る。
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
import structlog
import yaml
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

from src.ingest.bq import REPO_ROOT

log = structlog.get_logger()

MODELS_PATH = REPO_ROOT / "config" / "models.yaml"
POLICY_PATH = REPO_ROOT / "config" / "ip_api_policy.yaml"
_LEDGER_FILE_LOCK = threading.Lock()  # 複数モデルのクライアントが同じ台帳に書くため
_MEDIA_RES = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}


class PanelResponse(BaseModel):
    purchase_intent: int = Field(ge=1, le=5)
    purchase_prob: float = Field(ge=0, le=1)
    would_pay_jpy: int = Field(ge=0)
    reason: str


class BudgetExceeded(RuntimeError):
    pass


class CallResult(BaseModel):
    cache_key: str
    response: Any  # response_model のインスタンス（既定 PanelResponse）または None
    raw_text: str | None
    parse_error: str | None
    cache_hit: bool
    usd_cost: float


def _sha256(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


def load_models() -> dict[str, Any]:
    with open(MODELS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _extract_json(raw: str | None) -> str:
    """Claude は JSON を ```json で囲むことがあるので中身だけ取り出す。Gemini（JSON モード）の出力はそのまま通る。"""
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if t and t[0] not in "{[" and "{" in t and "}" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    return t


def make_client(model_key: str, **kwargs: Any) -> "_PanelClient":
    """config/models.yaml の provider でクライアントを選ぶ。"""
    spec = next((m for m in load_models()["models"] if m["key"] == model_key), None)
    if spec is None:
        raise ValueError(f"モデル {model_key} が config/models.yaml にありません")
    cls = {"vertex_ai": GeminiPanelClient, "anthropic_vertex": ClaudePanelClient}[spec["provider"]]
    return cls(model_key, **kwargs)


class _PanelClient:
    """スレッドセーフ。1 インスタンスを複数スレッドで共有してよい。サブクラスは _invoke を実装する。"""

    provider: str = ""

    def __init__(self, model_key: str, *, budget_usd: float, media_resolution: str = "high",
                 batch_pricing: bool = False, max_tokens: int | None = None) -> None:
        cfg = load_models()
        spec = next((m for m in cfg["models"] if m["key"] == model_key), None)
        if spec is None or spec["provider"] != self.provider:
            raise ValueError(f"{self.provider} のモデル {model_key} が config/models.yaml にありません")
        if spec.get("training_cutoff") is None:
            raise ValueError(f"{model_key} の training_cutoff が未記入です（§5.3）")
        self.spec = spec
        self.model_key = model_key
        self.model_id: str = spec["model_id"]
        sfx = "_batch" if batch_pricing else ""
        self.usd_in = float(spec[f"usd_per_1k_input{sfx}"]) / 1000
        self.usd_out = float(spec[f"usd_per_1k_output{sfx}"]) / 1000
        self.media_resolution = media_resolution
        self.max_tokens = int(max_tokens or cfg["defaults"]["max_tokens"])
        self.cache_dir = REPO_ROOT / cfg["cache"]["dir"]
        self.ledger_path = REPO_ROOT / cfg["cache"]["ledger"]
        with open(POLICY_PATH, encoding="utf-8") as fh:
            self.allow_ip = bool(yaml.safe_load(fh)["allow_external_api_for_ip"])
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.thinking = ""
        self._lock = threading.Lock()
        self._ledger_buf: list[dict[str, Any]] = []

    # ---- キャッシュ ----
    def _key(self, prompt_sha: str, image_sha: str | None, temperature: float, seed: int) -> str:
        fields = [self.model_key, self.model_id, prompt_sha, image_sha or "", f"{temperature:.3f}", str(seed),
                  f"think={self.thinking}", f"res={self.media_resolution}", f"max={self.max_tokens}"]
        return _sha256("|".join(fields))

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / self.model_key / key[:2] / f"{key}.json"

    # ---- 呼び出し ----
    def call(self, *, system: str, user_text: str, image_path: str | None = None, image_sha256: str | None = None,
             is_original: bool, temperature: float, seed: int,
             images: list[tuple[str, str]] | None = None,
             response_model: type[BaseModel] = PanelResponse) -> CallResult:
        """images は (パス, sha256) のリスト。複数画像（2 案比較）用で、image_path とは併用しない。"""
        if images is not None and image_path is not None:
            raise ValueError("image_path と images は併用しない")
        if image_path is not None:
            images = [(image_path, image_sha256 or "")]
        images = images or []
        if images and not is_original and not self.allow_ip:
            raise PermissionError("IP 画像の外部 API 送信は ip_api_policy.yaml で禁止されています")
        image_sha256 = "+".join(sha for _, sha in images) or None
        prompt_sha = _sha256(system + "\n\n" + user_text)
        key = self._key(prompt_sha, image_sha256, temperature, seed)
        path = self._cache_path(key)
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            self._ledger(rec, cache_hit=True, usd=0.0, latency=0.0)
            return self._result(key, rec, cache_hit=True, usd=0.0, model=response_model)

        with self._lock:
            if self.spent_usd >= self.budget_usd:
                raise BudgetExceeded(f"予算上限 ${self.budget_usd:.2f} に達しました（{self.model_key}）")

        t0 = time.monotonic()
        out = self._invoke(system=system, user_text=user_text, images=images, temperature=temperature, seed=seed)
        latency = time.monotonic() - t0
        n_in, n_out = out["input_tokens"], out["output_tokens"]
        usd = n_in * self.usd_in + n_out * self.usd_out
        rec = {
            "cache_key": key, "model_key": self.model_key, "model_id": self.model_id,
            "model_version": out["model_version"], "prompt_sha256": prompt_sha, "image_sha256": image_sha256,
            "temperature": temperature, "seed": seed, "thinking": self.thinking,
            "media_resolution": self.media_resolution, "system": system, "user_text": user_text,
            "raw_text": out["raw_text"], "finish_reason": out["finish_reason"],
            "input_tokens": n_in, "output_tokens": n_out, "thoughts_tokens": out["thoughts_tokens"],
            "usd_cost": usd, "is_original": is_original,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        with self._lock:
            self.spent_usd += usd
        self._ledger(rec, cache_hit=False, usd=usd, latency=latency)
        return self._result(key, rec, cache_hit=False, usd=usd, model=response_model)

    def _invoke(self, *, system: str, user_text: str, images: list[tuple[str, str]], temperature: float,
                seed: int) -> dict[str, Any]:
        """API を 1 回呼び、raw_text / model_version / finish_reason / input_tokens / output_tokens（thinking 込み）/
        thoughts_tokens を返す。"""
        raise NotImplementedError

    @staticmethod
    def _result(key: str, rec: dict[str, Any], *, cache_hit: bool, usd: float,
                model: type[BaseModel] = PanelResponse) -> CallResult:
        raw = rec.get("raw_text")
        parsed, err = None, None
        try:
            parsed = model.model_validate_json(_extract_json(raw))
        except ValidationError as e:
            err = str(e).splitlines()[0][:200]
        return CallResult(cache_key=key, response=parsed, raw_text=raw, parse_error=err, cache_hit=cache_hit, usd_cost=usd)

    # ---- 台帳 ----
    def _ledger(self, rec: dict[str, Any], *, cache_hit: bool, usd: float, latency: float) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "model_key": self.model_key,
            "model_id": self.model_id, "model_version": rec.get("model_version"), "provider": self.provider,
            "is_local": False, "prompt_sha256": rec.get("prompt_sha256"), "image_sha256": rec.get("image_sha256"),
            "temperature": rec.get("temperature"), "seed": rec.get("seed"),
            "thinking": rec.get("thinking"), "media_resolution": rec.get("media_resolution"),
            "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
            "usd_cost": usd, "latency_seconds": latency, "cache_hit": cache_hit,
            "ip_policy_decision": "original" if rec.get("is_original") else ("ip_allowed" if rec.get("image_sha256") else "ip_text_only"),
        }
        with self._lock:
            self._ledger_buf.append(row)
            if len(self._ledger_buf) >= 500:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._ledger_buf:
            return
        new = pd.DataFrame(self._ledger_buf)
        with _LEDGER_FILE_LOCK:
            if self.ledger_path.exists():
                new = pd.concat([pd.read_parquet(self.ledger_path), new], ignore_index=True)
            tmp = self.ledger_path.with_suffix(".tmp")
            new.to_parquet(tmp, index=False)
            tmp.replace(self.ledger_path)
        self._ledger_buf.clear()


class GeminiPanelClient(_PanelClient):
    provider = "vertex_ai"

    def __init__(self, model_key: str, **kwargs: Any) -> None:
        super().__init__(model_key, **kwargs)
        # thinking の指定方法はモデルで違う（config/models.yaml の thinking）
        th = self.spec.get("thinking") or {}
        if "budget" in th:
            self.thinking = f"budget={int(th['budget'])}"
            self._thinking_config = types.ThinkingConfig(thinking_budget=int(th["budget"]))
        elif "level" in th:
            self.thinking = f"level={th['level']}"
            self._thinking_config = types.ThinkingConfig(thinking_level=th["level"])
        else:
            raise ValueError(f"{model_key} の thinking が config/models.yaml にありません")
        self._client = genai.Client(vertexai=True, project=self.spec.get("project", "your-gcp-project"),
                                    location=self.spec.get("location", "global"))

    def _invoke(self, *, system: str, user_text: str, images: list[tuple[str, str]], temperature: float,
                seed: int) -> dict[str, Any]:
        contents: list[Any] = []
        for i, (img, _) in enumerate(images):
            if len(images) > 1:
                contents.append(f"【画像 {chr(65 + i)}】")
            contents.append(types.Part.from_bytes(data=(REPO_ROOT / img).read_bytes(), mime_type="image/jpeg"))
        contents.append(user_text)
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, seed=seed, max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            thinking_config=self._thinking_config,
            media_resolution=_MEDIA_RES[self.media_resolution] if images else None,
        )
        resp = self._generate(contents, config)
        u = resp.usage_metadata
        n_think = int(u.thoughts_token_count or 0)
        return {
            "raw_text": resp.text if resp.candidates else None, "model_version": resp.model_version,
            "finish_reason": resp.candidates[0].finish_reason.name if resp.candidates and resp.candidates[0].finish_reason else None,
            "input_tokens": int(u.prompt_token_count or 0),
            "output_tokens": int(u.candidates_token_count or 0) + n_think, "thoughts_tokens": n_think,
        }

    def _generate(self, contents: list[Any], config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        for attempt in range(8):
            try:
                return self._client.models.generate_content(model=self.model_id, contents=contents, config=config)
            except (errors.ServerError, errors.ClientError) as e:
                code = getattr(e, "code", None)
                if isinstance(e, errors.ClientError) and code not in (429, 408):
                    raise
                wait = min(60.0, 2**attempt) * (0.5 + random.random())
                log.warning("gemini_retry", model=self.model_id, code=code, attempt=attempt, wait=round(wait, 1))
                time.sleep(wait)
        raise RuntimeError(f"{self.model_id}: リトライ上限に達しました")


class ClaudePanelClient(_PanelClient):
    """Vertex AI の Claude。thinking は送らない（Sonnet 4.6 は省略すると thinking なし）。画像は原寸で送る（API 側で縮小）。"""

    provider = "anthropic_vertex"

    def __init__(self, model_key: str, **kwargs: Any) -> None:
        super().__init__(model_key, **kwargs)
        self.thinking = "off"
        self.media_resolution = "native"
        self._client = anthropic.AnthropicVertex(project_id=self.spec.get("project", "your-gcp-project"),
                                                 region=self.spec.get("location", "global"), max_retries=8)

    def _invoke(self, *, system: str, user_text: str, images: list[tuple[str, str]], temperature: float,
                seed: int) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for i, (img, _) in enumerate(images):
            if len(images) > 1:
                content.append({"type": "text", "text": f"【画像 {chr(65 + i)}】"})
            data = base64.standard_b64encode((REPO_ROOT / img).read_bytes()).decode("ascii")
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
        content.append({"type": "text", "text": user_text})
        # SDK 1.x は temperature 引数を廃止したが、Sonnet 4.6 の API は受け付けるので extra_body で送る
        resp = self._client.messages.create(model=self.model_id, max_tokens=self.max_tokens, system=system,
                                            messages=[{"role": "user", "content": content}],
                                            extra_body={"temperature": temperature})
        return {
            "raw_text": "".join(b.text for b in resp.content if b.type == "text"), "model_version": resp.model,
            "finish_reason": resp.stop_reason, "input_tokens": int(resp.usage.input_tokens),
            "output_tokens": int(resp.usage.output_tokens), "thoughts_tokens": 0,
        }
