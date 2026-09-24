"""Which models this bot may call, and the checks that keep it that way.

Why this file exists: forecasting-tools picks default models from whichever API
keys are in the environment, and it prefers OpenAI and Anthropic keys when they
are present. This bot must only spend donated or tournament credits, so every
model is pinned here, anything outside the allow-list is refused at startup, and
the OpenAI/Anthropic keys are removed from the process before the library reads
them.

Every model can be overridden by an env var (BOT_DEFAULT_MODEL, BOT_SUMMARIZER_MODEL,
BOT_PARSER_MODEL, BOT_RESEARCHER); the allow-list still applies to overrides.
"""

import logging
import os
from typing import Mapping, MutableMapping

logger = logging.getLogger(__name__)

# openrouter/ = our OpenRouter key (or credits Metaculus adds to it)
# metaculus/  = the Metaculus LLM proxy, billed to METACULUS_TOKEN
ALLOWED_MODEL_PREFIXES = ("openrouter/", "metaculus/")
# The researcher may also be one of the AskNews presets main.py knows how to call
# (billed to the AskNews keys), or the "do no research" sentinel.
ASKNEWS_PRESETS = (
    "asknews/news-summaries",
    "asknews/deep-research/low-depth",
    "asknews/deep-research/medium-depth",
    "asknews/deep-research/high-depth",
)
NO_RESEARCH_VALUES = ("no_research",)
# Keys that would let the library (or litellm) route to a provider we do not pay through.
BLOCKED_ENV_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

DEFAULT_MODEL = "openrouter/openai/gpt-5.6-terra"
CHEAP_MODEL = "openrouter/openai/gpt-5.6-luna"
ASKNEWS_RESEARCHER = "asknews/news-summaries"
FALLBACK_RESEARCHER = "openrouter/perplexity/sonar"


class DisallowedModelError(RuntimeError):
    pass


def scrub_blocked_keys(env: MutableMapping[str, str] | None = None) -> list[str]:
    """Remove BLOCKED_ENV_KEYS from the environment. Returns the names removed."""
    env = os.environ if env is None else env
    removed = [k for k in BLOCKED_ENV_KEYS if k in env]
    for k in removed:
        del env[k]
    if removed:
        logger.warning(f"Removed {', '.join(removed)} from this process; the bot may not use them.")
    return removed


def has_asknews(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(
        (env.get("ASKNEWS_CLIENT_ID") and env.get("ASKNEWS_SECRET"))
        or env.get("ASKNEWS_API_KEY")
    )


def model_name(llm) -> str | None:
    """The model string behind a GeneralLlm or a plain string."""
    if llm is None or isinstance(llm, str):
        return llm
    return getattr(llm, "model", None)


def is_allowed(purpose: str, name) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if purpose == "researcher" and name in ASKNEWS_PRESETS + NO_RESEARCH_VALUES:
        return True
    return name.startswith(ALLOWED_MODEL_PREFIXES)


def assert_allowed(llms: Mapping[str, object]) -> None:
    """Raise DisallowedModelError if any configured model is outside the allow-list.

    Fails closed: a missing or empty model name is refused, not skipped.
    """
    bad = [
        f"{purpose}={model_name(llm)!r}"
        for purpose, llm in llms.items()
        if not is_allowed(purpose, model_name(llm))
    ]
    if bad:
        raise DisallowedModelError(
            "Refusing to start: these models are outside the allow-list "
            f"{ALLOWED_MODEL_PREFIXES} (researcher may also be an AskNews preset "
            f"or no_research): {', '.join(bad)}"
        )


def pinned_llms(env: Mapping[str, str] | None = None) -> dict[str, object]:
    """The bot's llms dict: GeneralLlm per purpose, AskNews/no-research as strings."""
    from forecasting_tools import GeneralLlm

    temperature = {"default": 0.3, "summarizer": 0.3, "parser": 0.3, "researcher": 0.1}
    out: dict[str, object] = {}
    for purpose, name in pinned_model_names(env).items():
        if name in ASKNEWS_PRESETS + NO_RESEARCH_VALUES:
            out[purpose] = name
        else:
            out[purpose] = GeneralLlm(
                model=name, temperature=temperature[purpose], timeout=120, allowed_tries=2
            )
    return out


def pinned_model_names(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Model name per purpose, from env overrides or the pinned defaults."""
    env = os.environ if env is None else env
    researcher = env.get("BOT_RESEARCHER") or (
        ASKNEWS_RESEARCHER if has_asknews(env) else FALLBACK_RESEARCHER
    )
    if researcher.startswith("asknews/") and not has_asknews(env):
        logger.warning(
            f"BOT_RESEARCHER={researcher} but no AskNews keys are set; using {FALLBACK_RESEARCHER}."
        )
        researcher = FALLBACK_RESEARCHER
    return {
        "default": env.get("BOT_DEFAULT_MODEL") or DEFAULT_MODEL,
        "summarizer": env.get("BOT_SUMMARIZER_MODEL") or CHEAP_MODEL,
        "researcher": researcher,
        "parser": env.get("BOT_PARSER_MODEL") or CHEAP_MODEL,
    }
