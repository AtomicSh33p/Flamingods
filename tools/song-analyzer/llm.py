"""Optional Claude API route for section labelling (needs ANTHROPIC_API_KEY).

label.py does the same through the Claude Code CLI and a Claude subscription instead.
"""

import json

import anthropic

from sections import SCHEMA, SYSTEM

MODEL = "claude-opus-5-5"


def label_sections(payload: dict) -> dict | None:
    """Ask Claude for sections. Returns None (after printing why) if the call cannot be made."""
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
    except anthropic.AuthenticationError:
        print("  Claude pass skipped: the API key was rejected.")
        return None
    except anthropic.RateLimitError:
        print("  Claude pass skipped: rate limited. Rerun later; local results are cached.")
        return None
    except anthropic.APIStatusError as e:
        print(f"  Claude pass skipped: API error {e.status_code}: {e.message}")
        return None
    except anthropic.APIConnectionError:
        print("  Claude pass skipped: no connection to the API.")
        return None
    except anthropic.AnthropicError as e:
        print(f"  Claude pass skipped: {e}")
        return None

    if response.stop_reason != "end_turn":
        print(f"  Claude pass skipped: response ended with stop_reason={response.stop_reason}.")
        return None
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
