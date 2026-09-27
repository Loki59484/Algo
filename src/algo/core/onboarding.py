import json
import os
from pathlib import Path
from typing import Any, Dict
from algo.core.config import ENV_FILE, USER_SETTINGS_FILE, USER_CONFIG_DIR


def is_setup_complete(config_dir: Path | None = None) -> bool:
    """Checks whether both the .env secret file and user_settings.json exist."""
    c_dir = config_dir or CONFIG_DIR
    env_path = c_dir / ".env"
    settings_path = c_dir / "user_settings.json"
    return env_path.is_file() and settings_path.is_file()


def save_user_profile(data: Dict[str, Any], config_dir: Path | None = None) -> bool:
    """
    Persists credentials to .env and non-sensitive metadata to user_settings.json.
    """
    target_dir = config_dir or CONFIG_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        env_lines = [
            f"UPSTOX_API_KEY={data.get('api_key', '').strip()}",
            f"UPSTOX_API_SECRET={data.get('api_secret', '').strip()}",
            f"UPSTOX_REDIRECT_URI={data.get('redirect_uri', '').strip()}",
            f"UPSTOX_TOTP_SECRET={data.get('totp_secret', '').strip()}",
            f"UPSTOX_PINCODE={data.get('pincode', '').strip()}",
        ]

        env_file = target_dir / ".env"
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("\n".join(env_lines) + "\n")

        settings = {
            "profile": {
                "name": data.get("name", "").strip(),
                "mobile": data.get("mobile", "").strip(),
            }
        }

        settings_file = target_dir / "user_settings.json"
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)

        return True

    except Exception:
        return False


def load_user_profile(config_dir: Path | None = None) -> Dict[str, Any]:
    """Loads non-sensitive configuration parameters."""
    target_file = (config_dir or CONFIG_DIR) / "user_settings.json"
    if not target_file.is_file():
        return {}
    with open(target_file, "r", encoding="utf-8") as f:
        return json.load(f)
