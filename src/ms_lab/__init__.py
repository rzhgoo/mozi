import os
from pathlib import Path

import warp as wp

ms_lab_SRC_PATH: Path = Path(__file__).parent


def configure_warp() -> None:
  """Configure Warp globally for ms_lab."""
  wp.config.enable_backward = False

  # Keep warp verbose by default to show kernel compilation progress.
  # Override with ms_lab_WARP_QUIET=1 environment variable if needed.
  quiet = os.environ.get("ms_lab_WARP_QUIET", "").lower() in ("1", "true", "yes")
  wp.config.quiet = quiet


configure_warp()
