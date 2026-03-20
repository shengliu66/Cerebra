"""
Centralised resolver for EHR header files.

Resolution order (first match wins):
1. Explicit ``path`` argument passed by the caller.
2. Environment variable ``CEREBRA_EHR_HEADER_PATH`` (single override).
3. Size-based env vars:
   - ``CEREBRA_EHR_HEADER_PATH_SMALL`` (< 5 000 features)
   - ``CEREBRA_EHR_HEADER_PATH_LARGE`` (>= 5 000 features)
4. FileNotFoundError with a clear message.
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Module-level cache: path -> list of header strings
_HEADER_CACHE: dict = {}

FEATURE_SIZE_THRESHOLD = 5000


def load_ehr_headers(
    n_features: Optional[int] = None,
    explicit_path: Optional[str] = None,
) -> List[str]:
    """Return a list of EHR header strings.

    Parameters
    ----------
    n_features : int, optional
        Number of features in the dataset.  Used to pick the correct
        size-based env var when *explicit_path* is not given.
    explicit_path : str, optional
        If provided, this path is used directly (highest priority).

    Returns
    -------
    list[str]
        Header names, one per feature column.

    Raises
    ------
    FileNotFoundError
        If no header file can be located.
    """
    path = _resolve_path(n_features, explicit_path)

    if path in _HEADER_CACHE:
        logger.debug("EHR headers served from cache (%s)", path)
        return _HEADER_CACHE[path]

    logger.info("Loading EHR headers from %s", path)
    with open(path, "r") as fh:
        headers = [line for line in fh.read().split("\n") if line]

    _HEADER_CACHE[path] = headers

    if n_features is not None and len(headers) < n_features:
        logger.warning(
            "Header file has %d entries but data has %d features; "
            "some features will lack names.",
            len(headers),
            n_features,
        )

    return headers


def _resolve_path(
    n_features: Optional[int],
    explicit_path: Optional[str],
) -> str:
    """Determine which header file to use."""

    # 1. Caller-supplied path
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f"Explicit EHR header path does not exist: {explicit_path}"
            )
        return explicit_path

    # 2. Single env-var override
    env_single = os.environ.get("CEREBRA_EHR_HEADER_PATH")
    if env_single:
        if not os.path.isfile(env_single):
            raise FileNotFoundError(
                f"CEREBRA_EHR_HEADER_PATH points to a missing file: {env_single}"
            )
        return env_single

    # 3. Size-based env vars
    if n_features is not None and n_features < FEATURE_SIZE_THRESHOLD:
        env_key = "CEREBRA_EHR_HEADER_PATH_SMALL"
    else:
        env_key = "CEREBRA_EHR_HEADER_PATH_LARGE"

    env_val = os.environ.get(env_key)
    if env_val:
        if not os.path.isfile(env_val):
            raise FileNotFoundError(
                f"{env_key} points to a missing file: {env_val}"
            )
        return env_val

    # 4. Nothing found
    raise FileNotFoundError(
        "No EHR header file configured. Set one of the following env vars:\n"
        "  CEREBRA_EHR_HEADER_PATH          (single file for all sizes)\n"
        "  CEREBRA_EHR_HEADER_PATH_SMALL     (< 5000 features)\n"
        "  CEREBRA_EHR_HEADER_PATH_LARGE     (>= 5000 features)\n"
        "Or pass an explicit path via the ehr_header_path argument."
    )
