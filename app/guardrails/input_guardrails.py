
import re
from typing import Tuple


MAX_QUERY_LENGTH = 2000


def validate_input(query: str) -> Tuple[bool, str]:

    if not query or not query.strip():
        return False, "Please enter a question."

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        return False, "Your question is too long."

    # Example: reject obvious credential-sharing requests
    sensitive_patterns = [
        r"password\s*[:=]\s*\S+",
        r"api[_-]?key\s*[:=]\s*\S+",
        r"secret\s*[:=]\s*\S+",
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, (
                "Please do not share passwords, API keys, "
                "or other credentials."
            )

    return True, ""
