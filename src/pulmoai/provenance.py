"""Preserve declared example-data provenance without asserting uploaded data is clinically verified."""
from typing import Literal

DataOrigin = Literal["user_supplied_unverified", "generated_example"]


def data_origin(contents: bytes, filename: str, declared: DataOrigin = "user_supplied_unverified") -> DataOrigin:
    prefix = contents[:4096].lower()
    name = filename.lower()
    if declared == "generated_example" or name.startswith(("synthetic_", "synthetic-")):
        return "generated_example"
    if any(marker in prefix for marker in (b"synthetic demo", b"generated example", b"synthetic lab report")):
        return "generated_example"
    return "user_supplied_unverified"
