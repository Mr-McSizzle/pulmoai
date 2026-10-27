"""Conservative longitudinal comparisons. Model scores are not disease severity."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

COMPARISON_VERSION = "longitudinal-v2"
NOTICE = ("Lab direction describes movement relative to a reference interval, not overall health. "
          "Image scores describe model output, not disease severity or treatment response. "
          "Only reviewed, dated, compatible observations are compared.")


def _bounds(text: str) -> tuple[float | None, float | None] | None:
    text = text.strip().replace("–", "-").replace("−", "-").replace("≤", "<=").replace("≥", ">=")
    number = r"(\d+(?:\.\d+)?)"
    both = re.fullmatch(number + r"\s*-\s*" + number + r"(?:\s+[a-zA-Z%/\^\d]+)?", text)
    if both:
        lo, hi = map(float, both.groups())
        return (lo, hi) if hi > lo else None
    one = re.fullmatch(r"([<>]=?)\s*" + number + r"(?:\s+[a-zA-Z%/\^\d]+)?", text)
    if one:
        op, value = one.groups()
        return (None, float(value)) if op.startswith("<") else (float(value), None)
    return None


def _distance(value: float, bounds: tuple[float | None, float | None]) -> float:
    lo, hi = bounds
    return max((lo - value) if lo is not None else 0, (value - hi) if hi is not None else 0, 0)


def build_timeline(patient: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    labs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    images: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded = 0
    for record in records:
        if record.get("review_status") not in {"reviewed", "partially_reviewed"} or not record.get("observed_on"):
            excluded += 1
            continue
        result = record["result"]
        base = {"date": record["observed_on"], "record_id": record["id"], "filename": record["filename"]}
        if record["kind"] == "lab_report":
            for item in result.get("extracted_tests", []):
                if record.get("review_status") == "partially_reviewed" and item["canonical_name"] not in result.get("reviewed_tests", []):
                    continue
                value = result.get("reviewed_values", {}).get(item["canonical_name"], item["value"])
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not item.get("unit"):
                    continue
                labs[(item["canonical_name"], item["unit"])].append({**base, "value": value,
                    "display_name": item["display_name"], "reference_range": item["reference_range"],
                    "corrected": value != item["value"]})
        elif record["kind"] == "medical_image":
            if result.get("status") != "success" or not result.get("recognized") or not record.get("series_label"):
                continue
            version = result.get("artifact_id")
            if not version or result.get("staging_notice"):
                continue
            for label, score in (result.get("class_probabilities") or result.get("probabilities") or {}).items():
                if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                    continue
                images[(result["image_type"], record["series_label"], version, label)].append({**base, "value": score * 100})

    lab_series = []
    for (name, unit), points in sorted(labs.items()):
        points.sort(key=lambda p: (p["date"], p["record_id"]))
        first, last = points[0], points[-1]
        direction, reason = "insufficient_data", "Two reviewed observations on different dates are required."
        change = None
        if len(points) >= 2 and first["date"] != last["date"]:
            endpoints_unique = sum(p["date"] == first["date"] for p in points) == 1 and sum(p["date"] == last["date"] for p in points) == 1
            change = round(last["value"] - first["value"], 6) if endpoints_unique else None
            a, b = _bounds(first["reference_range"]), _bounds(last["reference_range"])
            if not endpoints_unique:
                direction, reason = "not_comparable", "Multiple observations share a baseline or latest date. A directional comparison is ambiguous."
            elif not a or a != b:
                direction, reason = "numeric_change_only", "Reference intervals differ or cannot be parsed; only the numeric change is shown."
            else:
                d1, d2 = _distance(first["value"], a), _distance(last["value"], b)
                if math.isclose(d1, d2, abs_tol=1e-8):
                    direction = "within_range" if d2 == 0 else "unchanged_distance"
                else:
                    direction = "toward_range" if d2 < d1 else "away_from_range"
                reason = "Baseline to latest, using the same reference interval and normalized unit."
        lab_series.append({"id": f"{name}:{unit}", "name": first["display_name"], "unit": unit,
                           "points": points, "direction": direction, "reason": reason, "change": change})
    image_series = []
    for (modality, series, version, label), points in sorted(images.items()):
        points.sort(key=lambda p: (p["date"], p["record_id"]))
        unique_dates = len({p["date"] for p in points}) == len(points)
        change = round(points[-1]["value"] - points[0]["value"], 2) if len(points) > 1 and unique_dates else None
        image_series.append({"id": f"{modality}:{series}:{version}:{label}", "name": label,
            "modality": modality, "series_label": series, "artifact_id": version,
            "unit": "% model score", "points": points, "change": change,
            "direction": "score_change_only", "reason": "Same modality, acquisition series and model artifact. A score change does not establish improvement or decline."})
    assessment, risk_series, rna_history = combined_assessment(records, lab_series, image_series)
    return {"assessment": assessment, "risk_series": risk_series, "rna_history": rna_history, "patient": patient, "records": records, "lab_series": lab_series, "image_series": image_series,
            "excluded_unreviewed": excluded, "comparison_version": COMPARISON_VERSION, "notice": NOTICE}


def _output(record: dict) -> dict:
    result = record.get("result", {})
    return result.get("result", result) if isinstance(result.get("result", result), dict) else {}


def _probability(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1 else None


def combined_assessment(records: list[dict], lab_series: list[dict], image_series: list[dict]) -> tuple[dict, list, list]:
    """Summarize recorded evidence without inventing a fused diagnosis or health score."""
    ordered = sorted(records, key=lambda r: (r.get("observed_on") or "", r.get("uploaded_at") or "", r["id"]))
    coverage = {kind: sum(r["kind"] == kind for r in records) for kind in
                ("scrna_csv", "lab_report", "risk_assessment", "future_risk_assessment", "medical_image")}
    labs = []
    for series in lab_series:
        point = series["points"][-1]
        bounds = _bounds(point.get("reference_range", ""))
        flag = "UNKNOWN"
        if bounds:
            low, high = bounds
            flag = "LOW" if low is not None and point["value"] < low else "HIGH" if high is not None and point["value"] > high else "NORMAL"
        labs.append({"name": series["name"], "unit": series["unit"], "value": point["value"], "flag": flag,
                     "reference_range": point.get("reference_range", ""), "date": point["date"], "record_id": point["record_id"],
                     "direction": series["direction"], "change": series["change"]})
    latest_risk, latest_future, latest_images = {}, {}, {}
    rna = []
    risk_points = defaultdict(list)
    for record in ordered:
        result = _output(record)
        base = {"record_id": record["id"], "date": record.get("observed_on"), "review_status": record.get("review_status", "unreviewed"),
                "data_origin": record.get("result", {}).get("data_origin", result.get("data_origin", "user_supplied_unverified"))}
        artifact = record.get("result", {}).get("artifact_id") or result.get("artifact_id")
        if record["kind"] == "scrna_csv" and result.get("predicted_class"):
            rna.append({**base, "predicted_class": result["predicted_class"], "artifact_id": artifact,
                        "sex_context": result.get("sex_context"), "interpretation": result.get("interpretation", "Research classification only.")})
        if record["kind"] == "risk_assessment":
            for key, item in result.get("risk_predictions", {}).items():
                probability = _probability(item.get("calibrated_risk"))
                latest_risk[key] = {**base, "name": item.get("display_name", key), "probability": probability,
                    "tier": item.get("risk_category", "Indeterminate"), "horizon": item.get("time_horizon", "Existing status"),
                    "estimand": item.get("estimand_label", "Existing or reported diagnosis status")}
                if probability is not None and artifact and record.get("review_status") == "reviewed" and base["date"]:
                    risk_points[("current", key, artifact, item.get("time_horizon", "Existing status"))].append(
                        {**base, "filename": record["filename"], "value": probability * 100, "name": item.get("display_name", key)})
        if record["kind"] == "future_risk_assessment":
            for key, item in result.get("future_endpoints", {}).items():
                for horizon, value in item.get("time_horizons", {}).items():
                    probability = _probability(value.get("risk_probability"))
                    latest_future[(key, horizon)] = {**base, "name": item.get("display_name", key), "probability": probability,
                        "tier": item.get("risk_tier_5yr", "Indeterminate") if horizon == "5_year" else "Not tiered",
                        "horizon": horizon.replace("_", " "), "estimand": item.get("estimand_label", "Future mortality estimate")}
                    if probability is not None and artifact and record.get("review_status") == "reviewed" and base["date"]:
                        risk_points[("future", key, artifact, horizon)].append({**base, "filename": record["filename"], "value": probability * 100,
                                                                             "name": item.get("display_name", key)})
        if record["kind"] == "medical_image":
            modality = result.get("image_type", "unspecified")
            series = record.get("series_label", "")
            accepted = result.get("recognized") and result.get("status") == "success" and not result.get("staging_notice")
            latest_images[(modality, series)] = {**base, "modality": modality, "series_label": series,
                "status": result.get("status", "unknown"), "predicted_class": result.get("predicted_class") if accepted else None,
                "active_conditions": result.get("active_conditions", []) if accepted else [],
                "message": result.get("message") or ("Accepted research model output." if accepted else "No interpretable screening result."),
                "artifact_id": artifact}
    risk_series = []
    for (scope, key, artifact, horizon), points in sorted(risk_points.items()):
        points.sort(key=lambda p: (p["date"], p["record_id"]))
        unambiguous = len({p["date"] for p in points}) == len(points)
        change = round(points[-1]["value"] - points[0]["value"], 2) if len(points) >= 2 and unambiguous else None
        risk_series.append({"id": f"{scope}:{key}:{artifact}:{horizon}", "name": f"{points[0]['name']} · {horizon.replace('_', ' ')}",
                            "unit": "% model estimate", "points": points, "change": change, "direction": "estimate_change_only",
                            "artifact_id": artifact, "reason": "Reviewed observations using the same model, target and horizon. A changed estimate does not establish clinical improvement or decline."})
    unreviewed = sum(r.get("review_status") != "reviewed" for r in records)
    flagged = [item for item in labs if item["flag"] in {"HIGH", "LOW"}]
    high_tiers = [item for item in list(latest_risk.values()) + list(latest_future.values()) if str(item["tier"]).casefold() == "high"]
    gaps = []
    if not coverage["lab_report"]: gaps.append("No lab reports linked to this patient.")
    if coverage["lab_report"] and not labs: gaps.append("Lab values await source confirmation or a usable unit.")
    if not coverage["scrna_csv"]: gaps.append("No RNA classification is linked.")
    if not coverage["risk_assessment"]: gaps.append("No chronic-risk calculation is linked.")
    if not coverage["future_risk_assessment"]: gaps.append("No future-risk calculation is linked.")
    if not coverage["medical_image"]: gaps.append("No imaging observations are linked.")
    if any(item["status"] != "success" for item in latest_images.values()): gaps.append("Some images are archived, rejected or have no accepted screening output.")
    if unreviewed: gaps.append(f"{unreviewed} observations still require full source/input review.")
    interpreted = bool(labs or rna or any(item["probability"] is not None for item in list(latest_risk.values()) + list(latest_future.values())) or any(item["status"] == "success" and item["predicted_class"] for item in latest_images.values()))
    if not records or not interpreted:
        headline, status = "No interpreted measurements or accepted model findings are available yet.", "insufficient_evidence"
    else:
        headline = f"{len(flagged)} reviewed lab values outside reported intervals; {len(high_tiers)} high-tier model estimates."
        status = "review_required" if unreviewed else "review_priorities" if flagged or high_tiers else "observations_available"
    assessment = {"title": "Overall research assessment", "status": status, "headline": headline,
        "as_of": ordered[-1].get("observed_on") if ordered else None,
        "generated_example_count": sum(r.get("result", {}).get("data_origin") == "generated_example" for r in records),
        "coverage": coverage, "unreviewed_count": unreviewed, "labs": labs,
        "chronic_risks": list(latest_risk.values()), "future_risks": list(latest_future.values()),
        "rna_findings": rna[-10:], "image_findings": list(latest_images.values()), "gaps": gaps,
        "trend_summary": {"toward_range": sum(s["direction"] == "toward_range" for s in lab_series),
                          "away_from_range": sum(s["direction"] == "away_from_range" for s in lab_series),
                          "comparable_image_series": sum(s["change"] is not None for s in image_series),
                          "comparable_risk_series": sum(s["change"] is not None for s in risk_series)},
        "interpretation": "This consolidates recorded measurements and model findings; it is not a medical diagnosis. No validated multimodal fusion model or combined disease probability is deployed. Existing-status risk, future mortality estimates, RNA classes and image labels remain distinct."}
    return assessment, risk_series, rna
