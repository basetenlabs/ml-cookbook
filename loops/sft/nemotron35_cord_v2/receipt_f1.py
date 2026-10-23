import json
import re
from collections import Counter


def parse_tags(text):
    cursor = 0

    def parse_object(end_key=None):
        nonlocal cursor
        objects = [{}]
        while cursor < len(text):
            if end_key and text.startswith(f"</s_{end_key}>", cursor):
                cursor += len(f"</s_{end_key}>")
                return objects[0] if len(objects) == 1 else objects
            if text.startswith("<sep/>", cursor):
                objects.append({})
                cursor += len("<sep/>")
                continue
            match = re.match(r"<s_([^<>]+)>", text[cursor:])
            if not match:
                raise ValueError("Unexpected text outside a receipt field.")
            key = match.group(1)
            cursor += len(match.group())
            if text.startswith("<s_", cursor):
                value = parse_object(key)
            else:
                ending = text.find(f"</s_{key}>", cursor)
                if ending < 0:
                    raise ValueError("Unclosed receipt field.")
                value = text[cursor:ending]
                if "<s_" in value or "</s_" in value:
                    raise ValueError("Malformed receipt nesting.")
                value = value.split("<sep/>") if "<sep/>" in value else value
                cursor = ending + len(f"</s_{key}>")
            if key in objects[-1]:
                raise ValueError("Duplicate receipt key.")
            objects[-1][key] = value
        if end_key:
            raise ValueError("Unclosed receipt object.")
        return objects[0] if len(objects) == 1 else objects

    result = parse_object()
    if not isinstance(result, dict) or not result:
        raise ValueError("Expected a nonempty receipt object.")
    return result


def parse_prediction(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("Unfinished reasoning; no final receipt parse.")
    text = text.strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.removesuffix(marker).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    if text.startswith("{"):
        result = json.loads(text)
        if not isinstance(result, dict) or not result:
            raise ValueError("Expected a nonempty receipt JSON object.")
        return result
    return parse_tags(text)


def fields(value, prefix=""):
    result = Counter()
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(fields(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for child in value:
            result.update(fields(child, prefix))
    elif isinstance(value, (str, int, float)) and str(value).strip():
        result[(prefix, str(value).strip())] += 1
    return result


def score(rows):
    true_positive = predicted = expected = invalid = truncated = 0
    for row in rows:
        try:
            prediction = parse_prediction(row["text"])
        except (ValueError, TypeError):
            prediction = {}
            invalid += 1
        predicted_fields = fields(prediction)
        expected_fields = fields(row["ground_truth"])
        true_positive += sum((predicted_fields & expected_fields).values())
        predicted += sum(predicted_fields.values())
        expected += sum(expected_fields.values())
        truncated += row["finish_reason"] == "length"
    return {
        "rows": len(rows),
        "true_positive": true_positive,
        "predicted_fields": predicted,
        "ground_truth_fields": expected,
        "precision": true_positive / predicted if predicted else 0.0,
        "recall": true_positive / expected if expected else 0.0,
        "f1": 2 * true_positive / (predicted + expected)
        if predicted + expected
        else 0.0,
        "invalid_outputs": invalid,
        "output_truncations": truncated,
        "metric": "Micro exact receipt field-path/value F1 (case-sensitive, whitespace stripped, list order ignored)",
        "invalid_output_policy": "Empty prediction; all ground-truth fields counted as missed.",
    }
