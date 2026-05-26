"""Preprocessing: categorize examples and format CoT prompt templates."""
import re

CATEGORY_MAP = {
    "bit_manipulation": "bit manipulation rule",
    "cipher":           "encryption rules",
    "number_system":    "numeral system",
    "unit_conversion":  "unit conversion",
    "physics":          "gravitational constant",
    "equation":         "transformation rules is applied to equations",
}

CATEGORY_HINTS = {
    "bit_manipulation": "Focus on bit-level operations: XOR, AND, OR, NOT, shifts (left/right), and rotations on 8-bit binary strings.",
    "cipher":           "Focus on character-level substitution, Caesar-shift variants, word reordering, or mapping rules between words.",
    "number_system":    "Focus on base conversions (binary, octal, hex, decimal) or classical numeral systems (Roman numerals, etc.).",
    "unit_conversion":  "Focus on unit conversion factors — the constant may be non-standard. Derive the multiplier from the examples.",
    "physics":          "Focus on the modified constant. Derive its value from the example observations using the relevant physical formula.",
    "equation":         "Focus on operator substitution, symbol remapping, or algebraic transformations applied to the expression.",
}

SYSTEM_PROMPT = (
    "You are an expert at identifying hidden rules from input-output examples.\n"
    "Given a puzzle, reason step by step:\n"
    "  1. Study each example carefully.\n"
    "  2. Hypothesize the hidden rule.\n"
    "  3. Verify your hypothesis against all examples.\n"
    "  4. Apply the rule to the final input.\n"
    "Always end your response with exactly:\n"
    "Answer: <your answer>"
)

TRAINING_TEMPLATE = (
    "### Problem:\n{prompt}\n\n"
    "### Reasoning:\n{cot}\n\n"
    "### Answer:\n{answer}"
)


def categorize(prompt: str) -> str:
    first_line = prompt.split("\n")[0]
    for cat, marker in CATEGORY_MAP.items():
        if marker in first_line:
            return cat
    return "other"


def build_teacher_user_message(prompt: str, category: str) -> str:
    hint = CATEGORY_HINTS.get(category, "")
    hint_line = f"\nHint: {hint}\n" if hint else ""
    return (
        f"{prompt}"
        f"{hint_line}\n"
        "Reason step by step to identify the hidden rule, then apply it to the final input.\n"
        "End your response with exactly: Answer: <answer>"
    )


def extract_answer(response_text: str) -> str | None:
    """Parse the last 'Answer: X' line from the model response."""
    matches = re.findall(r"Answer:\s*(.+)", response_text, re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def format_training_example(prompt: str, cot: str, answer: str) -> str:
    return TRAINING_TEMPLATE.format(prompt=prompt, cot=cot, answer=answer)
