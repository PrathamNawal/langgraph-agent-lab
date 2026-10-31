"""Real eval set for Module 1's calculator agent: arithmetic questions with
exact, hand-computed ground-truth answers. Used by the PM dashboard's
"Try it - Human" / "Try it - Agent" pages and by Track Performance's accuracy
metric - every item here has a verifiable correct answer, so "accuracy" always
means an exact match, never a judgment call.
"""

EVAL_QUESTIONS = [
    {"id": "q1", "question": "What is 847 * 23?", "answer": 19481},
    {"id": "q2", "question": "What is 123 * 9?", "answer": 1107},
    {"id": "q3", "question": "What is 56 + 78?", "answer": 134},
    {"id": "q4", "question": "What is 1000 divided by 8?", "answer": 125},
    {"id": "q5", "question": "What is 15% of 240?", "answer": 36},
    {"id": "q6", "question": "What is 2 + 2 * 10?", "answer": 22},
    {"id": "q7", "question": "What is 500 minus 237?", "answer": 263},
    {"id": "q8", "question": "A shirt costs $45 and is on sale for 20% off. What's the sale price?", "answer": 36},
    {"id": "q9", "question": "A team of 7 splits a $154 bill evenly. How much does each person owe?", "answer": 22},
    {"id": "q10", "question": "What is 123456 + 654321?", "answer": 777777},
]


def get_by_id(item_id):
    return next(item for item in EVAL_QUESTIONS if item["id"] == item_id)
