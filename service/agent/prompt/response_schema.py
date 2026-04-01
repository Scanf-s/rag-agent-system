RESPONSE_SCHEMA: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "legal_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "간결한 풀이 과정 (각 선지 O/X 판정 포함)",
                },
                "answer": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "최종 정답",
                },
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False,
        },
    },
}
