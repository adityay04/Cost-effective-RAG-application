import json
import re
from pathlib import Path
from statistics import mean

from groq import Groq

from app.config import settings
from app.generation.answer import ask_question


DATASET_PATH = Path("evaluation/dataset.json")
RESULTS_PATH = Path("evaluation/results/answer_results.json")
AUDIT_PATH = Path("logs/judge_audit.jsonl")

judge_client = Groq(api_key=settings.groq_api_key)

DEFAULT_CRITERIA = [
    "faithfulness",
    "answer_relevance",
    "completeness",
    "instruction_following"
]


def extract_json(raw_response):
    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            raise ValueError("Judge response did not contain valid JSON")

        return json.loads(match.group(0))


def build_judge_prompt(
    question,
    context,
    answer,
    expected_answer,
    criteria
):
    evaluation_input = {
        "question": question,
        "context": context,
        "model_answer": answer,
        "expected_answer": expected_answer,
        "criteria": criteria
    }

    rubric = {
        "faithfulness": "The answer is supported by the provided context.",
        "answer_relevance": "The answer directly addresses the question.",
        "completeness": "The answer covers the important information available in the context.",
        "instruction_following": "The answer follows the system instruction to use only context and cite sources.",
        "abstention": "When context is insufficient, the answer clearly abstains instead of inventing information."
    }

    return f"""
You are an impartial evaluator of a retrieval-augmented generation system.

Evaluate the model answer using only the supplied evaluation input.

Score every requested criterion from 1 to 5.

Score definitions:
1 = Very poor
2 = Poor
3 = Partially satisfactory
4 = Good
5 = Excellent

Rubric:
{json.dumps(rubric, indent=2)}

Requirements:
- Faithfulness must be judged against the supplied context.
- Do not reward unsupported claims.
- Answer relevance must be judged against the question.
- Completeness must consider only information supported by the context.
- If an expected answer is supplied, use it as a reference, not as a replacement for the context.
- If the context is empty, evaluate whether the model appropriately abstained.
- Give a short, concrete rationale for every criterion.
- Return valid JSON only.
- Do not include Markdown fences.

Required JSON structure:
{{
  "scores": {{
    "criterion_name": {{
      "score": 1,
      "rationale": "Specific explanation"
    }}
  }},
  "overall_score": 1,
  "overall_rationale": "Short explanation"
}}

Evaluation input:
{json.dumps(evaluation_input, indent=2)}
"""


def validate_verdict(verdict, criteria):
    scores = verdict.get("scores")

    if not isinstance(scores, dict):
        raise ValueError("Judge verdict is missing scores")

    validated_scores = {}

    for criterion in criteria:
        result = scores.get(criterion)

        if not isinstance(result, dict):
            raise ValueError(f"Missing score for criterion: {criterion}")

        score = result.get("score")
        rationale = result.get("rationale")

        if not isinstance(score, (int, float)) or not 1 <= score <= 5:
            raise ValueError(f"Invalid score for criterion: {criterion}")

        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"Missing rationale for criterion: {criterion}")

        validated_scores[criterion] = {
            "score": float(score),
            "rationale": rationale.strip()
        }

    overall_score = verdict.get("overall_score")

    if not isinstance(overall_score, (int, float)):
        raise ValueError("Missing overall score")

    if not 1 <= overall_score <= 5:
        raise ValueError("Overall score must be between 1 and 5")

    overall_rationale = verdict.get("overall_rationale", "")

    return {
        "scores": validated_scores,
        "overall_score": float(overall_score),
        "overall_rationale": str(overall_rationale)
    }


def judge_answer(
    question,
    context,
    answer,
    expected_answer=None,
    criteria=None
):
    criteria = criteria or DEFAULT_CRITERIA

    if not context.strip() and "abstention" not in criteria:
        criteria = criteria + ["abstention"]

    prompt = build_judge_prompt(
        question=question,
        context=context,
        answer=answer,
        expected_answer=expected_answer,
        criteria=criteria
    )

    raw_response = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for attempt in range(2):
        response = judge_client.chat.completions.create(
            model=settings.judge_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        raw_response = response.choices[0].message.content or ""

        usage = response.usage

        if usage:
            total_prompt_tokens += usage.prompt_tokens or 0
            total_completion_tokens += usage.completion_tokens or 0

        AUDIT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with AUDIT_PATH.open("a", encoding="utf-8") as audit_file:
            audit_file.write(
                json.dumps({
                    "judge_model": settings.judge_model,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "attempt": attempt + 1
                }) + "\n"
            )

        try:
            verdict = extract_json(raw_response)
            validated = validate_verdict(verdict, criteria)

            return {
                **validated,
                "judge_calls": attempt + 1,
                "judge_prompt_tokens": total_prompt_tokens,
                "judge_completion_tokens": total_completion_tokens
            }

        except (ValueError, json.JSONDecodeError):
            if attempt == 1:
                raise

            prompt += "\nReturn corrected valid JSON matching the required structure."


def evaluate():
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    case_results = []

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        expected_answer = item.get("expected_answer")
        criteria = item.get("criteria", DEFAULT_CRITERIA)

        generation_result = ask_question(question)

        verdict = judge_answer(
            question=question,
            context=generation_result["context"],
            answer=generation_result["answer"],
            expected_answer=expected_answer,
            criteria=criteria
        )

        criterion_scores = {
            name: result["score"]
            for name, result in verdict["scores"].items()
        }

        passed = all(
            score >= 4
            for score in criterion_scores.values()
        )

        case_results.append({
            "id": question_id,
            "question": question,
            "expected_answer": expected_answer,
            "answer": generation_result["answer"],
            "context": generation_result["context"],
            "citations": generation_result["citations"],
            "generation_metrics": {
                "latency_ms": generation_result["latency_ms"],
                "prompt_tokens": generation_result["prompt_tokens"],
                "completion_tokens": generation_result["completion_tokens"],
                "total_tokens": generation_result["total_tokens"]
            },
            "judge_metrics": {
                "judge_model": settings.judge_model,
                "judge_calls": verdict["judge_calls"],
                "prompt_tokens": verdict["judge_prompt_tokens"],
                "completion_tokens": verdict["judge_completion_tokens"]
            },
            "scores": verdict["scores"],
            "overall_score": verdict["overall_score"],
            "overall_rationale": verdict["overall_rationale"],
            "passed": passed
        })

    criterion_names = sorted({
        criterion
        for result in case_results
        for criterion in result["scores"]
    })

    mean_scores = {}

    for criterion in criterion_names:
        values = [
            result["scores"][criterion]["score"]
            for result in case_results
            if criterion in result["scores"]
        ]

        mean_scores[criterion] = round(mean(values), 4) if values else None

    total_generation_tokens = sum(
        result["generation_metrics"]["total_tokens"]
        for result in case_results
    )

    total_judge_tokens = sum(
        result["judge_metrics"]["prompt_tokens"]
        + result["judge_metrics"]["completion_tokens"]
        for result in case_results
    )

    total_judge_calls = sum(
        result["judge_metrics"]["judge_calls"]
        for result in case_results
    )

    summary = {
        "total_cases": len(case_results),
        "pass_rate": round(
            sum(result["passed"] for result in case_results)
            / len(case_results),
            4
        ) if case_results else 0,
        "mean_overall_score": round(
            mean(result["overall_score"] for result in case_results),
            4
        ) if case_results else None,
        "mean_criterion_scores": mean_scores,
        "total_generation_tokens": total_generation_tokens,
        "total_judge_tokens": total_judge_tokens,
        "total_judge_calls": total_judge_calls,
        "generator_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "results": case_results
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps({
        "pass_rate": summary["pass_rate"],
        "mean_overall_score": summary["mean_overall_score"],
        "mean_criterion_scores": summary["mean_criterion_scores"],
        "total_generation_tokens": summary["total_generation_tokens"],
        "total_judge_tokens": summary["total_judge_tokens"],
        "total_judge_calls": summary["total_judge_calls"]
    }, indent=2))

    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Audit log saved to: {AUDIT_PATH}")


if __name__ == "__main__":
    evaluate()
