"""Manager for traveler/public submitted questions in TravelMate."""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import uuid

DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "data" / "submitted_questions.json"


@dataclass
class SubmittedQuestion:
    """A question posted by a traveler or public user."""

    id: str
    question: str
    submitted_by: str = "Traveler"
    category: str = "General"
    submitted_date: str = ""
    status: str = "Pending"  # "Pending", "Answered", "Dismissed"
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubmittedQuestion":
        return cls(
            id=str(data.get("id") or str(uuid.uuid4())[:8]),
            question=str(data.get("question", "")).strip(),
            submitted_by=str(data.get("submitted_by", "Traveler")).strip(),
            category=str(data.get("category", "General")).strip(),
            submitted_date=str(data.get("submitted_date") or datetime.now().strftime("%Y-%m-%d")),
            status=str(data.get("status", "Pending")),
            answer=data.get("answer"),
        )


def _get_seed_questions() -> list[SubmittedQuestion]:
    """Default sample submitted questions for demonstration."""
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        SubmittedQuestion(
            id="sq-101",
            question="Is the campus hostel wifi available 24/7 or is there a curfew?",
            submitted_by="Alex (First Year)",
            category="Travel tips",
            submitted_date=today,
            status="Pending",
        ),
        SubmittedQuestion(
            id="sq-102",
            question="Can final year travelers get parking permits for four-wheelers?",
            submitted_by="Priya (B.Tech)",
            category="Traveler services",
            submitted_date=today,
            status="Pending",
        ),
    ]


def load_submitted_questions(file_path: str | Path = DEFAULT_QUESTIONS_PATH) -> list[SubmittedQuestion]:
    """Load submitted questions from JSON file."""
    path = Path(file_path)
    if not path.exists():
        seeds = _get_seed_questions()
        save_submitted_questions(seeds, path)
        return seeds

    try:
        raw_text = path.read_text(encoding="utf-8")
        if not raw_text.strip():
            return []
        data = json.loads(raw_text)
        if isinstance(data, list):
            return [SubmittedQuestion.from_dict(item) for item in data]
        return []
    except Exception:
        seeds = _get_seed_questions()
        save_submitted_questions(seeds, path)
        return seeds


def save_submitted_questions(questions: list[SubmittedQuestion], file_path: str | Path = DEFAULT_QUESTIONS_PATH) -> None:
    """Save submitted questions to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [q.to_dict() for q in questions]
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def submit_question(
    question: str,
    submitted_by: str = "Traveler",
    category: str = "General",
    file_path: str | Path = DEFAULT_QUESTIONS_PATH,
) -> SubmittedQuestion:
    """Submit a new public question."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    questions = load_submitted_questions(file_path)
    new_id = f"sq-{uuid.uuid4().hex[:6]}"
    today = datetime.now().strftime("%Y-%m-%d")

    entry = SubmittedQuestion(
        id=new_id,
        question=question.strip(),
        submitted_by=submitted_by.strip() or "Traveler",
        category=category.strip() or "General",
        submitted_date=today,
        status="Pending",
        answer=None,
    )
    questions.insert(0, entry)
    save_submitted_questions(questions, file_path)
    return entry


def mark_question_answered(
    question_id: str,
    answer: str,
    file_path: str | Path = DEFAULT_QUESTIONS_PATH,
) -> bool:
    """Mark a submitted question as answered."""
    questions = load_submitted_questions(file_path)
    for q in questions:
        if q.id == question_id:
            q.status = "Answered"
            q.answer = answer.strip()
            save_submitted_questions(questions, file_path)
            return True
    return False


def dismiss_question(
    question_id: str,
    file_path: str | Path = DEFAULT_QUESTIONS_PATH,
) -> bool:
    """Dismiss or archive a submitted question."""
    questions = load_submitted_questions(file_path)
    for q in questions:
        if q.id == question_id:
            q.status = "Dismissed"
            save_submitted_questions(questions, file_path)
            return True
    return False


def delete_submitted_question(
    question_id: str,
    file_path: str | Path = DEFAULT_QUESTIONS_PATH,
) -> bool:
    """Permanently delete a submitted question."""
    questions = load_submitted_questions(file_path)
    initial_length = len(questions)
    questions = [q for q in questions if q.id != question_id]
    if len(questions) != initial_length:
        save_submitted_questions(questions, file_path)
        return True
    return False


def get_pending_questions(file_path: str | Path = DEFAULT_QUESTIONS_PATH) -> list[SubmittedQuestion]:
    """Get all questions waiting for admin answer."""
    return [q for q in load_submitted_questions(file_path) if q.status == "Pending"]
