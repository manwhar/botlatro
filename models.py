from dataclasses import dataclass

from matcher import TemplateMatch


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    template_name: str
    found: bool
    match: TemplateMatch | None
    clicked: bool | None
    template_filename: str | None = None
