from typing import Any, Dict, List, Set

from .base_evaluator import BaseEvaluator
from seokar import Seokar
import logging


def _extract_content_issues(report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract only issues relevant to content structure and readability.
    """
    content_elements: Set[str] = {
        "Title",
        "Meta Description",
        "H1 Tag",
        "H1 Content",
        "Headings Hierarchy",
        "Content Readability",
        "Content Quality",
    }

    issues = report_data.get("issues", [])
    return [
        issue
        for issue in issues
        if (issue.get("element_type") in content_elements and issue.get("recommendation") is not None)
    ]


class SeoEvaluator(BaseEvaluator):
    async def evaluate(self, html_content: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate SEO aspects of HTML content using seokar.
        Returns score and findings from the report, without exception handling.
        """
        logging.disable(level=logging.ERROR)
        analyzer = Seokar(html_content=html_content)
        report = analyzer.analyze()
        logging.disable(level=logging.NOTSET)

        score_value = report["seo_health"]["score"]
        if isinstance(score_value, str):
            try:
                score = float(score_value) / 100.0
            except ValueError:
                score = 0.5
        else:
            score = float(score_value) if score_value is not None else 0.5

        findings = _extract_content_issues(report_data=report)
        return {
            "evaluator": "SeoEvaluator",
            "score": score,
            "passed": score >= 0.7,
            "findings": findings,
        }
