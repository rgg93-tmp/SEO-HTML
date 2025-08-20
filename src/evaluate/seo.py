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
    return [issue for issue in issues if issue.get("element_type") in content_elements]


class SeoEvaluator(BaseEvaluator):
    async def evaluate(self, html_content: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate SEO aspects of HTML content using seokar.

        Args:
            html_content: HTML content to analyze
            **kwargs: Additional parameters

        Returns:
            Standardized evaluation results dictionary
        """
        try:
            logging.disable(logging.ERROR)
            analyzer = Seokar(html_content=html_content)
            report = analyzer.analyze()
            logging.disable(logging.NOTSET)

            score_value = report.get("score", 0.5)
            if isinstance(score_value, str):
                try:
                    score = float(score_value) / 100.0
                except ValueError:
                    score = 0.5
            else:
                score = float(score_value) if score_value is not None else 0.5

            findings = []

            # Process structured content issues
            content_issues = _extract_content_issues(report)
            for issue in content_issues:
                severity = issue.get("severity", "info").lower()
                findings.append(
                    {
                        "component": f"SEO {issue.get('element_type', 'Issue')}",
                        "message": issue.get("message", ""),
                        "severity": severity,
                    }
                )

            # Process unstructured report data as a fallback
            for key, severity, component_name in [
                ("warnings", "warning", "SEO Warning"),
                ("errors", "error", "SEO Error"),
                ("suggestions", "info", "SEO Suggestion"),
            ]:
                for item in report.get(key, []):
                    if isinstance(item, str):
                        findings.append({"component": component_name, "message": item, "severity": severity})

            # Remove duplicates
            unique_findings = [dict(t) for t in {tuple(d.items()) for d in findings}]

            return {
                "evaluator": "SeoEvaluator",
                "score": score,
                "passed": score >= 0.7,
                "summary": f"SEO analysis completed with score: {score:.2f}",
                "findings": unique_findings,
            }

        except Exception as e:
            # Fallback evaluation if seokar fails
            return {
                "evaluator": "SeoEvaluator",
                "score": 0.5,
                "passed": False,
                "summary": f"SEO analysis failed: {str(e)}",
                "findings": [
                    {"component": "SEO Analysis", "message": f"SEO analysis failed: {str(e)}", "severity": "error"}
                ],
            }
