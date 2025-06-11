#!/usr/bin/env python3
"""
AI Article Reviewer for Sigmoid NGO Article Repository
Analyzes articles for writing quality, structure, technical accuracy, and information veracity.
"""

import os
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess

# External libraries
try:
    from openai import OpenAI
    from github import Github
    import markdown
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


@dataclass
class ReviewCriteria:
    """Defines the criteria for article review."""

    writing_quality: str = "Clarity, readability, grammar, and style"
    technical_accuracy: str = "Correctness of technical information and code examples"
    structure: str = "Logical flow, organization, and proper use of headings"
    information_veracity: str = (
        "Accuracy of facts and references to latest technologies"
    )
    completeness: str = "Coverage of topic and practical examples"


class ArticleReviewer:
    """Main class for AI-powered article review."""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.github_client = Github(os.environ.get("GITHUB_TOKEN"))
        self.pr_number = int(os.environ.get("PR_NUMBER"))
        self.repository_name = os.environ.get("REPOSITORY")
        self.repo = self.github_client.get_repo(self.repository_name)
        self.criteria = ReviewCriteria()

    def get_changed_files(self) -> List[str]:
        """Get list of changed files in the PR."""
        pr = self.repo.get_pull(self.pr_number)
        changed_files = []

        for file in pr.get_files():
            if file.filename.startswith("article-") and file.filename.endswith(".md"):
                changed_files.append(file.filename)

        return changed_files

    def extract_article_content(self, file_path: str) -> Dict[str, Any]:
        """Extract and analyze article content."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Convert markdown to HTML for better analysis
            html = markdown.markdown(content, extensions=["codehilite", "fenced_code"])
            soup = BeautifulSoup(html, "html.parser")

            # Extract different components
            headers = [
                h.get_text().strip()
                for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            ]
            code_blocks = [code.get_text() for code in soup.find_all("code")]
            text_content = soup.get_text()

            # Calculate basic metrics
            word_count = len(text_content.split())
            code_count = len(code_blocks)
            header_count = len(headers)

            return {
                "raw_content": content,
                "text_content": text_content,
                "headers": headers,
                "code_blocks": code_blocks,
                "word_count": word_count,
                "code_count": code_count,
                "header_count": header_count,
                "file_path": file_path,
            }

        except Exception as e:
            print(f"Error extracting content from {file_path}: {e}")
            return None

    def analyze_with_ai(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI to analyze the article content."""

        # Create a comprehensive prompt for article analysis
        system_prompt = f"""
        You are an expert technical writer and AI researcher reviewing articles for a reputable AI organization. 
        
        Your task is to thoroughly review the submitted article based on these criteria:
        1. Writing Quality: {self.criteria.writing_quality}
        2. Technical Accuracy: {self.criteria.technical_accuracy}
        3. Structure: {self.criteria.structure}
        4. Information Veracity: {self.criteria.information_veracity}
        5. Completeness: {self.criteria.completeness}
        
        Please provide:
        - A score from 1-10 for each criterion
        - Detailed feedback for each criterion
        - An overall score (1-10)
        - Specific suggestions for improvement
        - Notes on technical accuracy and current technology alignment
        
        Be constructive but thorough in your feedback. Focus on AI/ML technical accuracy, 
        code quality, and adherence to current best practices in the field.
        """

        user_prompt = f"""
        Please review this article:
        
        **Article Path:** {article_data['file_path']}
        **Word Count:** {article_data['word_count']}
        **Code Blocks Count:** {article_data['code_count']}
        **Headers:** {', '.join(article_data['headers'])}
        
        **Content:**
        {article_data['raw_content'][:8000]}  # Limit content to avoid token limits
        
        Please provide your review in the following JSON format:
        {{
            "overall_score": <number 1-10>,
            "detailed_feedback": {{
                "writing_quality": {{
                    "score": <number 1-10>,
                    "feedback": "<detailed feedback>"
                }},
                "technical_accuracy": {{
                    "score": <number 1-10>,
                    "feedback": "<detailed feedback>"
                }},
                "structure": {{
                    "score": <number 1-10>,
                    "feedback": "<detailed feedback>"
                }},
                "information_veracity": {{
                    "score": <number 1-10>,
                    "feedback": "<detailed feedback>"
                }},
                "completeness": {{
                    "score": <number 1-10>,
                    "feedback": "<detailed feedback>"
                }}
            }},
            "suggestions": [
                "<specific suggestion 1>",
                "<specific suggestion 2>",
                "<specific suggestion 3>"
            ],
            "technical_accuracy_notes": "<notes about technical accuracy and current technology alignment>",
            "code_quality_issues": [
                "<code issue 1>",
                "<code issue 2>"
            ]
        }}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            # Parse the JSON response
            review_text = response.choices[0].message.content

            # Extract JSON from the response
            json_match = re.search(r"\{.*\}", review_text, re.DOTALL)
            if json_match:
                review_data = json.loads(json_match.group())
                return review_data
            else:
                # Fallback if JSON parsing fails
                return {
                    "overall_score": 5,
                    "detailed_feedback": {
                        "writing_quality": {
                            "score": 5,
                            "feedback": "Could not parse detailed feedback",
                        },
                        "technical_accuracy": {
                            "score": 5,
                            "feedback": "Could not parse detailed feedback",
                        },
                        "structure": {
                            "score": 5,
                            "feedback": "Could not parse detailed feedback",
                        },
                        "information_veracity": {
                            "score": 5,
                            "feedback": "Could not parse detailed feedback",
                        },
                        "completeness": {
                            "score": 5,
                            "feedback": "Could not parse detailed feedback",
                        },
                    },
                    "suggestions": ["Review formatting and structure"],
                    "technical_accuracy_notes": review_text,
                    "code_quality_issues": [],
                }

        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return {
                "overall_score": 0,
                "detailed_feedback": {
                    "writing_quality": {
                        "score": 0,
                        "feedback": f"Error in analysis: {e}",
                    },
                    "technical_accuracy": {
                        "score": 0,
                        "feedback": f"Error in analysis: {e}",
                    },
                    "structure": {"score": 0, "feedback": f"Error in analysis: {e}"},
                    "information_veracity": {
                        "score": 0,
                        "feedback": f"Error in analysis: {e}",
                    },
                    "completeness": {"score": 0, "feedback": f"Error in analysis: {e}"},
                },
                "suggestions": ["Please check the article format"],
                "technical_accuracy_notes": f"Analysis failed: {e}",
                "code_quality_issues": [],
            }

    def check_requirements_compliance(self, file_path: str) -> Dict[str, Any]:
        """Check if article follows the repository guidelines."""
        compliance_issues = []

        # Check if it's in a proper article directory
        if not file_path.startswith("article-"):
            compliance_issues.append(
                "Article should be in a directory named 'article-<topic>'"
            )

        # Check for README.md in the same directory
        article_dir = os.path.dirname(file_path)
        readme_path = os.path.join(article_dir, "README.md")
        if not os.path.exists(readme_path):
            compliance_issues.append("Missing README.md file in article directory")

        # Check for src directory if code is included
        src_path = os.path.join(article_dir, "src")
        if os.path.exists(src_path):
            requirements_path = os.path.join(article_dir, "requirements.txt")
            if not os.path.exists(requirements_path):
                compliance_issues.append(
                    "Missing requirements.txt file for code examples"
                )

        return {"compliant": len(compliance_issues) == 0, "issues": compliance_issues}

    def run_review(self) -> None:
        """Main method to run the article review process."""
        print("🤖 Starting AI Article Review...")

        # Get changed files
        changed_files = self.get_changed_files()

        if not changed_files:
            print("No article files found in this PR.")
            return

        print(f"Found {len(changed_files)} article file(s) to review:")
        for file in changed_files:
            print(f"  - {file}")

        all_reviews = {}

        for file_path in changed_files:
            print(f"\n📖 Reviewing: {file_path}")

            # Extract article content
            article_data = self.extract_article_content(file_path)
            if not article_data:
                continue

            # Check requirements compliance
            compliance = self.check_requirements_compliance(file_path)

            # Analyze with AI
            ai_review = self.analyze_with_ai(article_data)

            # Combine results
            review_result = {
                **ai_review,
                "file_path": file_path,
                "article_metrics": {
                    "word_count": article_data["word_count"],
                    "code_count": article_data["code_count"],
                    "header_count": article_data["header_count"],
                },
                "compliance": compliance,
            }

            all_reviews[file_path] = review_result

        # If multiple files, create a summary
        if len(all_reviews) == 1:
            final_review = list(all_reviews.values())[0]
        else:
            final_review = self.create_multi_file_summary(all_reviews)

        # Add metadata for better notifications
        final_review["review_metadata"] = {
            "reviewed_files": len(all_reviews),
            "total_files": len(changed_files),
            "review_timestamp": subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
            ).stdout.strip(),
            "pr_number": self.pr_number,
            "repository": self.repository_name,
            "reviewer_version": "1.0.0",
        }

        # Save results for GitHub Actions to use
        with open("review_results.json", "w") as f:
            json.dump(final_review, f, indent=2)

        print("✅ Review completed successfully!")
        print(f"📊 Final Score: {final_review['overall_score']}/10")
        if len(all_reviews) > 1:
            print(f"📄 Reviewed {len(all_reviews)} files in this PR")

    def create_multi_file_summary(self, reviews: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary review for multiple files."""
        total_score = sum(review["overall_score"] for review in reviews.values())
        avg_score = total_score / len(reviews)

        all_suggestions = []
        all_issues = []

        for file_path, review in reviews.items():
            all_suggestions.extend(
                [f"**{file_path}**: {s}" for s in review.get("suggestions", [])]
            )
            if not review.get("compliance", {}).get("compliant", True):
                all_issues.extend(
                    [
                        f"**{file_path}**: {issue}"
                        for issue in review["compliance"]["issues"]
                    ]
                )

        return {
            "overall_score": round(avg_score, 1),
            "detailed_feedback": {
                "summary": {
                    "score": round(avg_score, 1),
                    "feedback": f"Reviewed {len(reviews)} files. Individual scores: "
                    + ", ".join(
                        [
                            f"{Path(fp).name}: {r['overall_score']}/10"
                            for fp, r in reviews.items()
                        ]
                    ),
                }
            },
            "suggestions": all_suggestions[:10],  # Limit suggestions
            "technical_accuracy_notes": f"Multi-file review completed for {len(reviews)} articles.",
            "compliance_issues": all_issues,
            "individual_reviews": reviews,
        }


def main():
    """Main entry point."""
    try:
        reviewer = ArticleReviewer()
        reviewer.run_review()
    except Exception as e:
        print(f"❌ Review failed: {e}")
        # Create a minimal error response
        error_response = {
            "overall_score": 0,
            "detailed_feedback": {
                "error": {
                    "score": 0,
                    "feedback": f"Review failed due to error: {str(e)}",
                }
            },
            "suggestions": ["Please check the article format and try again"],
            "technical_accuracy_notes": f"Review process failed: {str(e)}",
        }

        with open("review_results.json", "w") as f:
            json.dump(error_response, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
