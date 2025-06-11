#!/usr/bin/env python3
"""
AI Article Reviewer for Sigmoid NGO Article Repository
Analyzes articles for writing quality, structure, technical accuracy, and information veracity.
"""

import os
import json
import sys
import re
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess
from datetime import datetime, timezone

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

# Import configuration
try:
    from review_config import CONFIG, QUALITY_THRESHOLDS
except ImportError:
    print("Warning: Could not import review_config. Using default settings.")
    CONFIG = None
    QUALITY_THRESHOLDS = {
        "excellent": 8.5,
        "good": 7.0,
        "acceptable": 6.0,
        "needs_improvement": 4.0,
        "poor": 0.0,
    }


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
    """Main class for AI-powered article review.

    Updated to fetch file contents directly from GitHub PR via API instead of
    using local filesystem, resolving file access issues in CI/CD environments.
    """

    def __init__(self):
        # Validate required environment variables
        self._validate_environment_variables()

        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.github_client = Github(os.environ.get("GITHUB_TOKEN"))
        self.pr_number = int(os.environ.get("PR_NUMBER"))
        self.repository_name = os.environ.get("REPOSITORY")
        self.repo = self.github_client.get_repo(self.repository_name)
        self.criteria = ReviewCriteria()

    def _validate_environment_variables(self):
        """Validate that all required environment variables are present and valid."""
        errors = []

        # Check OPENAI_API_KEY
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            errors.append("OPENAI_API_KEY environment variable is required but not set")
        elif not openai_key.strip():
            errors.append("OPENAI_API_KEY environment variable is empty")

        # Check GITHUB_TOKEN
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            errors.append("GITHUB_TOKEN environment variable is required but not set")
        elif not github_token.strip():
            errors.append("GITHUB_TOKEN environment variable is empty")

        # Check PR_NUMBER
        pr_number_str = os.environ.get("PR_NUMBER")
        if not pr_number_str:
            errors.append("PR_NUMBER environment variable is required but not set")
        else:
            try:
                pr_number = int(pr_number_str)
                if pr_number <= 0:
                    errors.append("PR_NUMBER must be a positive integer")
            except ValueError:
                errors.append(
                    f"PR_NUMBER must be a valid integer, got: '{pr_number_str}'"
                )

        # Check REPOSITORY
        repository = os.environ.get("REPOSITORY")
        if not repository:
            errors.append("REPOSITORY environment variable is required but not set")
        elif not repository.strip():
            errors.append("REPOSITORY environment variable is empty")
        elif "/" not in repository:
            errors.append(
                f"REPOSITORY must be in format 'owner/repo', got: '{repository}'"
            )

        # If there are validation errors, print them and exit
        if errors:
            print("❌ Environment variable validation failed:")
            for error in errors:
                print(f"  - {error}")
            print("\nRequired environment variables:")
            print("  - OPENAI_API_KEY: Your OpenAI API key")
            print("  - GITHUB_TOKEN: GitHub personal access token or workflow token")
            print("  - PR_NUMBER: Pull request number (positive integer)")
            print("  - REPOSITORY: Repository name in format 'owner/repo'")
            sys.exit(1)

    def get_changed_files(self) -> List[str]:
        """Get list of changed files in the PR that could be articles.

        Uses specific criteria to identify article files while avoiding false positives
        from common documentation files like README.md, CHANGELOG.md, etc.
        Also filters out non-text files like images.
        """
        pr = self.repo.get_pull(self.pr_number)
        changed_files = []

        print(
            f"🔍 Analyzing {pr.get_files().totalCount} changed files in PR #{self.pr_number}:"
        )

        for file in pr.get_files():
            filename = file.filename.lower()
            print(f"  📄 Checking: {file.filename}")

            # Skip non-text files (images, videos, etc.)
            if self._is_non_text_file(filename):
                print(f"    ⏭️  Skipped (non-text file): {file.filename}")
                continue

            # Skip files that are too large (likely binary or generated)
            if file.changes > 10000:  # Skip files with excessive changes
                print(f"    ⏭️  Skipped (too many changes): {file.filename}")
                continue

            # Detect article files using multiple criteria to avoid false positives
            # from common documentation files like README.md, CHANGELOG.md, etc.
            is_article_file = (
                # 1. Traditional article naming patterns
                filename.startswith("article-")
                or filename.startswith("article_")
                or
                # 2. Files with 'article' in directory path (e.g., "content/article-ai.md")
                "/article" in filename
                or
                # 3. Markdown files with 'article' in filename (e.g., "my-article.md")
                (filename.endswith(".md") and "article" in filename)
                or
                # 4. Files in directories containing 'article' (e.g., "articles/intro.md")
                any(part for part in filename.split("/") if "article" in part.lower())
                or
                # 5. README files specifically in article directories
                (
                    filename.endswith("readme.md")
                    and any(
                        "article" in part.lower() for part in filename.split("/")[:-1]
                    )
                )
                or
                # 6. Content markdown files in subdirectories (excluding common docs)
                # This replaces the problematic catch-all that was causing false positives
                (
                    filename.endswith(".md")
                    and not filename.startswith(".github/")  # Exclude GitHub configs
                    and not filename.lower().endswith("readme.md")  # Exclude READMEs
                    and not filename.lower().endswith(
                        "changelog.md"
                    )  # Exclude changelogs
                    and not filename.lower().endswith("license.md")  # Exclude licenses
                    and not filename.lower().endswith(
                        "contributing.md"
                    )  # Exclude contrib guides
                    and not any(  # Exclude other common documentation files
                        common_doc in filename.lower()
                        for common_doc in [
                            "license",
                            "changelog",
                            "todo",
                            "authors",
                            "contributors",
                        ]
                    )
                    and len([part for part in filename.split("/") if part])
                    >= 2  # Must be in a subdirectory (not root-level docs)
                )
            )

            if is_article_file:
                changed_files.append(file.filename)
                print(f"    ✅ Added for review: {file.filename}")
            else:
                print(f"    ⏭️  Skipped: {file.filename}")

        print(f"📝 Total files selected for review: {len(changed_files)}")
        return changed_files

    def _is_non_text_file(self, filename: str) -> bool:
        """Check if a file is a non-text file that should not be reviewed."""
        # Common image extensions
        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".svg",
            ".webp",
            ".ico",
            ".tiff",
            ".tif",
            ".raw",
            ".heic",
            ".avif",
        }

        # Common video extensions
        video_extensions = {
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
            ".3gp",
            ".ogv",
            ".f4v",
        }

        # Common audio extensions
        audio_extensions = {
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".ogg",
            ".wma",
            ".m4a",
            ".opus",
        }

        # Common binary/executable extensions
        binary_extensions = {
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".app",
            ".deb",
            ".rpm",
            ".msi",
            ".dmg",
            ".pkg",
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
        }

        # Common document formats (that aren't plain text)
        document_extensions = {
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".odt",
            ".ods",
            ".odp",
        }

        # Common font extensions
        font_extensions = {".ttf", ".otf", ".woff", ".woff2", ".eot"}

        # Get file extension
        file_ext = Path(filename).suffix.lower()

        # Check if it's any type of non-text file
        non_text_extensions = (
            image_extensions
            | video_extensions
            | audio_extensions
            | binary_extensions
            | document_extensions
            | font_extensions
        )

        return file_ext in non_text_extensions

    def get_file_content_from_pr(self, file_path: str) -> Optional[str]:
        """Get file content directly from the PR using GitHub API."""
        try:
            pr = self.repo.get_pull(self.pr_number)

            # Get the head SHA from the PR
            head_sha = pr.head.sha

            # Get the file content from the specific commit
            file_content = self.repo.get_contents(file_path, ref=head_sha)

            # GitHub API always returns base64 encoded content
            decoded_bytes = base64.b64decode(file_content.content)

            # Try UTF-8 decoding first, fallback to latin-1 if it fails
            try:
                content = decoded_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content = decoded_bytes.decode("latin-1")
                    print(f"⚠️  Used latin-1 encoding fallback for {file_path}")
                except UnicodeDecodeError:
                    print(f"❌ Could not decode file {file_path} with UTF-8 or latin-1")
                    return None

            return content

        except Exception as e:
            print(f"❌ Error getting file content from PR for {file_path}: {e}")
            return None

    def extract_article_content(self, file_path: str) -> Dict[str, Any]:
        """Extract and analyze article content from PR."""
        try:
            print(f"📖 Getting content from GitHub API for: {file_path}")

            # Check if it's a text file we can process
            if self._is_non_text_file(file_path.lower()):
                print(f"⚠️  Skipping non-text file: {file_path}")
                return None

            # Get file content directly from the PR
            content = self.get_file_content_from_pr(file_path)
            if not content:
                print(f"❌ Could not get content for: {file_path}")
                return None

            # Skip empty files
            if not content.strip():
                print(f"⚠️  File is empty: {file_path}")
                return None

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
            print(f"❌ Error extracting content from {file_path}: {e}")
            return None

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text using proper bracket counting that accounts for string literals."""
        # Find the first opening brace
        start_idx = text.find("{")
        if start_idx == -1:
            return None

        # Count braces to find the matching closing brace, accounting for string literals
        brace_count = 0
        end_idx = start_idx
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                # Skip this character as it's escaped
                escape_next = False
                continue

            if char == "\\" and in_string:
                # Next character is escaped
                escape_next = True
                continue

            if char == '"' and not escape_next:
                # Toggle string state
                in_string = not in_string
                continue

            # Only count braces when not inside a string literal
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

        if brace_count != 0:
            return None

        # Extract and parse JSON
        json_str = text[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
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

            # Extract JSON from the response using bracket counting
            review_data = self._extract_json_from_text(review_text)
            if review_data:
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

    def check_file_exists_in_pr(self, file_path: str) -> bool:
        """Check if a file exists in the PR using GitHub API."""
        try:
            pr = self.repo.get_pull(self.pr_number)
            head_sha = pr.head.sha
            self.repo.get_contents(file_path, ref=head_sha)
            return True
        except Exception:
            return False

    def check_requirements_compliance(self, file_path: str) -> Dict[str, Any]:
        """Check if article follows the repository guidelines."""
        compliance_issues = []

        # More flexible check for article directory structure
        filename_lower = file_path.lower()
        in_article_directory = (
            filename_lower.startswith("article-")
            or filename_lower.startswith("article_")
            or "/article" in filename_lower
            or any("article" in part.lower() for part in file_path.split("/"))
        )

        if not in_article_directory and file_path.endswith(".md"):
            compliance_issues.append(
                "Article should be in a directory named 'article-<topic>' or 'article_<topic>', "
                "or have 'article' in the directory/filename"
            )

        # Check for README.md in the same directory using GitHub API
        article_dir = os.path.dirname(file_path)
        # Handle root-level files where dirname returns empty string
        if not article_dir:
            article_dir = "."

        # Check for various case variations of README files
        readme_variants = ["README.md", "readme.md", "Readme.md", "ReadMe.md"]
        readme_exists = False

        for variant in readme_variants:
            readme_path = os.path.join(article_dir, variant).replace("\\", "/")
            if self.check_file_exists_in_pr(readme_path):
                readme_exists = True
                break

        # Ensure we're not checking if the file itself is README.md
        file_name = os.path.basename(file_path)
        if file_name.lower() != "readme.md" and not readme_exists:
            # Only warn about missing README for files in subdirectories
            if article_dir != ".":
                compliance_issues.append("Missing README.md file in article directory")

        # Check for src directory and requirements.txt using GitHub API
        src_path = os.path.join(article_dir, "src").replace("\\", "/")
        if self.check_file_exists_in_pr(src_path):
            requirements_path = os.path.join(article_dir, "requirements.txt").replace(
                "\\", "/"
            )
            if not self.check_file_exists_in_pr(requirements_path):
                compliance_issues.append(
                    "Missing requirements.txt file for code examples"
                )

        return {"compliant": len(compliance_issues) == 0, "issues": compliance_issues}

    def run_review(self) -> None:
        """Main method to run the article review process."""
        print("🤖 Starting AI Article Review...")

        # Debug: Print current working directory and environment
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"🔧 Python executable: {sys.executable}")
        print(f"📋 Environment variables:")
        print(f"   - PR_NUMBER: {os.environ.get('PR_NUMBER', 'Not set')}")
        print(f"   - REPOSITORY: {os.environ.get('REPOSITORY', 'Not set')}")
        print(f"   - GITHUB_WORKSPACE: {os.environ.get('GITHUB_WORKSPACE', 'Not set')}")

        # Get changed files
        changed_files = self.get_changed_files()

        if not changed_files:
            print("No article files found in this PR.")
            return

        print(f"Found {len(changed_files)} article file(s) to review:")
        for file in changed_files:
            print(f"  📄 {file}")

        all_reviews = {}

        processed_files = []
        skipped_files = []

        for file_path in changed_files:
            print(f"\n📖 Reviewing: {file_path}")

            # Extract article content
            article_data = self.extract_article_content(file_path)
            if not article_data:
                print(f"⏭️  Skipping {file_path} - could not extract content")
                skipped_files.append(file_path)
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
            processed_files.append(file_path)

        # Report processing results
        print(f"\n📊 Processing Summary:")
        print(f"  ✅ Successfully processed: {len(processed_files)} files")
        if skipped_files:
            print(f"  ⏭️  Skipped: {len(skipped_files)} files")
            for skipped_file in skipped_files:
                print(f"    - {skipped_file}")

        if not all_reviews:
            print("❌ No files could be processed for review.")
            # Create a minimal response indicating no files were processed
            final_review = {
                "overall_score": 0,
                "detailed_feedback": {
                    "no_files": {
                        "score": 0,
                        "feedback": f"No reviewable files found. Checked {len(changed_files)} files, skipped {len(skipped_files)} files.",
                    }
                },
                "suggestions": [
                    "Ensure article files are text-based (e.g., .md files) and exist in the repository"
                ],
                "technical_accuracy_notes": "No files were available for review",
                "review_metadata": {
                    "reviewed_files": 0,
                    "total_files": len(changed_files),
                    "skipped_files": len(skipped_files),
                    "review_timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "pr_number": self.pr_number,
                    "repository": self.repository_name,
                    "reviewer_version": "1.0.0",
                },
            }

            # Save results
            with open("review_results.json", "w") as f:
                json.dump(final_review, f, indent=2)

            print("⚠️  Review completed with no processable files.")
            return

        # If multiple files, create a summary
        if len(all_reviews) == 1:
            final_review = list(all_reviews.values())[0]
        else:
            final_review = self.create_multi_file_summary(all_reviews)

        # Add metadata for better notifications
        final_review["review_metadata"] = {
            "reviewed_files": len(all_reviews),
            "total_files": len(changed_files),
            "review_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pr_number": self.pr_number,
            "repository": self.repository_name,
            "reviewer_version": "1.0.0",
            "primary_reviewer": (
                CONFIG.NOTIFICATION_CONFIG["primary_reviewer"]
                if CONFIG
                else "eduard-balamatiuc"
            ),
            "quality_thresholds": QUALITY_THRESHOLDS,
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
