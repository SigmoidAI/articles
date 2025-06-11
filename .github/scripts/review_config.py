"""
Configuration settings for the AI Article Reviewer
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ReviewConfig:
    """Configuration for article review process."""

    # Scoring weights for different criteria (must sum to 1.0)
    SCORING_WEIGHTS = {
        "writing_quality": 0.25,
        "technical_accuracy": 0.30,
        "structure": 0.20,
        "information_veracity": 0.15,
        "completeness": 0.10,
    }

    # Minimum scores to pass review
    MINIMUM_SCORES = {"overall": 6.0, "technical_accuracy": 5.0, "writing_quality": 5.0}

    # OpenAI model settings
    OPENAI_MODEL = "gpt-4o"
    TEMPERATURE = 0.3
    MAX_TOKENS = 2000

    # Content limits
    MAX_CONTENT_LENGTH = 8000  # characters
    MAX_SUGGESTIONS = 10

    # File patterns to review
    ARTICLE_PATTERNS = ["article-*/article.md", "article-*/*.md"]

    # Required files in article directories
    REQUIRED_FILES = {
        "README.md": "Article README with usage instructions",
        "requirements.txt": "Python dependencies (if src/ directory exists)",
    }

    # AI-specific technical areas to focus on
    AI_FOCUS_AREAS = [
        "Machine Learning algorithms and implementations",
        "Deep Learning architectures and frameworks",
        "Computer Vision techniques",
        "Natural Language Processing methods",
        "Data preprocessing and feature engineering",
        "Model evaluation and validation",
        "AI ethics and bias considerations",
        "Current AI/ML best practices and standards",
    ]

    # Notification settings
    NOTIFICATION_CONFIG = {
        "primary_reviewer": "eduard-balamatiuc",
        "tag_on_completion": True,
        "tag_on_errors": True,
        "score_based_messaging": True,
        "include_metadata": True,
    }

    # Code quality checks
    CODE_QUALITY_CRITERIA = [
        "Proper imports and dependencies",
        "Clear variable and function naming",
        "Adequate comments and documentation",
        "Error handling and edge cases",
        "Performance considerations",
        "Reproducibility (random seeds, etc.)",
        "Framework-specific best practices",
    ]


# Advanced prompt templates
EXPERT_SYSTEM_PROMPT = """
You are a senior AI researcher and technical writer with expertise in machine learning, 
deep learning, computer vision, NLP, and AI system design. You are reviewing articles 
for a prestigious AI organization's publication.

Your expertise includes:
- Latest developments in AI/ML (2024 state-of-the-art)
- Popular frameworks: PyTorch, TensorFlow, Hugging Face, OpenAI APIs
- Best practices in ML engineering and MLOps
- AI ethics and responsible AI development
- Technical writing and communication

Focus areas for this review:
{focus_areas}

Evaluation criteria:
{criteria_details}

Be thorough but constructive. Provide specific, actionable feedback that will help 
the author improve their work while maintaining high standards for publication.
"""

DETAILED_REVIEW_PROMPT = """
Analyze this AI/ML article with deep technical expertise:

**Article Information:**
- Path: {file_path}
- Word Count: {word_count}
- Code Blocks: {code_count}
- Structure: {headers}

**Content to Review:**
{content}

**Review Instructions:**
1. Technical Accuracy: Verify algorithms, code, and concepts are correct and current
2. Writing Quality: Assess clarity, flow, grammar, and technical communication
3. Structure: Evaluate organization, logical progression, and formatting
4. Information Veracity: Check facts, references, and alignment with current technology
5. Completeness: Assess coverage, examples, and practical value

**Code Quality Focus:**
{code_criteria}

**AI-Specific Considerations:**
- Are the AI/ML concepts explained accurately?
- Is the code following current best practices?
- Are the frameworks and libraries up-to-date?
- Are there any potential ethical considerations not addressed?
- Is the technical depth appropriate for the intended audience?

Please provide your detailed assessment in the specified JSON format with specific, 
actionable feedback for each criterion.
"""

# Review quality thresholds
QUALITY_THRESHOLDS = {
    "excellent": 8.5,
    "good": 7.0,
    "acceptable": 6.0,
    "needs_improvement": 4.0,
    "poor": 0.0,
}

# Automated checks configuration
AUTOMATED_CHECKS = {
    "min_word_count": 500,
    "max_word_count": 5000,
    "min_headers": 3,
    "required_sections": ["introduction", "implementation", "conclusion"],
    "code_requirements": {
        "min_comments_ratio": 0.1,  # 10% of code lines should be comments
        "max_line_length": 120,
        "required_imports_check": True,
    },
}

# Comment templates for different score ranges
COMMENT_TEMPLATES = {
    "excellent": "🌟 **Outstanding Work!** This article demonstrates excellent technical depth and clarity.",
    "good": "✨ **Great Article!** Well-written with good technical content. Minor improvements suggested.",
    "acceptable": "👍 **Good Foundation** with room for enhancement. Please address the feedback below.",
    "needs_improvement": "📝 **Needs Revision** before publication. Please review and address the concerns.",
    "poor": "🔄 **Major Revision Required** - Please significantly rework based on the detailed feedback.",
}

# Specific AI/ML validation rules
AI_VALIDATION_RULES = {
    "deprecated_libraries": [
        "tensorflow.compat.v1",
        "keras.models",  # Old Keras syntax
        "sklearn.cross_validation",  # Deprecated sklearn module
    ],
    "recommended_patterns": [
        "import torch",
        "from transformers import",
        "import numpy as np",
        "import pandas as pd",
    ],
    "current_frameworks": {
        "PyTorch": ["torch", "torchvision", "transformers"],
        "TensorFlow": ["tensorflow", "keras"],
        "Scikit-learn": ["sklearn"],
        "Hugging Face": ["transformers", "datasets", "accelerate"],
    },
}

# Export configuration
CONFIG = ReviewConfig()
