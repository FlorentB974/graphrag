"""
Document summarization functionality for extracting summaries, document types, and hashtags.
"""

import logging
from typing import Dict, List, Any, Optional

from config.settings import settings
from core.llm import llm_manager

logger = logging.getLogger(__name__)

# Comprehensive document type taxonomy
DOCUMENT_TYPES = [
    "quote",
    "invoice",
    "receipt",
    "purchase_order",
    "contract",
    "agreement",
    "report",
    "financial_report",
    "research_report",
    "business_report",
    "technical_report",
    "resume",
    "cv",
    "cover_letter",
    "insurance_document",
    "insurance_policy",
    "claim_form",
    "medical_record",
    "prescription",
    "legal_document",
    "court_document",
    "deed",
    "will",
    "power_of_attorney",
    "academic_paper",
    "thesis",
    "dissertation",
    "article",
    "blog_post",
    "news_article",
    "press_release",
    "whitepaper",
    "specification",
    "technical_specification",
    "manual",
    "user_manual",
    "guide",
    "tutorial",
    "presentation",
    "slide_deck",
    "proposal",
    "business_proposal",
    "project_proposal",
    "grant_proposal",
    "memo",
    "memorandum",
    "letter",
    "business_letter",
    "email",
    "form",
    "application_form",
    "registration_form",
    "tax_form",
    "financial_statement",
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
    "budget",
    "forecast",
    "plan",
    "business_plan",
    "project_plan",
    "marketing_plan",
    "strategy_document",
    "policy_document",
    "procedure_document",
    "sop",
    "checklist",
    "schedule",
    "calendar",
    "agenda",
    "minutes",
    "meeting_minutes",
    "transcript",
    "interview_transcript",
    "certificate",
    "diploma",
    "license",
    "permit",
    "warranty",
    "guarantee",
    "specification_sheet",
    "datasheet",
    "brochure",
    "catalog",
    "flyer",
    "pamphlet",
    "booklet",
    "book",
    "ebook",
    "chapter",
    "section",
    "reference_document",
    "documentation",
    "api_documentation",
    "code_documentation",
    "readme",
    "changelog",
    "release_notes",
    "announcement",
    "notice",
    "notification",
    "alert",
    "bulletin",
    "newsletter",
    "journal_entry",
    "log",
    "record",
    "note",
    "annotation",
    "comment",
    "review",
    "feedback",
    "survey",
    "questionnaire",
    "assessment",
    "evaluation",
    "test",
    "exam",
    "quiz",
    "worksheet",
    "assignment",
    "homework",
    "syllabus",
    "curriculum",
    "lesson_plan",
    "lecture_notes",
    "study_guide",
    "reference_sheet",
    "cheat_sheet",
    "other"
]


class DocumentSummarizer:
    """Handles document summarization, type classification, and hashtag extraction."""

    def __init__(self):
        """Initialize the document summarizer."""
        pass

    def extract_summary(
        self,
        chunks: List[Dict[str, Any]],
        max_summary_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract a summary, document type, and hashtags from document chunks.

        Args:
            chunks: List of document chunks with 'content' field
            max_summary_length: Maximum length of summary (defaults to chunk_size from settings)

        Returns:
            Dictionary containing:
                - summary: Markdown-formatted summary
                - document_type: Classified document type
                - hashtags: List of relevant hashtags
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for summary extraction")
                return {
                    "summary": "",
                    "document_type": "other",
                    "hashtags": []
                }

            # Combine all chunks into full content
            full_content = "\n\n".join([chunk.get("content", "") for chunk in chunks])

            # Use chunk_size from settings as max summary length
            if max_summary_length is None:
                max_summary_length = settings.chunk_size

            # Create the LLM prompt for summary extraction
            system_message = f"""You are an expert document analyst specialized in creating CONCISE, structured summaries.

Your task:
1. Create a brief, well-structured summary (max {max_summary_length} chars)
2. Classify the document type
3. Generate 5-8 relevant hashtags

Format requirements:
- Use Markdown: **bold** for key info, bullet points for lists
- Be concise: focus on essentials, no fluff
- Structure: 2-3 short paragraphs or bullet points
- Preserve key details: names, dates, amounts, main topics

Document types to choose from:
{', '.join(DOCUMENT_TYPES)}

Return JSON format:
{{
    "summary": "Concise MD summary here",
    "document_type": "selected_type",
    "hashtags": ["#hashtag1", "#hashtag2"]
}}

Remember: Be concise and use Markdown formatting for better readability."""

            # Prepare document content for LLM with smart truncation
            # If document is larger than 15000 chars, truncate but try to include more context
            if len(full_content) > 15000:
                # Take first 12000 chars and try to end at a sentence boundary
                content_to_send = full_content[:12000]
                last_period = content_to_send.rfind('.')
                if last_period > 10000:  # Only truncate at period if it's reasonably far into the text
                    content_to_send = content_to_send[:last_period + 1]
            else:
                content_to_send = full_content

            prompt = f"""Document content to analyze:

{content_to_send}

Provide a concise summary (max {max_summary_length} chars), document type, and hashtags as JSON."""

            # Generate the analysis with controlled token limit
            response = llm_manager.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=800  # Limit response to encourage brevity
            )

            # Parse the JSON response
            import json
            import re

            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse JSON from LLM response, using fallback")
                result = {
                    "summary": response[:max_summary_length],
                    "document_type": "other",
                    "hashtags": []
                }

            # Validate and clean the result
            summary = result.get("summary", "")
            document_type = result.get("document_type", "other").lower().strip()
            hashtags = result.get("hashtags", [])

            # Ensure document type is valid
            if document_type not in DOCUMENT_TYPES:
                logger.warning(f"Invalid document type '{document_type}', using 'other'")
                document_type = "other"

            # Clean and validate hashtags
            cleaned_hashtags = []
            for tag in hashtags:
                if isinstance(tag, str):
                    tag = tag.strip()
                    if not tag.startswith('#'):
                        tag = '#' + tag
                    cleaned_hashtags.append(tag)

            # Limit summary length if needed
            max_len = int(max_summary_length * 1.5)
            min_len = int(max_summary_length * 0.8)
            if len(summary) > max_len:
                summary = summary[:max_len]
                # Try to end at a sentence boundary
                last_period = summary.rfind('.')
                if last_period > min_len:
                    summary = summary[:last_period + 1]

            logger.info(
                f"Extracted summary ({len(summary)} chars), type: {document_type}, "
                f"hashtags: {len(cleaned_hashtags)}"
            )

            return {
                "summary": summary,
                "document_type": document_type,
                "hashtags": cleaned_hashtags
            }

        except Exception as e:
            logger.error(f"Failed to extract document summary: {e}")
            return {
                "summary": "",
                "document_type": "other",
                "hashtags": []
            }


# Global document summarizer instance
document_summarizer = DocumentSummarizer()
