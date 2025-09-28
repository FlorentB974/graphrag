"""
CSV document loader with intelligent processing.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CSVLoader:
    """Loads and intelligently processes content from CSV files."""

    def __init__(self):
        """Initialize the CSV loader."""
        self.max_rows_preview = 1000  # Limit for large CSV files
        self.max_columns_summary = 20  # Limit for very wide CSV files

    def _detect_csv_properties(self, file_path: Path) -> Dict[str, Any]:
        """
        Detect CSV properties like delimiter, encoding, and structure.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary with CSV properties
        """
        properties = {
            "delimiter": ",",
            "encoding": "utf-8",
            "has_header": True,
            "quotechar": '"',
        }

        # Try to detect delimiter and encoding
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        delimiters = [",", ";", "\t", "|"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    sample = f.read(8192)  # Read first 8KB for detection

                # Test different delimiters
                for delimiter in delimiters:
                    if sample.count(delimiter) > 0:
                        properties["delimiter"] = delimiter
                        properties["encoding"] = encoding
                        return properties

            except UnicodeDecodeError:
                continue

        return properties

    def _analyze_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        """
        Analyze DataFrame and create intelligent summary.

        Args:
            df: Pandas DataFrame
            filename: Original filename

        Returns:
            Formatted analysis text
        """
        analysis = []

        # Basic info
        analysis.append(f"=== CSV File Analysis: {filename} ===")
        analysis.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        analysis.append("")

        # Column information
        analysis.append("=== Column Structure ===")
        for i, col in enumerate(df.columns):
            if i >= self.max_columns_summary:
                analysis.append(f"... and {len(df.columns) - i} more columns")
                break

            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()

            # Sample values (first few non-null)
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(v) for v in sample_values)
            if len(sample_str) > 100:
                sample_str = sample_str[:100] + "..."

            analysis.append(f"Column '{col}':")
            analysis.append(f"  - Type: {dtype}")
            analysis.append(
                f"  - Non-null values: {non_null}/{df.shape[0]} ({non_null/df.shape[0]*100:.1f}%)"
            )
            if null_count > 0:
                analysis.append(f"  - Missing values: {null_count}")
            analysis.append(f"  - Sample values: {sample_str}")
            analysis.append("")

        # Data insights
        analysis.append("=== Data Insights ===")

        # Identify potential key columns
        key_candidates = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95 and df[col].dtype in ["object", "int64"]:
                key_candidates.append(col)

        if key_candidates:
            analysis.append(f"Potential key columns: {', '.join(key_candidates)}")

        # Identify numeric columns with statistics
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            analysis.append(
                f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}"
            )
            for col in numeric_cols[:5]:  # Show stats for first 5 numeric columns
                try:
                    stats = df[col].describe()
                    analysis.append(
                        f"  {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}"
                    )
                except:
                    pass

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            analysis.append(
                f"Text/categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}"
            )
            for col in categorical_cols[:5]:  # Show unique counts for first 5
                unique_count = df[col].nunique()
                if unique_count < 100:  # Only show if reasonable number
                    top_values = df[col].value_counts().head(3)
                    top_str = ", ".join(f"'{k}' ({v})" for k, v in top_values.items())
                    analysis.append(
                        f"  {col}: {unique_count} unique values. Top: {top_str}"
                    )

        return "\n".join(analysis)

    def _create_sample_data(self, df: pd.DataFrame) -> str:
        """
        Create a formatted sample of the data.

        Args:
            df: Pandas DataFrame

        Returns:
            Formatted sample data
        """
        sample_lines = []
        sample_lines.append("\n=== Sample Data ===")

        # Show first few rows
        sample_rows = min(10, len(df))
        sample_df = df.head(sample_rows)

        # Convert to string representation, handling large values
        sample_lines.append("First {} rows:".format(sample_rows))

        # Create a readable table format
        for idx, row in sample_df.iterrows():
            row_data = []
            for col in sample_df.columns:
                value = str(row[col])
                if len(value) > 50:
                    value = value[:47] + "..."
                row_data.append(f"{col}: {value}")
            sample_lines.append(f"Row {idx + 1}: {' | '.join(row_data)}")

        if len(df) > sample_rows:
            sample_lines.append(f"... and {len(df) - sample_rows} more rows")

        return "\n".join(sample_lines)

    def load(self, file_path: Path) -> Optional[str]:
        """
        Load and intelligently process content from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Processed text content or None if failed
        """
        try:
            # Detect CSV properties
            properties = self._detect_csv_properties(file_path)

            # Load CSV with detected properties
            df = pd.read_csv(
                file_path,
                delimiter=properties["delimiter"],
                encoding=properties["encoding"],
                quotechar=properties["quotechar"],
                na_values=["", "NULL", "null", "NA", "n/a", "#N/A"],
                keep_default_na=True,
                nrows=self.max_rows_preview,  # Limit for very large files
            )

            if df.empty:
                logger.warning(f"Empty CSV file: {file_path}")
                return None

            # Clean column names (remove extra whitespace, standardize)
            df.columns = [str(col).strip() for col in df.columns]

            # Create comprehensive content
            content_parts = []

            # 1. File analysis
            analysis = self._analyze_dataframe(df, file_path.name)
            content_parts.append(analysis)

            # 2. Sample data
            sample_data = self._create_sample_data(df)
            content_parts.append(sample_data)

            # 3. Additional insights for business/research context
            content_parts.append(self._generate_business_insights(df))

            full_content = "\n\n".join(content_parts)

            logger.info(
                f"Successfully loaded CSV: {file_path} "
                f"({df.shape[0]} rows, {df.shape[1]} columns)"
            )

            return full_content

        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            return None

    def _generate_business_insights(self, df: pd.DataFrame) -> str:
        """
        Generate business-relevant insights from the data.

        Args:
            df: Pandas DataFrame

        Returns:
            Business insights text
        """
        insights = []
        insights.append("\n=== Business Context & Insights ===")

        # Time series detection
        date_columns = []
        for col in df.columns:
            if any(
                keyword in col.lower()
                for keyword in ["date", "time", "created", "updated", "timestamp"]
            ):
                try:
                    pd.to_datetime(df[col], errors="raise")
                    date_columns.append(col)
                except:
                    pass

        if date_columns:
            insights.append(
                f"Time-based analysis possible with columns: {', '.join(date_columns)}"
            )

        # Relationship detection
        potential_relationships = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ["id", "key", "ref", "code"]):
                unique_ratio = df[col].nunique() / len(df)
                if 0.1 < unique_ratio < 1.0:  # Not too few, not all unique
                    potential_relationships.append(col)

        if potential_relationships:
            insights.append(
                f"Potential relationship/grouping columns: {', '.join(potential_relationships)}"
            )

        # Business metrics detection
        metric_columns = []
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"] and any(
                keyword in col.lower()
                for keyword in [
                    "amount",
                    "price",
                    "cost",
                    "revenue",
                    "count",
                    "total",
                    "sum",
                    "avg",
                    "rate",
                    "percent",
                ]
            ):
                metric_columns.append(col)

        if metric_columns:
            insights.append(
                f"Business metric columns identified: {', '.join(metric_columns)}"
            )

        # Data quality insights
        completeness_issues = []
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 20:  # More than 20% missing
                completeness_issues.append(f"{col} ({missing_pct:.1f}% missing)")

        if completeness_issues:
            insights.append(
                f"Data quality attention needed: {', '.join(completeness_issues)}"
            )

        # Suggest analysis possibilities
        analysis_suggestions = []
        if date_columns and metric_columns:
            analysis_suggestions.append("Time series analysis possible")
        if potential_relationships:
            analysis_suggestions.append("Grouping/segmentation analysis possible")
        if len(metric_columns) > 1:
            analysis_suggestions.append("Correlation analysis between metrics possible")

        if analysis_suggestions:
            insights.append(f"Suggested analyses: {', '.join(analysis_suggestions)}")

        return "\n".join(insights)
