#!/usr/bin/env python3
"""
CLI script for ingesting documents into the GraphRAG pipeline.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from ingestion.document_processor import document_processor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_single_file(file_path: Path) -> bool:
    """
    Ingest a single file.

    Args:
        file_path: Path to the file to ingest

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing file: {file_path}")
        result = document_processor.process_file(file_path)

        if result and result.get("status") == "success":
            chunks_created = result.get("chunks_created", 0)
            logger.info(
                f"✅ Successfully processed {file_path}: {chunks_created} chunks created"
            )
            return True
        else:
            error = (
                result.get("error", "Unknown error") if result else "Processing failed"
            )
            logger.error(f"❌ Failed to process {file_path}: {error}")
            return False

    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")
        return False


def ingest_directory(directory_path: Path, recursive: bool = False) -> tuple[int, int]:
    """
    Ingest all files in a directory.

    Args:
        directory_path: Path to the directory
        recursive: Whether to process subdirectories

    Returns:
        Tuple of (successful_files, total_files)
    """
    try:
        logger.info(f"Processing directory: {directory_path} (recursive: {recursive})")
        results = document_processor.process_directory(directory_path, recursive)

        successful = sum(1 for result in results if result.get("status") == "success")
        total = len(results)

        logger.info(
            f"✅ Directory processing complete: {successful}/{total} files successful"
        )

        # Log details for failed files
        for result in results:
            if result.get("status") != "success":
                file_path = result.get("file_path", "unknown")
                error = result.get("error", "unknown error")
                logger.warning(f"❌ Failed: {file_path} - {error}")

        return successful, total

    except Exception as e:
        logger.error(f"❌ Error processing directory {directory_path}: {e}")
        return 0, 0


def main():
    """Main entry point for the document ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the GraphRAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single file
  python scripts/ingest_documents.py --file /path/to/document.pdf
  
  # Ingest all files in a directory
  python scripts/ingest_documents.py --input-dir /path/to/documents
  
  # Ingest recursively
  python scripts/ingest_documents.py --input-dir /path/to/documents --recursive
  
  # Show supported file types
  python scripts/ingest_documents.py --show-supported
        """,
    )

    parser.add_argument(
        "--file", "-f", type=Path, help="Path to a single file to ingest"
    )

    parser.add_argument(
        "--input-dir",
        "-d",
        type=Path,
        help="Path to directory containing files to ingest",
    )

    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Process directories recursively"
    )

    parser.add_argument(
        "--show-supported", action="store_true", help="Show supported file extensions"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show supported file types
    if args.show_supported:
        extensions = document_processor.get_supported_extensions()
        print("Supported file extensions:")
        for ext in sorted(extensions):
            print(f"  {ext}")
        return

    # Validate arguments
    if not args.file and not args.input_dir:
        parser.error("Must specify either --file or --input-dir")

    if args.file and args.input_dir:
        parser.error("Cannot specify both --file and --input-dir")

    # Process files
    try:
        if args.file:
            # Single file processing
            if not args.file.exists():
                logger.error(f"File not found: {args.file}")
                sys.exit(1)

            success = ingest_single_file(args.file)
            sys.exit(0 if success else 1)

        elif args.input_dir:
            # Directory processing
            if not args.input_dir.exists():
                logger.error(f"Directory not found: {args.input_dir}")
                sys.exit(1)

            if not args.input_dir.is_dir():
                logger.error(f"Path is not a directory: {args.input_dir}")
                sys.exit(1)

            successful, total = ingest_directory(args.input_dir, args.recursive)

            if total == 0:
                logger.warning("No files found to process")
                sys.exit(1)
            elif successful == 0:
                logger.error("No files were successfully processed")
                sys.exit(1)
            elif successful < total:
                logger.warning(f"Some files failed to process ({successful}/{total})")
                sys.exit(1)
            else:
                logger.info(f"All files processed successfully ({successful}/{total})")
                sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
