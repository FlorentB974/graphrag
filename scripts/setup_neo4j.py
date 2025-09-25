#!/usr/bin/env python3
"""
Setup script for Neo4j database initialization.
"""
import argparse
import logging
import sys
from pathlib import Path
# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from core.graph_db import graph_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_connection():
    """Test Neo4j database connection."""
    try:
        logger.info("Testing Neo4j connection...")
        stats = graph_db.get_graph_stats()
        logger.info("✅ Connection successful!")
        logger.info(f"Current stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False


def setup_indexes():
    """Create necessary database indexes."""
    try:
        logger.info("Setting up database indexes...")
        graph_db.setup_indexes()
        logger.info("✅ Indexes created successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Index creation failed: {e}")
        return False


def clear_database():
    """Clear all data from the database (use with caution!)."""
    try:
        logger.warning("⚠️  CLEARING ALL DATABASE DATA...")
        with graph_db.driver.session() as session:  # type: ignore
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")

        logger.info("✅ Database cleared!")
        return True
    except Exception as e:
        logger.error(f"❌ Database clear failed: {e}")
        return False


def show_stats():
    """Display database statistics."""
    try:
        stats = graph_db.get_graph_stats()

        print("\n📊 Neo4j Database Statistics:")
        print("=" * 40)
        print(f"📄 Documents: {stats.get('documents', 0)}")
        print(f"🧩 Chunks: {stats.get('chunks', 0)}")
        print(f"🔗 Document-Chunk relations: {stats.get('has_chunk_relations', 0)}")
        print(f"🔄 Similarity relations: {stats.get('similarity_relations', 0)}")
        print("=" * 40)

        return True
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        return False


def find_and_fix_bad_embeddings(session, apply: bool = False):
    """Scan chunks for missing/empty/invalid embeddings and optionally recompute them.

    Notes:
    - This routine uses hardcoded behavior: it inspects up to 10000 chunks and shows 3 previews.
    - If `apply` is False the function performs a dry-run and only prints what would be changed.
    """
    from core.embeddings import embedding_manager

    # Hardcoded parameters per request
    params = {"limit": 10000}
    query = (
        "MATCH (c:Chunk) RETURN c.id as id, c.embedding as embedding, c.content as content,"
        " c.filename as filename ORDER BY c.id LIMIT $limit"
    )

    result = session.run(query, **params)

    total = 0
    missing = []
    empty = []
    bad_type = []
    samples = []
    updates = 0

    # preview count hardcoded
    preview = 3
    for record in result:
        total += 1
        cid = record["id"]
        emb = record["embedding"]
        content = record.get("content") or ""

        is_bad = False
        if emb is None:
            missing.append(cid)
            is_bad = True
        elif isinstance(emb, list):
            if len(emb) == 0:
                empty.append(cid)
                is_bad = True
        else:
            bad_type.append((cid, type(emb).__name__))
            is_bad = True

        if is_bad and len(samples) < preview:
            samples.append((cid, content[:120]))

        if is_bad:
            # Recompute embedding
            try:
                new_emb = embedding_manager.get_embedding(content)
            except Exception as e:
                logger.error(f"Failed computing embedding for chunk {cid}: {e}")
                continue

            if not apply:
                print(f"DRY-RUN: chunk {cid} -> new embedding length {len(new_emb)}")
            else:
                try:
                    session.run("MATCH (c:Chunk {id: $cid}) SET c.embedding = $emb", cid=cid, emb=new_emb)
                    updates += 1
                    logger.info(f"Updated embedding for chunk {cid} (len={len(new_emb)})")
                except Exception as e:
                    logger.error(f"Failed to update chunk {cid}: {e}")

    print(f"Inspected: {total}")
    print(f"Missing embeddings: {len(missing)}")
    print(f"Empty embeddings: {len(empty)}")
    if bad_type:
        print(f"Embeddings with unexpected type: {len(bad_type)}")
        for cid, tname in bad_type[:10]:
            print(f" - {cid}: {tname}")

    if samples:
        print("Sample problematic chunks:")
        for cid, preview_text in samples:
            print(f" - {cid}: preview={preview_text!r}")

    print(f"Updates applied: {updates} (apply={apply})")
    return {
        "inspected": total,
        "missing": missing,
        "empty": empty,
        "bad_type": bad_type,
        "updates": updates,
    }


def main():
    """Main entry point for the Neo4j setup script."""
    parser = argparse.ArgumentParser(
        description="Setup and manage Neo4j database for GraphRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test database connection
  python scripts/setup_neo4j.py --test
  
  # Setup indexes
  python scripts/setup_neo4j.py --setup
  
  # Show database statistics
  python scripts/setup_neo4j.py --stats
  
  # Clear all data (DANGEROUS!)
  python scripts/setup_neo4j.py --clear
        """,
    )

    parser.add_argument(
        "--test", "-t", action="store_true", help="Test database connection"
    )

    parser.add_argument(
        "--setup", "-s", action="store_true", help="Setup database indexes"
    )

    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    parser.add_argument(
        "--clear", action="store_true", help="Clear all data from database (DANGEROUS!)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Embeddings fix flag (other parameters are hardcoded)
    parser.add_argument(
        "--fix-embeddings",
        action="store_true",
        help="Scan for bad embeddings and optionally recompute them (dry-run by default)",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates to the DB when fixing embeddings (otherwise dry-run)",
    )

    args = parser.parse_args()

    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If no action specified, run all setup tasks
    if not any([args.test, args.setup, args.stats, args.clear]):
        logger.info("No specific action requested, running full setup...")
        args.test = True
        args.setup = True
        args.stats = True

    success = True

    try:
        if args.test:
            success &= test_connection()

        # Handle embeddings fix option (uses hardcoded parameters)
        if args.fix_embeddings:
            from core.graph_db import graph_db
            with graph_db.driver.session() as session:  # type: ignore
                find_and_fix_bad_embeddings(session, apply=args.apply)

        if args.clear:
            # Ask for confirmation before clearing
            response = input(
                "⚠️  Are you sure you want to clear ALL database data? (yes/no): "
            )
            if response.lower() == "yes":
                success &= clear_database()
            else:
                logger.info("Database clear cancelled")

        if args.setup:
            success &= setup_indexes()

        if args.stats:
            success &= show_stats()

        if success:
            logger.info("✅ All operations completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Some operations failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Ensure database connection is closed
        try:
            from core.graph_db import graph_db
            graph_db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
