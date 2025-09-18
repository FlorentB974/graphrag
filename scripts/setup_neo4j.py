#!/usr/bin/env python3
"""
Setup script for Neo4j database initialization.
"""
import argparse
import logging
import sys
from pathlib import Path
from core.graph_db import graph_db
from config.settings import settings

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        """
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test database connection"
    )
    
    parser.add_argument(
        "--setup", "-s",
        action="store_true",
        help="Setup database indexes"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all data from database (DANGEROUS!)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
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
        
        if args.clear:
            # Ask for confirmation before clearing
            response = input("⚠️  Are you sure you want to clear ALL database data? (yes/no): ")
            if response.lower() == 'yes':
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
        graph_db.close()


if __name__ == "__main__":
    main()