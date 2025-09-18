#!/usr/bin/env python3
"""
Script to create similarity relationships for existing chunks in the Neo4j database.
"""
import sys
import logging
import argparse
from pathlib import Path
from core.graph_db import graph_db
from config.settings import settings

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_similarities(threshold: float = None, doc_id: str = None):  # type: ignore
    """Create similarity relationships between chunks."""
    try:
        if doc_id:
            # Process specific document
            logger.info(f"Creating similarity relationships for document: {doc_id}")
            relationships_created = graph_db.create_chunk_similarities(doc_id, threshold)
            print(f"✅ Created {relationships_created} similarity relationships for document {doc_id}")
        else:
            # Process all documents
            logger.info("Creating similarity relationships for all documents")
            results = graph_db.create_all_chunk_similarities(threshold)
            
            total_relationships = sum(results.values())
            successful_docs = len([v for v in results.values() if v > 0])
            
            print("\n📊 Similarity Relationship Creation Results:")
            print("=" * 50)
            print(f"📄 Documents processed: {len(results)}")
            print(f"✅ Documents with relationships: {successful_docs}")
            print(f"🔗 Total relationships created: {total_relationships}")
            print("=" * 50)
            
            # Show per-document breakdown
            for doc_id, count in results.items():
                status = "✅" if count > 0 else "⚠️"
                print(f"{status} {doc_id}: {count} relationships")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create similarity relationships: {e}")
        return False


def show_stats():
    """Display current database statistics."""
    try:
        stats = graph_db.get_graph_stats()
        
        print("\n📊 Current Database Statistics:")
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
    """Main function."""
    parser = argparse.ArgumentParser(description='Create similarity relationships between document chunks')
    parser.add_argument('--threshold', type=float, default=None,
                        help=f'Similarity threshold (default: {settings.similarity_threshold})')
    parser.add_argument('--doc-id', type=str, default=None,
                        help='Process specific document ID only')
    parser.add_argument('--stats', action='store_true',
                        help='Show current database statistics')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    if args.stats:
        success = show_stats()
        sys.exit(0 if success else 1)
    
    threshold = args.threshold or settings.similarity_threshold
    
    print(f"🚀 Starting similarity relationship creation")
    print(f"📊 Configuration:")
    print(f"   - Similarity threshold: {threshold}")
    print(f"   - Max connections per chunk: {settings.max_similarity_connections}")
    
    if args.doc_id:
        print(f"   - Target document: {args.doc_id}")
    else:
        print(f"   - Target: All documents")
    
    if args.dry_run:
        print("   - Mode: DRY RUN (no changes will be made)")
        print("\n⚠️  This would create similarity relationships based on the above configuration.")
        return
    
    print()
    
    # Show current stats
    show_stats()
    
    # Create similarities
    success = create_similarities(threshold, args.doc_id)
    
    print()
    
    # Show final stats
    show_stats()
    
    if success:
        print("\n🎉 Similarity relationship creation completed successfully!")
    else:
        print("\n❌ Similarity relationship creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()