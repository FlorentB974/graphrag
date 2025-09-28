#!/usr/bin/env python3
"""
Script to create similarity relationships for existing chunks in the Neo4j database.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from core.graph_db import graph_db

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_similarities(threshold: float = None, doc_id: str = None, mode: str = "chunk"):  # type: ignore
    """Create similarity relationships between chunks and/or entities."""
    try:
        if doc_id:
            # Process specific document
            logger.info(
                f"Creating {mode} similarity relationships for document: {doc_id}"
            )

            if mode == "chunk":
                relationships_created = graph_db.create_chunk_similarities(
                    doc_id, threshold
                )
                print(
                    f"✅ Created {relationships_created} chunk similarity relationships for document {doc_id}"
                )
            elif mode == "entity":
                relationships_created = graph_db.create_entity_similarities(
                    doc_id, threshold
                )
                print(
                    f"✅ Created {relationships_created} entity similarity relationships for document {doc_id}"
                )
            elif mode == "hybrid":
                chunk_rels = graph_db.create_chunk_similarities(doc_id, threshold)
                entity_rels = graph_db.create_entity_similarities(doc_id, threshold)
                relationships_created = chunk_rels + entity_rels
                print(
                    f"✅ Created {chunk_rels} chunk + {entity_rels} entity similarity relationships for document {doc_id}"
                )
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be 'chunk', 'entity', or 'hybrid'"
                )

        else:
            # Process all documents
            logger.info(f"Creating {mode} similarity relationships for all documents")

            if mode == "chunk":
                results = graph_db.create_all_chunk_similarities(threshold)
                _print_chunk_results(results)
            elif mode == "entity":
                results = graph_db.create_all_entity_similarities(threshold)
                _print_entity_results(results)
            elif mode == "hybrid":
                chunk_results = graph_db.create_all_chunk_similarities(threshold)
                entity_results = graph_db.create_all_entity_similarities(threshold)
                _print_hybrid_results(chunk_results, entity_results)
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be 'chunk', 'entity', or 'hybrid'"
                )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to create {mode} similarity relationships: {e}")
        return False


def _print_chunk_results(results):
    """Print results for chunk similarity creation."""
    total_relationships = sum(results.values())
    successful_docs = len([v for v in results.values() if v > 0])

    print("\n📊 Chunk Similarity Relationship Creation Results:")
    print("=" * 60)
    print(f"📄 Documents processed: {len(results)}")
    print(f"✅ Documents with relationships: {successful_docs}")
    print(f"🔗 Total chunk relationships created: {total_relationships}")
    print("=" * 60)

    # Show per-document breakdown
    for doc_id, count in results.items():
        status = "✅" if count > 0 else "⚠️"
        print(f"{status} {doc_id}: {count} chunk relationships")


def _print_entity_results(results):
    """Print results for entity similarity creation."""
    total_relationships = sum(results.values())
    successful_docs = len([v for v in results.values() if v > 0])

    print("\n� Entity Similarity Relationship Creation Results:")
    print("=" * 60)
    print(f"�📄 Documents processed: {len(results)}")
    print(f"✅ Documents with entity relationships: {successful_docs}")
    print(f"🔗 Total entity relationships created: {total_relationships}")
    print("=" * 60)

    # Show per-document breakdown
    for doc_id, count in results.items():
        status = "✅" if count > 0 else "⚠️"
        print(f"{status} {doc_id}: {count} entity relationships")


def _print_hybrid_results(chunk_results, entity_results):
    """Print results for hybrid similarity creation."""
    total_chunk_rels = sum(chunk_results.values())
    total_entity_rels = sum(entity_results.values())

    # Combine results for overall statistics
    all_docs = set(chunk_results.keys()) | set(entity_results.keys())
    successful_chunk_docs = len([v for v in chunk_results.values() if v > 0])
    successful_entity_docs = len([v for v in entity_results.values() if v > 0])

    print("\n📊 Hybrid Similarity Relationship Creation Results:")
    print("=" * 65)
    print(f"📄 Documents processed: {len(all_docs)}")
    print(f"🧩 Documents with chunk relationships: {successful_chunk_docs}")
    print(f"🔗 Documents with entity relationships: {successful_entity_docs}")
    print(f"📊 Total chunk relationships: {total_chunk_rels}")
    print(f"🎯 Total entity relationships: {total_entity_rels}")
    print(f"🔗 Combined total relationships: {total_chunk_rels + total_entity_rels}")
    print("=" * 65)

    # Show per-document breakdown
    for doc_id in sorted(all_docs):
        chunk_count = chunk_results.get(doc_id, 0)
        entity_count = entity_results.get(doc_id, 0)

        if chunk_count > 0 or entity_count > 0:
            status = "✅"
        else:
            status = "⚠️"

        print(
            f"{status} {doc_id}: {chunk_count} chunk + {entity_count} entity relationships"
        )


def show_stats():
    """Display current database statistics."""
    try:
        stats = graph_db.get_graph_stats()

        print("\n📊 Current Database Statistics:")
        print("=" * 50)
        print(f"📄 Documents: {stats.get('documents', 0)}")
        print(f"🧩 Chunks: {stats.get('chunks', 0)}")
        print(f"🎯 Entities: {stats.get('entities', 0)}")
        print("=" * 50)
        print("� Relationships:")
        print(f"  �🔗 Document-Chunk: {stats.get('has_chunk_relations', 0)}")
        print(f"  📊 Chunk-Entity: {stats.get('chunk_entity_relations', 0)}")
        print(f"  🔄 Chunk Similarities: {stats.get('similarity_relations', 0)}")
        print(f"  ↔️ Entity Relations: {stats.get('entity_relations', 0)}")
        print("=" * 50)

        return True
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create similarity relationships between document chunks and/or entities"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Similarity threshold (default: {settings.similarity_threshold})",
    )
    parser.add_argument(
        "--doc-id", type=str, default=None, help="Process specific document ID only"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chunk", "entity", "hybrid"],
        default="chunk",
        help="Similarity computation mode: chunk (traditional), entity (entity-based), or hybrid (both)",
    )
    parser.add_argument(
        "--update-embeddings",
        action="store_true",
        help="Update existing entities with missing embeddings before creating similarities",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show current database statistics"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    if args.stats:
        success = show_stats()
        sys.exit(0 if success else 1)

    threshold = args.threshold or settings.similarity_threshold

    print("🚀 Starting similarity relationship creation")
    print("📊 Configuration:")
    print(f"   - Similarity threshold: {threshold}")
    print(f"   - Max connections per node: {settings.max_similarity_connections}")
    print(f"   - Processing mode: {args.mode}")

    if args.doc_id:
        print(f"   - Target document: {args.doc_id}")
    else:
        print("   - Target: All documents")

    if args.dry_run:
        print("   - Mode: DRY RUN (no changes will be made)")
        mode_desc = {
            "chunk": "chunk-to-chunk similarity relationships",
            "entity": "entity-to-entity similarity relationships",
            "hybrid": "both chunk and entity similarity relationships",
        }
        print(
            f"\n⚠️  This would create {mode_desc[args.mode]} based on the above configuration."
        )
        return

    print()

    # Show current stats
    show_stats()

    # Update entities with embeddings if requested
    if args.update_embeddings:
        print("\n🔄 Updating entities with missing embeddings...")
        updated_count = graph_db.update_entities_with_embeddings()
        print(f"✅ Updated {updated_count} entities with embeddings")
        print()
        # Show updated stats
        show_stats()

    # Create similarities
    success = create_similarities(threshold, args.doc_id, args.mode)

    print()

    # Show final stats
    show_stats()

    if success:
        print(
            f"\n🎉 {args.mode.title()} similarity relationship creation completed successfully!"
        )
    else:
        print(f"\n❌ {args.mode.title()} similarity relationship creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
