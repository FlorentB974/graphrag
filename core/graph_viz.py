"""
Graph visualization utilities for Neo4j data using NetworkX and Plotly.
"""

import logging
from typing import Any, Dict, List

import networkx as nx
import plotly.graph_objects as go

from core.graph_db import graph_db

logger = logging.getLogger(__name__)


def get_graph_data(limit: int = 100, retrieval_mode: str = "chunk_only") -> Dict[str, Any]:
    """
    Retrieve graph data from Neo4j for visualization based on retrieval mode.

    Args:
        limit: Maximum number of nodes to retrieve
        retrieval_mode: "chunk_only", "entity_only", or "hybrid"

    Returns:
        Dictionary containing nodes and edges for visualization
    """
    try:
        with graph_db.driver.session() as session:  # type: ignore
            # Build node query based on retrieval mode
            if retrieval_mode == "chunk_only":
                node_filters = "n:Document OR n:Chunk"
            elif retrieval_mode == "entity_only":
                node_filters = "n:Document OR n:Entity"
            else:  # hybrid mode
                node_filters = "n:Document OR n:Chunk OR n:Entity"
            
            # Get nodes (documents, chunks, and/or entities)
            nodes_query = f"""
            MATCH (n)
            WHERE {node_filters}
            RETURN
                elementId(n) as node_id,
                labels(n) as labels,
                n.id as entity_id,
                CASE
                    WHEN n:Document THEN n.filename
                    WHEN n:Chunk THEN substring(n.content, 0, 50) + "..."
                    WHEN n:Entity THEN n.name
                    ELSE "Unknown"
                END as title,
                CASE
                    WHEN n:Document THEN size(n.content)
                    WHEN n:Chunk THEN size(n.content)
                    WHEN n:Entity THEN coalesce(n.importance_score * 50, 20)
                    ELSE 0
                END as content_size,
                CASE
                    WHEN n:Entity THEN n.type
                    ELSE null
                END as entity_type,
                CASE
                    WHEN n:Entity THEN coalesce(n.importance_score, 0.5)
                    ELSE null
                END as confidence
            LIMIT $limit
            """

            nodes_result = session.run(nodes_query, limit=limit)
            nodes = []
            node_ids = set()

            for record in nodes_result:
                node_data = record.data()
                if node_data["content_size"] is None:
                    node_data["content_size"] = 0
                
                node_info = {
                    "id": str(node_data["node_id"]),
                    "entity_id": node_data["entity_id"],
                    "label": (
                        node_data["labels"][0] if node_data["labels"] else "Unknown"
                    ),
                    "title": node_data["title"] or "Untitled",
                    "size": min(
                        max(int(node_data["content_size"]) / 100, 10), 50
                    ),  # Scale size
                }
                
                # Add entity-specific properties
                if node_data.get("entity_type"):
                    node_info["entity_type"] = node_data["entity_type"]
                    node_info["confidence"] = node_data["confidence"]
                
                nodes.append(node_info)
                node_ids.add(str(node_data["node_id"]))

            # Get relationships between the nodes
            edges_query = f"""
            MATCH (n)-[r]-(m)
            WHERE ({node_filters}) AND ({node_filters.replace('n:', 'm:')})
            AND elementId(n) IN $node_ids AND elementId(m) IN $node_ids
            RETURN
                elementId(n) as source_id,
                elementId(m) as target_id,
                type(r) as relationship_type,
                coalesce(properties(r)['similarity'], properties(r)['strength'], 1.0) as weight
            LIMIT $limit
            """

            edges_result = session.run(
                edges_query, node_ids=list(node_ids), limit=limit
            )
            edges = []

            for record in edges_result:
                edge_data = record.data()
                edges.append(
                    {
                        "source": str(edge_data["source_id"]),
                        "target": str(edge_data["target_id"]),
                        "type": edge_data["relationship_type"],
                        "weight": edge_data["weight"],
                    }
                )

            return {
                "nodes": nodes,
                "edges": edges,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "retrieval_mode": retrieval_mode,
            }

    except Exception as e:
        logger.error(f"Failed to retrieve graph data: {e}")
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0, "retrieval_mode": retrieval_mode}


def create_networkx_graph(graph_data: Dict[str, Any]) -> nx.Graph:
    """
    Create a NetworkX graph from the Neo4j data.

    Args:
        graph_data: Dictionary containing nodes and edges

    Returns:
        NetworkX graph object
    """
    G = nx.Graph()

    # Add nodes
    for node in graph_data["nodes"]:
        G.add_node(
            node["id"],
            label=node["label"],
            title=node["title"],
            entity_id=node["entity_id"],
            size=node["size"],
        )

    # Add edges
    for edge in graph_data["edges"]:
        if edge["source"] in G.nodes and edge["target"] in G.nodes:
            G.add_edge(
                edge["source"], edge["target"], type=edge["type"], weight=edge["weight"]
            )

    return G


def create_plotly_graph(
    graph_data: Dict[str, Any], layout_algorithm: str = "spring"
) -> go.Figure:
    """
    Create an interactive Plotly graph visualization with support for entities.

    Args:
        graph_data: Dictionary containing nodes and edges
        layout_algorithm: Layout algorithm ('spring', 'circular', 'kamada_kawai')

    Returns:
        Plotly figure object
    """
    if not graph_data["nodes"]:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No graph data available",
            showlegend=False,
            xaxis={"showgrid": False, "zeroline": False, "visible": False},
            yaxis={"showgrid": False, "zeroline": False, "visible": False},
        )
        return fig

    # Create NetworkX graph for layout calculation
    G = create_networkx_graph(graph_data)

    # Calculate positions using NetworkX layout algorithms
    if layout_algorithm == "circular":
        pos = nx.circular_layout(G)
    elif layout_algorithm == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:  # default to spring
        pos = nx.spring_layout(G, k=1, iterations=50)

    # Enhanced color map for different node types
    color_map = {
        "Document": "#FF6B6B",  # Red for documents
        "Chunk": "#4ECDC4",     # Teal for chunks
        "Entity": "#9B59B6",    # Purple for entities
        "Unknown": "#95A5A6"    # Gray for unknown
    }
    
    # Shape map for different node types (using symbols)
    symbol_map = {
        "Document": "square",
        "Chunk": "circle",
        "Entity": "diamond",
        "Unknown": "circle"
    }
    
    # Size scaling for better entity visibility
    size_map = {
        "Document": lambda size: max(size * 1.2, 25),  # Larger documents
        "Chunk": lambda size: max(size, 15),          # Normal chunks
        "Entity": lambda size: max(size * 1.5, 20),   # Larger entities for visibility
        "Unknown": lambda size: max(size, 15)
    }

    # Prepare node traces by type for better legend support
    node_types = {}
    for node in graph_data["nodes"]:
        node_type = node["label"]
        if node_type not in node_types:
            node_types[node_type] = {
                "x": [], "y": [], "text": [], "sizes": [], "colors": [],
                "hover_texts": [], "symbols": []
            }

    # Organize nodes by type
    for node in graph_data["nodes"]:
        node_id = node["id"]
        if node_id in pos:
            x, y = pos[node_id]
            node_type = node["label"]
            
            node_types[node_type]["x"].append(x)
            node_types[node_type]["y"].append(y)
            node_types[node_type]["text"].append(node["title"][:15])  # Truncate for display
            
            # Apply size scaling based on node type
            raw_size = max(node["size"], 15)
            scaled_size = size_map.get(node_type, lambda s: s)(raw_size)
            node_types[node_type]["sizes"].append(scaled_size)
            
            node_types[node_type]["colors"].append(color_map.get(node_type, "#95A5A6"))
            node_types[node_type]["symbols"].append(symbol_map.get(node_type, "circle"))
            
            # Create detailed hover text
            hover_text = f"<b>{node_type}:</b> {node['title']}<br>"
            hover_text += f"<b>ID:</b> {node.get('entity_id', 'N/A')}<br>"
            
            if node.get("entity_type"):
                hover_text += f"<b>Type:</b> {node['entity_type']}<br>"
                hover_text += f"<b>Confidence:</b> {node.get('confidence', 0):.2f}<br>"
            
            node_types[node_type]["hover_texts"].append(hover_text)

    # Create separate traces for each node type
    node_traces = []
    for node_type, data in node_types.items():
        if data["x"]:  # Only create trace if there are nodes of this type
            node_trace = go.Scatter(
                x=data["x"],
                y=data["y"],
                text=data["text"],
                textposition="middle center",
                mode="markers+text",
                hovertext=data["hover_texts"],
                hoverinfo="text",
                name=node_type,
                marker=dict(
                    size=data["sizes"],
                    color=color_map.get(node_type, "#95A5A6"),
                    symbol=data["symbols"][0] if data["symbols"] else "circle",
                    line=dict(width=2, color="white")
                ),
                showlegend=True
            )
            node_traces.append(node_trace)

    # Prepare edge traces with enhanced styling
    edge_traces = []
    edge_color_map = {
        "HAS_CHUNK": {"color": "rgba(125,125,125,0.5)", "width": 2, "name": "Document-Chunk"},
        "SIMILAR_TO": {"color": "rgba(255,107,107,0.7)", "width": 1, "dash": "dash", "name": "Similar Content"},
        "CONTAINS_ENTITY": {"color": "rgba(155,89,182,0.6)", "width": 2, "dash": "dot", "name": "Contains Entity"},
        "RELATED_TO": {"color": "rgba(52,152,219,0.6)", "width": 1.5, "dash": "dashdot", "name": "Entity Relation"}
    }

    edge_legend_added = set()  # Track which edge types have been added to legend
    
    for edge in graph_data["edges"]:
        source_id, target_id = edge["source"], edge["target"]
        if source_id in pos and target_id in pos:
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]

            # Get edge styling
            edge_style = edge_color_map.get(
                edge["type"],
                {"color": "rgba(125,125,125,0.3)", "width": 1, "name": "Other"}
            )
            
            line_dict = {
                "color": edge_style["color"],
                "width": edge_style["width"]
            }
            if "dash" in edge_style:
                line_dict["dash"] = edge_style["dash"]

            # Determine if this edge type should show in legend
            show_legend = edge["type"] not in edge_legend_added
            if show_legend:
                edge_legend_added.add(edge["type"])

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=line_dict,
                hoverinfo="text" if show_legend else "none",
                hovertext=f"{edge_style['name']}: {edge.get('weight', 1.0):.2f}" if show_legend else None,
                name=edge_style["name"],
                showlegend=show_legend,
                legendgroup="edges" if show_legend else None
            )
            edge_traces.append(edge_trace)

    # Create figure with all traces
    fig = go.Figure(data=node_traces + edge_traces)

    # Determine title based on retrieval mode
    retrieval_mode = graph_data.get("retrieval_mode", "unknown")
    mode_descriptions = {
        "chunk_only": "Chunk-based Knowledge Graph",
        "entity_only": "Entity-based Knowledge Graph",
        "hybrid": "Hybrid Knowledge Graph (Chunks + Entities)"
    }
    title_text = mode_descriptions.get(retrieval_mode, "Knowledge Graph")
    title_text += f" ({graph_data['total_nodes']} nodes, {graph_data['total_edges']} edges)"

    # Update layout with enhanced styling
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=60),
        annotations=[
            dict(
                text=f"Mode: {retrieval_mode.replace('_', ' ').title()} | Hover over nodes and edges for details",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.05,
                xanchor="center",
                yanchor="bottom",
                font=dict(color="gray", size=10),
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=700,  # Slightly taller to accommodate legend
    )

    return fig


def get_query_graph_data(
    query_results: List[Dict[str, Any]], query_text: str = "Your Question"
) -> Dict[str, Any]:
    """
    Create simplified graph data for visualizing query results with only essential relationships.

    Args:
        query_results: List of retrieved chunks with metadata
        query_text: The actual query text to display

    Returns:
        Dictionary containing nodes and edges for a clean, readable query result graph
    """
    from config.settings import settings
    
    nodes = []
    edges = []

    # Filter out chunks with low or missing similarity scores up-front
    filtered_results: List[Dict[str, Any]] = []
    for result in query_results:
        similarity = result.get("similarity")
        if similarity is not None and similarity >= settings.min_retrieval_similarity:
            filtered_results.append(result)
        elif similarity is not None:
            logger.debug(
                f"Filtered chunk with similarity {similarity} (below threshold {settings.min_retrieval_similarity})"
            )

    query_results = filtered_results

    if not query_results:
        # Return empty graph if no results meet the threshold
        return {"nodes": nodes, "edges": edges, "total_nodes": 0, "total_edges": 0}

    # Build document-level aggregation from the retrieved chunks
    # Key by document_id when available, otherwise use document_name
    doc_map: Dict[str, Dict[str, Any]] = {}

    for res in query_results:
        doc_id = res.get("document_id") or res.get("filename") or res.get("document_name") or "unknown_doc"
        doc_name = res.get("document_name") or res.get("filename") or str(doc_id)
        chunk_id = res.get("chunk_id")
        similarity = res.get("similarity", 0.0)

        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "document_id": doc_id,
                "document_name": doc_name,
                "chunks": set(),
                "chunk_similarities": [],
                "entities": set(),
            }

        if chunk_id:
            doc_map[doc_id]["chunks"].add(chunk_id)
            doc_map[doc_id]["chunk_similarities"].append(similarity)

        # Try to collect entities for this chunk from different possible places
        # Prefer any chunk_data populated earlier (if present in local scope), else look in result
        entities_list = []
        if isinstance(res.get("entities"), list):
            entities_list = [e.get("entity_name") if isinstance(e, dict) else e for e in res.get("entities", [])]
        elif isinstance(res.get("contained_entities"), list):
            entities_list = res.get("contained_entities", [])
        # Normalize
        for ent in entities_list:
            if ent:
                doc_map[doc_id]["entities"].add(ent)

    # Add query node
    query_display = query_text[:60] + "..." if len(query_text) > 60 else query_text
    nodes.append({
        "id": "query",
        "entity_id": "query",
        "label": "Query",
        "title": query_display,
        "full_text": query_text,
        "size": 30,
        "node_type": "query",
    })

    # Create document nodes
    doc_id_to_node_id = {}
    for idx, (doc_id, info) in enumerate(doc_map.items()):
        node_id = f"doc_{idx}"
        doc_id_to_node_id[doc_id] = node_id
        chunk_count = len(info["chunks"])
        entity_count = len(info["entities"])
        avg_similarity = (sum(info["chunk_similarities"]) / len(info["chunk_similarities"])) if info["chunk_similarities"] else 0.0
        display_title = info.get("document_name") or str(doc_id)

        nodes.append({
            "id": node_id,
            "entity_id": doc_id,
            "label": "Document",
            "title": display_title,
            "document_name": display_title,
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "avg_similarity": avg_similarity,
            "size": max(20, min(60, 20 + (chunk_count * 5) + (entity_count * 3))),
        })

        # Add edge from query to document summarizing counts
        rel_label = f"{chunk_count} chunk{'s' if chunk_count != 1 else ''}, {entity_count} entit{'ies' if entity_count != 1 else 'y'}"
        edges.append({
            "source": "query",
            "target": node_id,
            "type": "RELEVANT_TO",
            "relationship_label": rel_label,
            "weight": avg_similarity or 0.1,
            "edge_type": "retrieval",
        })

    # Create document-document edges indicating shared entities (what they share in common)
    doc_items = list(doc_map.items())
    for i in range(len(doc_items)):
        id_i, info_i = doc_items[i]
        for j in range(i + 1, len(doc_items)):
            id_j, info_j = doc_items[j]
            shared_entities = set(info_i["entities"]) & set(info_j["entities"])
            if shared_entities:
                source_node = doc_id_to_node_id[id_i]
                target_node = doc_id_to_node_id[id_j]
                # Human readable label: list top 3 shared entities
                rel_label = "shares: " + ", ".join(list(shared_entities)[:3])
                edges.append({
                    "source": source_node,
                    "target": target_node,
                    "type": "SHARES_ENTITY",
                    "relationship_label": rel_label,
                    "weight": len(shared_entities),
                    "edge_type": "shared",
                })

    # --- NEW: include top-N chunks per document and chunk-chunk relationships ---
    # Select top N chunks per document by similarity across the filtered query results
    top_chunks: List[Dict[str, Any]] = []
    top_n_per_doc = 5
    # Group chunks by document id
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for r in query_results:
        did = r.get("document_id") or r.get("document_name") or "unknown_doc"
        chunks_by_doc.setdefault(did, []).append(r)

    # For each document pick the top N chunks
    for did, chunks in chunks_by_doc.items():
        sorted_chunks = sorted(chunks, key=lambda r: r.get("similarity", 0.0), reverse=True)
        top_chunks.extend(sorted_chunks[:top_n_per_doc])

    # Track which chunk ids we've added to avoid duplicates
    added_chunk_ids = set()

    for idx, ch in enumerate(top_chunks):
        chunk_id = ch.get("chunk_id")
        if not chunk_id or chunk_id in added_chunk_ids:
            continue

        similarity = ch.get("similarity", 0.0)
        doc_id = ch.get("document_id") or ch.get("document_name") or "unknown_doc"
        doc_node = doc_id_to_node_id.get(doc_id)
        display_preview = (ch.get("full_content") or ch.get("content") or ch.get("text") or ch.get("title") or "")[0:140]

        # Try to extract contained entities for richer chunk-chunk labels
        chunk_entities = []
        if isinstance(ch.get("entities"), list):
            chunk_entities = [e.get("entity_name") if isinstance(e, dict) else e for e in ch.get("entities", [])]
        elif isinstance(ch.get("contained_entities"), list):
            chunk_entities = ch.get("contained_entities", [])

        nodes.append({
            "id": str(chunk_id),
            "entity_id": chunk_id,
            "label": "Chunk",
            "title": display_preview or f"Chunk {idx + 1}",
            "full_content": display_preview,
            "similarity": similarity,
            "document_name": doc_map.get(doc_id, {}).get("document_name") if doc_id in doc_map else ch.get("document_name") or ch.get("filename"),
            "entities": chunk_entities,
            "size": max(12, min(40, 12 + int(similarity * 40))),
            "chunk_index": idx,
        })

        # Add edge from document node to chunk node
        if doc_node:
            edges.append({
                "source": doc_node,
                "target": str(chunk_id),
                "type": "HAS_CHUNK",
                "weight": similarity or 0.1,
                "edge_type": "doc-chunk",
            })

        added_chunk_ids.add(chunk_id)

    # Attempt to fetch chunk-chunk relationships from the DB for the selected chunks
    try:
        if added_chunk_ids and hasattr(graph_db, "driver"):
            with graph_db.driver.session() as session:  # type: ignore
                db_query = """
                MATCH (c1:Chunk)-[r]-(c2:Chunk)
                WHERE elementId(c1) IN $chunk_ids AND elementId(c2) IN $chunk_ids
                RETURN elementId(c1) as source_id, elementId(c2) as target_id, type(r) as rel_type,
                       coalesce(properties(r)['similarity'], properties(r)['weight'], 1.0) as weight
                LIMIT 200
                """
                db_result = session.run(db_query, chunk_ids=list(added_chunk_ids))
                seen_pairs = set()
                for rec in db_result:
                    rd = rec.data()
                    s = str(rd.get("source_id"))
                    t = str(rd.get("target_id"))
                    if s == t:
                        continue
                    pair = tuple(sorted((s, t)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    weight = rd.get("weight", 1.0)
                    rel_type = rd.get("rel_type") or "SIMILAR_TO"
                    rel_label = None
                    try:
                        ents_a = next((n.get("entities") for n in nodes if str(n.get("id")) == s), []) or []
                        ents_b = next((n.get("entities") for n in nodes if str(n.get("id")) == t), []) or []
                        shared = list(set(ents_a) & set(ents_b))
                        if shared:
                            rel_label = "shares: " + ", ".join(shared[:3])
                        else:
                            rel_label = (
                                "high similarity"
                                if weight >= 0.85
                                else ("moderate similarity" if weight >= 0.6 else "related")
                            )
                    except Exception:
                        rel_label = None

                    edges.append({
                        "source": s,
                        "target": t,
                        "type": rel_type,
                        "relationship_label": rel_label,
                        "weight": weight,
                        "edge_type": "chunk-related",
                    })
    except Exception as e:
        logger.debug(f"Could not fetch chunk-chunk relationships from DB: {e}")

    # As a fallback/enrichment, link chunks from the same document (if DB links not found)
    # This makes the hierarchy clearer: Query -> Document -> Chunk and Chunk <-> Chunk
    chunk_list = list(added_chunk_ids)
    # Map chunk_id to its parent document node id (if available)
    chunk_to_doc = {}
    for ch in top_chunks:
        cid = ch.get("chunk_id")
        if not cid:
            continue
        did = ch.get("document_id") or ch.get("document_name")
        if did and did in doc_id_to_node_id:
            chunk_to_doc[str(cid)] = doc_id_to_node_id[did]

    # Add SAME_DOCUMENT relationships between chunks that share a document
    seen_chunk_rel = set()
    for a, b in ((a, b) for i, a in enumerate(chunk_list) for b in chunk_list[i + 1 :]):
        doc_a = chunk_to_doc.get(str(a))
        doc_b = chunk_to_doc.get(str(b))
        if doc_a and doc_b and doc_a == doc_b:
            pair = tuple(sorted((str(a), str(b))))
            if pair in seen_chunk_rel:
                continue
            seen_chunk_rel.add(pair)
            # We don't display the label on the edge; keep it for structure only
            edges.append({
                "source": str(a),
                "target": str(b),
                "type": "SAME_DOCUMENT_CHUNK",
                "relationship_label": None,
                "weight": 0.5,
                "edge_type": "chunk-related",
            })
    # Build a per-chunk shared-entities map so we can show shares in chunk hover text
    # Map chunk_id -> {shared_entities: set(), shared_with: list of chunk ids}
    chunk_shared_info: Dict[str, Dict[str, Any]] = {}
    # Initialize map for added chunks
    for n in nodes:
        if n.get("label") == "Chunk":
            chunk_shared_info[str(n["id"])] = {"shared_entities": set(n.get("entities", [])), "shared_with": {}}

    # Build a helper map to convert chunk id -> display number (C#)
    chunk_display_map: Dict[str, int] = {}
    for n in nodes:
        if n.get("label") == "Chunk":
            cid = str(n.get("id"))
            # chunk_index is zero-based; display C{index+1}
            chunk_display_map[cid] = int(n.get("chunk_index", 0)) + 1

    # Compare chunks for shared entities and fill shared_with mapping
    chunk_ids_list = list(chunk_shared_info.keys())
    for i in range(len(chunk_ids_list)):
        a = chunk_ids_list[i]
        for j in range(i + 1, len(chunk_ids_list)):
            b = chunk_ids_list[j]
            ents_a = chunk_shared_info[a]["shared_entities"]
            ents_b = chunk_shared_info[b]["shared_entities"]
            shared = list(ents_a & ents_b)
            if shared:
                # record for both directions
                chunk_shared_info[a]["shared_with"][b] = shared
                chunk_shared_info[b]["shared_with"][a] = shared

    # Attach a compact summary string to the chunk nodes for hover display
    for n in nodes:
        if n.get("label") == "Chunk":
            cid = str(n.get("id"))
            info = chunk_shared_info.get(cid, {})
            shared_with = info.get("shared_with", {})
            if shared_with:
                # Build top-5 per-chunk shared-entity lines like:
                # shares: Alice with C4
                # shares: Bob with C2
                # Iterate other chunks and their shared entities
                # We want to group by entity so each entity lists all other chunks it is shared with
                entity_to_chunks: Dict[str, List[str]] = {}
                for other_cid, ents in shared_with.items():
                    display_num = chunk_display_map.get(str(other_cid))
                    other_display = f"C{display_num}" if display_num is not None else str(other_cid)
                    for ent in ents[:5]:
                        entity_to_chunks.setdefault(ent, []).append(other_display)

                grouped_lines = []
                for ent, chunk_list in entity_to_chunks.items():
                    # dedupe and keep order
                    seen = []
                    for c in chunk_list:
                        if c not in seen:
                            seen.append(c)
                    grouped_lines.append(f"<b>shares: </b> {ent} with {', '.join(seen)}")

                # Keep only top 5 grouped lines for compactness
                n["shares_summary"] = "<br>".join(grouped_lines[:5]) if grouped_lines else None
            else:
                n["shares_summary"] = None
    return {
        "nodes": nodes,
        # Remove any chunk->chunk edges for readability (keep document->chunk and doc-doc edges)
        "edges": [
            e for e in edges
            if not (
                any(n.get("label") == "Chunk" and str(n.get("id")) == str(e.get("source")) for n in nodes)
                and any(n.get("label") == "Chunk" and str(n.get("id")) == str(e.get("target")) for n in nodes)
            )
        ],
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


def create_query_result_graph(
    query_results: List[Dict[str, Any]], query_text: str = "Your Question"
) -> go.Figure:
    """
    Create an enhanced graph visualization for query results with full database relationships.

    Args:
        query_results: List of retrieved chunks with metadata
        query_text: The actual query text to display

    Returns:
        Plotly figure showing query relationships with enhanced styling
    """
    graph_data = get_query_graph_data(query_results, query_text)

    if not graph_data["nodes"] or len(graph_data["nodes"]) <= 1:
        # Return empty figure if no meaningful data
        fig = go.Figure()
        fig.update_layout(
            title="No query results to visualize",
            showlegend=False,
            xaxis={"showgrid": False, "zeroline": False, "visible": False},
            yaxis={"showgrid": False, "zeroline": False, "visible": False},
        )
        return fig

    # Create NetworkX graph for layout calculation (we will override positions
    # to force a hierarchical layout: Query -> Documents -> Chunks)
    G = create_networkx_graph(graph_data)

    # Build simple hierarchical positions: query at top, documents in middle, chunks at bottom
    pos: Dict[str, Any] = {}
    # Collect node ids by type
    docs = [n for n in graph_data["nodes"] if n["label"] == "Document"]
    chunks = [n for n in graph_data["nodes"] if n["label"] == "Chunk"]
    queries = [n for n in graph_data["nodes"] if n["label"] == "Query"]

    # Query position: top center
    if queries:
        qid = queries[0]["id"]
        pos[qid] = (0.0, 1.0)

    # Documents: evenly spaced on middle x, but alternate vertical offset up/down for readability
    doc_count = len(docs) or 1
    for i, d in enumerate(docs):
        x = -1.0 + 2.0 * (i / max(1, doc_count - 1)) if doc_count > 1 else 0.0
        # alternate y offset: up (0.28), down (0.12), up, down...
        y = 0.28 if (i % 2) == 0 else 0.12
        pos[d["id"]] = (x, y)

    # Group chunks by their parent doc (if available) to place them under their document
    chunks_by_doc = {}
    for c in chunks:
        # try to find parent doc via nodes list edges or document_name on chunk
        parent = None
        parent_doc_name = c.get("document_name")
        if parent_doc_name:
            # find matching document node id
            for dn in docs:
                if dn.get("document_name") == parent_doc_name or dn.get("title") == parent_doc_name:
                    parent = dn.get("id")
                    break
        chunks_by_doc.setdefault(parent, []).append(c)

    # Place chunks under each document, but with extra spacing for readability
    for idx, (parent, clist) in enumerate(chunks_by_doc.items()):
        count = len(clist)
        for j, c in enumerate(clist):
            if parent and parent in pos:
                parent_x, parent_y = pos[parent]
                # wider horizontal spread and more vertical gap than before
                span = 0.8 + 0.15 * max(0, count - 1)
                x = parent_x - span / 2 + (span * (j / max(1, count - 1))) if count > 1 else parent_x
                y = parent_y - 0.6 - 0.08 * (j % 3)
            else:
                # fall back to spread across bottom row with more spacing
                x = -1.0 + 2.0 * (j / max(1, count - 1)) if count > 1 else 0.0
                y = -0.9
            pos[c["id"]] = (x, y)

    # For any remaining nodes not positioned (entities, extras), fall back to spring layout positions
    remaining = [n["id"] for n in graph_data["nodes"] if n["id"] not in pos]
    if remaining:
        # compute spring positions for the whole graph then copy only missing ones
        fallback = nx.spring_layout(G, k=1, iterations=50)
        for nid in remaining:
            if nid in fallback:
                pos[nid] = fallback[nid]

    # Enhanced color map for different node types (matching full graph)
    color_map = {
        "Query": "#E74C3C",      # Bright red for query
        "Document": "#FF6B6B",   # Red for documents
        "Chunk": "#4ECDC4",      # Teal for chunks
        "Entity": "#9B59B6",     # Purple for entities
        "Unknown": "#95A5A6"     # Gray for unknown
    }
    
    # Shape map for different node types (using symbols)
    symbol_map = {
        "Query": "star",
        "Document": "square",
        "Chunk": "circle",
        "Entity": "diamond",
        "Unknown": "circle"
    }

    # Prepare node traces by type for better legend support
    node_types = {}
    for node in graph_data["nodes"]:
        node_type = node["label"]
        if node_type not in node_types:
            node_types[node_type] = {
                "x": [], "y": [], "text": [], "sizes": [], "colors": [],
                "hover_texts": [], "symbols": []
            }

    # Organize nodes by type
    for node in graph_data["nodes"]:
        node_id = node["id"]
        if node_id in pos:
            x, y = pos[node_id]
            node_type = node["label"]
            
            node_types[node_type]["x"].append(x)
            node_types[node_type]["y"].append(y)
            
            # Create appropriate display text based on node type
            if node_type == "Query":
                display_text = "Q"
                hover_text = f"<b>Your Query:</b><br>{node.get('full_text', 'Your Question')}"
            elif node_type == "Document":
                display_text = "📄"
                file_size = node.get("file_size", 0)
                size_str = ""
                if file_size:
                    if file_size > 1024 * 1024:
                        size_str = f" ({file_size / (1024 * 1024):.1f} MB)"
                    elif file_size > 1024:
                        size_str = f" ({file_size / 1024:.1f} KB)"
                chunk_count = node.get("chunk_count", 0)
                entity_count = node.get("entity_count", 0)
                avg_sim = node.get("avg_similarity", 0.0)
                hover_text = (
                    f"<b>Document:</b> {node['title']}{size_str}<br>"
                    f"<b>Chunks:</b> {chunk_count} &nbsp; <b>Entities:</b> {entity_count}<br>"
                    f"<b>Avg Relevance:</b> {avg_sim:.3f}"
                )
            elif node_type == "Entity":
                display_text = node["title"][:3]
                entity_type = node.get("entity_type", "Unknown")
                confidence = node.get("confidence", 0)
                hover_text = f"<b>Entity:</b> {node['title']}<br><b>Type:</b> {entity_type}<br><b>Confidence:</b> {confidence:.2f}"
            else:  # Chunk
                chunk_num = node.get("chunk_index", 0)
                display_text = f"C{chunk_num + 1}"
                similarity = node.get("similarity", 0.0)
                # Compact hover: include short content snippet, relevance, and shares lines
                snippet = (node.get("full_content") or node.get("title") or "")[0:120]
                if len(snippet) >= 120:
                    snippet = snippet.rstrip() + "..."
                hover_text = f"<b>Chunk {chunk_num + 1}</b><br><b>Relevance:</b> {similarity:.3f}"
                if snippet:
                    hover_text += f"<br><b>Snippet:</b> {snippet}"
                # If this chunk shares entities with other chunks, include the prepared shares_summary
                shares_summary = node.get("shares_summary")
                if shares_summary:
                    # shares_summary already contains 'shares: ...' lines joined with <br>
                    hover_text += f"<br>{shares_summary}"
            
            node_types[node_type]["text"].append(display_text)
            node_types[node_type]["sizes"].append(node["size"])
            node_types[node_type]["colors"].append(color_map.get(node_type, "#95A5A6"))
            node_types[node_type]["symbols"].append(symbol_map.get(node_type, "circle"))
            node_types[node_type]["hover_texts"].append(hover_text)

    # Create separate traces for each node type
    node_traces = []
    for node_type, data in node_types.items():
        if data["x"]:  # Only create trace if there are nodes of this type
            node_trace = go.Scatter(
                x=data["x"],
                y=data["y"],
                text=data["text"],
                textposition="middle center",
                mode="markers+text",
                hovertext=data["hover_texts"],
                hoverinfo="text",
                name=node_type,
                marker=dict(
                    size=data["sizes"],
                    color=color_map.get(node_type, "#95A5A6"),
                    symbol=data["symbols"][0] if data["symbols"] else "circle",
                    line=dict(width=2, color="white")
                ),
                showlegend=True
            )
            node_traces.append(node_trace)

    # Prepare simplified edge traces with readable labels
    edge_traces = []
    # No edge labels: links should have no visible labels
    
    edge_color_map = {
        "RELEVANT_TO": {"color": "rgba(231,76,60,0.8)", "width": 3, "name": "Relevant To"},
        "MENTIONS": {"color": "rgba(155,89,182,0.7)", "width": 2, "name": "Mentions"},
        "CONNECTED_VIA": {"color": "rgba(52,152,219,0.6)", "width": 2, "name": "Connected Via"}
    }

    edge_legend_added = set()  # Track which edge types have been added to legend
    
    for edge in graph_data["edges"]:
        source_id, target_id = edge["source"], edge["target"]
        if source_id in pos and target_id in pos:
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]

            # Get edge styling
            edge_style = edge_color_map.get(
                edge["type"],
                {"color": "rgba(125,125,125,0.5)", "width": 1, "name": "Related"}
            )
            
            line_dict = {
                "color": edge_style["color"],
                "width": edge_style["width"]
            }

            # Determine if this edge type should show in legend
            show_legend = edge["type"] not in edge_legend_added
            if show_legend:
                edge_legend_added.add(edge["type"])

            # Do not show hover text or labels for links per user request
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=line_dict,
                hoverinfo="none",
                hovertext=None,
                name=edge_style["name"],
                showlegend=show_legend,
                legendgroup="edges"
            )
            edge_traces.append(edge_trace)

    # We'll collect annotations for document-document labels only
    doc_edge_annotations = []

    # Create figure with all traces
    fig = go.Figure(data=node_traces + edge_traces)

    # Add document-document edge labels as simple annotations (no rotation)
    # Find document-document edges that have relationship_label
    for e in graph_data.get("edges", []):
        if e.get("type") == "SHARES_ENTITY" and e.get("relationship_label"):
            s = e.get("source")
            t = e.get("target")
            if s in pos and t in pos:
                mid_x = (pos[s][0] + pos[t][0]) / 2
                mid_y = (pos[s][1] + pos[t][1]) / 2
                doc_edge_annotations.append({
                    "x": mid_x,
                    "y": mid_y,
                    "xref": "x",
                    "yref": "y",
                    "text": e.get("relationship_label"),
                    "showarrow": False,
                    "font": {"size": 9, "color": "rgba(80,80,80,0.9)"},
                    "align": "center",
                    "bgcolor": "rgba(255,255,255,0.6)",
                })

    # Create title
    title_text = f"Query Context Graph ({graph_data['total_nodes']} nodes, {graph_data['total_edges']} edges)"

    # Update layout with enhanced styling
    # Footer annotation only (no edge annotations)
    footer_annotation = dict(
        text="Context visualization showing retrieved chunks and their relationships | Hover for details",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.05,
        xanchor="center",
        yanchor="bottom",
        font=dict(color="gray", size=10),
    )
    annotations_list = [footer_annotation]
    # append document edge annotations if any
    if 'doc_edge_annotations' in locals() and doc_edge_annotations:
        annotations_list.extend(doc_edge_annotations)

    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=60),
        annotations=annotations_list,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=600,
    )

    return fig
