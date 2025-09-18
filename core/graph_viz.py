"""
Graph visualization utilities for Neo4j data using NetworkX and Plotly.
"""

import logging
from typing import Any, Dict, List

import networkx as nx
import plotly.graph_objects as go

from core.graph_db import graph_db

logger = logging.getLogger(__name__)


def get_graph_data(limit: int = 100) -> Dict[str, Any]:
    """
    Retrieve graph data from Neo4j for visualization.

    Args:
        limit: Maximum number of nodes to retrieve

    Returns:
        Dictionary containing nodes and edges for visualization
    """
    try:
        with graph_db.driver.session() as session:  # type: ignore
            # Get nodes (documents and chunks)
            nodes_query = """
            MATCH (n)
            WHERE n:Document OR n:Chunk
            RETURN
                elementId(n) as node_id,
                labels(n) as labels,
                n.id as entity_id,
                CASE
                    WHEN n:Document THEN n.filename
                    WHEN n:Chunk THEN substring(n.content, 0, 50) + "..."
                    ELSE "Unknown"
                END as title,
                CASE
                    WHEN n:Document THEN size(n.content)
                    WHEN n:Chunk THEN size(n.content)
                    ELSE 0
                END as content_size
            LIMIT $limit
            """

            nodes_result = session.run(nodes_query, limit=limit)
            nodes = []
            node_ids = set()

            for record in nodes_result:
                node_data = record.data()
                if node_data["content_size"] is None:
                    node_data["content_size"] = 0
                nodes.append(
                    {
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
                )
                node_ids.add(str(node_data["node_id"]))

            # Get relationships between the nodes
            edges_query = """
            MATCH (n)-[r]-(m)
            WHERE (n:Document OR n:Chunk) AND (m:Document OR m:Chunk)
            AND elementId(n) IN $node_ids AND elementId(m) IN $node_ids
            RETURN
                elementId(n) as source_id,
                elementId(m) as target_id,
                type(r) as relationship_type,
                coalesce(properties(r)['similarity'], 1.0) as weight
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
            }

    except Exception as e:
        logger.error(f"Failed to retrieve graph data: {e}")
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}


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
    Create an interactive Plotly graph visualization.

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

    # Color map for different node types
    color_map = {"Document": "#FF6B6B", "Chunk": "#4ECDC4", "Unknown": "#95A5A6"}

    # Prepare node traces
    x_coords = []
    y_coords = []
    text_labels = []
    sizes = []
    colors = []

    for node in graph_data["nodes"]:
        node_id = node["id"]
        if node_id in pos:
            x, y = pos[node_id]
            x_coords.append(x)
            y_coords.append(y)
            text_labels.append(node["title"][:20])  # Truncate long titles
            sizes.append(max(node["size"], 15))
            colors.append(color_map.get(node["label"], "#95A5A6"))

    # Create node trace
    node_trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        text=text_labels,
        textposition="middle center",
        mode="markers+text",
        hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=2)),
    )

    # Prepare edge traces
    edge_traces = []

    for edge in graph_data["edges"]:
        source_id, target_id = edge["source"], edge["target"]
        if source_id in pos and target_id in pos:
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]

            # Different line styles for different relationship types
            line_style = {
                "HAS_CHUNK": dict(color="rgba(125,125,125,0.5)", width=2),
                "SIMILAR_TO": dict(color="rgba(255,107,107,0.7)", width=1, dash="dash"),
            }.get(edge["type"], dict(color="rgba(125,125,125,0.3)", width=1))

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=line_style,
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

    # Create figure
    fig = go.Figure(data=[node_trace] + edge_traces)

    # Update layout
    fig.update_layout(
        title={
            "text": f"Knowledge Graph ({graph_data['total_nodes']} nodes, {graph_data['total_edges']} edges)",
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Documents (red) connected to Chunks (teal). Dashed lines show similarity relationships.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                xanchor="left",
                yanchor="bottom",
                font=dict(color="gray", size=10),
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=600,
    )

    return fig


def get_query_graph_data(
    query_results: List[Dict[str, Any]], query_text: str = "Your Question"
) -> Dict[str, Any]:
    """
    Create graph data for visualizing query results and their relationships.

    Args:
        query_results: List of retrieved chunks with metadata
        query_text: The actual query text to display

    Returns:
        Dictionary containing nodes and edges for the query result graph
    """
    nodes = []
    edges = []

    # Add query node with actual query text
    query_display = query_text[:50] + "..." if len(query_text) > 50 else query_text
    nodes.append(
        {
            "id": "query",
            "entity_id": "query",
            "label": "Query",
            "title": query_display,
            "full_text": query_text,
            "size": 35,
            "node_type": "query",
        }
    )

    # Add result nodes and connections
    for i, result in enumerate(query_results):
        chunk_id = f"chunk_{i}"
        chunk_content = result.get("content", "No content")
        similarity = result.get("similarity", 0)
        document_name = result.get(
            "document_name", result.get("filename", "Unknown Document")
        )

        # Truncate content for display
        display_content = (
            chunk_content[:80] + "..." if len(chunk_content) > 80 else chunk_content
        )

        nodes.append(
            {
                "id": chunk_id,
                "entity_id": result.get("chunk_id", f"unknown_{i}"),
                "label": "Chunk",
                "title": display_content.replace("\n", " "),
                "full_content": chunk_content,
                "document_name": document_name,
                "size": 15 + min(similarity * 25, 20),  # Size based on similarity
                "similarity": similarity,
                "node_type": "chunk",
                "chunk_index": i,
            }
        )

        # Connect query to result with detailed relationship info
        edges.append(
            {
                "source": "query",
                "target": chunk_id,
                "type": "RETRIEVED",
                "relationship_label": f"Relevance: {similarity:.3f}",
                "weight": similarity,
                "edge_type": "retrieval",
            }
        )

    # Try to find relationships between chunks using content similarity and document source
    for i, result1 in enumerate(query_results):
        for j, result2 in enumerate(query_results[i + 1 :], i + 1):
            # Check if chunks are from same document
            doc1 = result1.get("document_name", result1.get("filename", ""))
            doc2 = result2.get("document_name", result2.get("filename", ""))

            if doc1 == doc2 and doc1:  # Same document relationship
                edges.append(
                    {
                        "source": f"chunk_{i}",
                        "target": f"chunk_{j}",
                        "type": "SAME_DOCUMENT",
                        "relationship_label": f"From: {doc1[:20]}...",
                        "weight": 0.8,
                        "edge_type": "document_relation",
                    }
                )
            else:
                # Simple word overlap similarity for different documents
                words1 = set(result1.get("content", "").lower().split())
                words2 = set(result2.get("content", "").lower().split())

                if words1 and words2:
                    overlap = len(words1.intersection(words2))
                    if overlap >= 3:  # Minimum word overlap threshold
                        similarity_score = overlap / len(words1.union(words2))
                        if (
                            similarity_score > 0.15
                        ):  # Slightly higher threshold for cross-doc relations
                            edges.append(
                                {
                                    "source": f"chunk_{i}",
                                    "target": f"chunk_{j}",
                                    "type": "CONTENT_SIMILAR",
                                    "relationship_label": f"Similarity: {similarity_score:.2f}",
                                    "weight": similarity_score,
                                    "edge_type": "content_relation",
                                }
                            )

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


def create_query_result_graph(
    query_results: List[Dict[str, Any]], query_text: str = "Your Question"
) -> go.Figure:
    """
    Create a graph visualization for query results.

    Args:
        query_results: List of retrieved chunks with metadata

    Returns:
        Plotly figure showing query relationships
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

    # Create NetworkX graph for layout
    G = create_networkx_graph(graph_data)
    pos = nx.spring_layout(G, k=3, iterations=50)  # Increased k for better spacing

    # Create node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        textposition="middle center",
        mode="markers+text",
        hoverinfo="text",
        marker=dict(size=[], color=[]),
    )

    # Enhanced color map for different node types
    color_map = {
        "Query": "#FF6B6B",  # Red for query
        "Chunk": "#4ECDC4",  # Teal for chunks
    }

    hover_texts = []
    x_coords = []
    y_coords = []
    text_labels = []
    sizes = []
    colors = []

    for node in graph_data["nodes"]:
        node_id = node["id"]
        if node_id in pos:
            x, y = pos[node_id]
            x_coords.append(x)
            y_coords.append(y)

            # Show abbreviated text on node
            if node["label"] == "Query":
                text_labels.append("Q")
                hover_text = (
                    f"<b>Your Query:</b><br>{node.get('full_text', 'Your Question')}"
                )
            else:
                chunk_num = node_id.split("_")[1]
                text_labels.append(f"C{int(chunk_num) + 1}")
                similarity = node.get("similarity", 0)
                doc_name = node.get("document_name", "Unknown Document")
                content_preview = node.get("full_content", node["title"])[:150] + "..."
                hover_text = (
                    f"<b>Chunk {int(chunk_num) + 1}</b><br>"
                    f"<b>Document:</b> {doc_name}<br>"
                    f"<b>Relevance:</b> {similarity:.3f}<br>"
                    f"<b>Content:</b> {content_preview}"
                )

            hover_texts.append(hover_text)
            sizes.append(node["size"])
            colors.append(color_map.get(node["label"], "#95A5A6"))

    # Update node trace with collected data
    node_trace.update(
        x=x_coords,
        y=y_coords,
        text=text_labels,
        hovertext=hover_texts,
        marker=dict(size=sizes, color=colors, line=dict(width=2)),
    )

    # Set hover text
    node_trace["hoverinfo"] = "text"

    # Create edge traces with labels
    edge_traces = []
    edge_labels = []

    for edge in graph_data["edges"]:
        source_id, target_id = edge["source"], edge["target"]
        if source_id in pos and target_id in pos:
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]

            # Different styling for different edge types
            edge_type = edge.get("edge_type", edge["type"])

            if edge_type == "retrieval" or edge["type"] == "RETRIEVED":
                # Line width and opacity based on similarity/weight
                line_width = max(2, edge["weight"] * 6)
                alpha = max(0.6, edge["weight"])
                color = f"rgba(78,205,196,{alpha})"
                dash_style = None
            elif edge_type == "document_relation" or edge["type"] == "SAME_DOCUMENT":
                line_width = 3
                alpha = 0.7
                color = f"rgba(52,152,219,{alpha})"  # Blue for same document
                dash_style = "dot"
            else:  # content_relation or CONTENT_SIMILAR
                line_width = max(1, edge["weight"] * 4)
                alpha = max(0.4, edge["weight"])
                color = f"rgba(255,107,107,{alpha})"  # Red for content similarity
                dash_style = "dash"

            # Create edge trace
            line_dict = dict(color=color, width=line_width)
            if dash_style:
                line_dict["dash"] = dash_style

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=line_dict,
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

            # Add relationship label at midpoint of edge
            if "relationship_label" in edge:
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                edge_labels.append(
                    {
                        "x": mid_x,
                        "y": mid_y,
                        "text": edge["relationship_label"],
                        "font_size": 8,
                        "font_color": "rgba(100,100,100,0.8)",
                    }
                )

    # Add edge labels as text annotations
    edge_label_traces = []
    if edge_labels:
        for label in edge_labels:
            edge_label_trace = go.Scatter(
                x=[label["x"]],
                y=[label["y"]],
                text=[label["text"]],
                mode="text",
                textfont=dict(size=label["font_size"], color=label["font_color"]),
                showlegend=False,
                hoverinfo="none",
            )
            edge_label_traces.append(edge_label_trace)

    # Create figure with all traces
    fig = go.Figure(data=[node_trace] + edge_traces + edge_label_traces)

    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=40, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Teal lines: Query relevance | Blue dots: Same document | Red dashes: Content similarity",
                showarrow=True,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.08,
                xanchor="center",
                yanchor="top",
                font=dict(color="gray", size=10),
            ),
            dict(
                text="Hover over nodes for detailed information",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.12,
                xanchor="center",
                yanchor="top",
                font=dict(color="gray", size=9),
            ),
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=600,
    )

    return fig
