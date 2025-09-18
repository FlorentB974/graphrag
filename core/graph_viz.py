"""
Graph visualization utilities for Neo4j data using NetworkX and Plotly.
"""
import logging
from typing import Dict, List, Any
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
                if node_data['content_size'] is None:
                    node_data['content_size'] = 0
                nodes.append({
                    'id': str(node_data['node_id']),
                    'entity_id': node_data['entity_id'],
                    'label': node_data['labels'][0] if node_data['labels'] else 'Unknown',
                    'title': node_data['title'] or 'Untitled',
                    'size': min(max(int(node_data['content_size']) / 100, 10), 50)  # Scale size
                })
                node_ids.add(str(node_data['node_id']))
            
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
            
            edges_result = session.run(edges_query, node_ids=list(node_ids), limit=limit)
            edges = []
            
            for record in edges_result:
                edge_data = record.data()
                edges.append({
                    'source': str(edge_data['source_id']),
                    'target': str(edge_data['target_id']),
                    'type': edge_data['relationship_type'],
                    'weight': edge_data['weight']
                })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
            
    except Exception as e:
        logger.error(f"Failed to retrieve graph data: {e}")
        return {'nodes': [], 'edges': [], 'total_nodes': 0, 'total_edges': 0}


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
    for node in graph_data['nodes']:
        G.add_node(
            node['id'],
            label=node['label'],
            title=node['title'],
            entity_id=node['entity_id'],
            size=node['size']
        )
    
    # Add edges
    for edge in graph_data['edges']:
        if edge['source'] in G.nodes and edge['target'] in G.nodes:
            G.add_edge(
                edge['source'],
                edge['target'],
                type=edge['type'],
                weight=edge['weight']
            )
    
    return G


def create_plotly_graph(graph_data: Dict[str, Any], layout_algorithm: str = 'spring') -> go.Figure:
    """
    Create an interactive Plotly graph visualization.
    
    Args:
        graph_data: Dictionary containing nodes and edges
        layout_algorithm: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        
    Returns:
        Plotly figure object
    """
    if not graph_data['nodes']:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No graph data available",
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
        )
        return fig
    
    # Create NetworkX graph for layout calculation
    G = create_networkx_graph(graph_data)
    
    # Calculate positions using NetworkX layout algorithms
    if layout_algorithm == 'circular':
        pos = nx.circular_layout(G)
    elif layout_algorithm == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:  # default to spring
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare node traces
    node_trace = go.Scatter(
        x=[], y=[], text=[], textposition='middle center',
        mode='markers+text', hoverinfo='text', marker=dict(size=[], color=[], line=dict(width=2))
    )
    
    # Color map for different node types
    color_map = {
        'Document': '#FF6B6B',
        'Chunk': '#4ECDC4',
        'Unknown': '#95A5A6'
    }
    
    for node in graph_data['nodes']:
        node_id = node['id']
        if node_id in pos:
            x, y = pos[node_id]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node['title'][:20],)  # Truncate long titles
            node_trace['marker']['size'] += (max(node['size'], 15),)
            node_trace['marker']['color'] += (color_map.get(node['label'], '#95A5A6'),)
    
    # Prepare edge traces
    edge_traces = []
    
    for edge in graph_data['edges']:
        source_id, target_id = edge['source'], edge['target']
        if source_id in pos and target_id in pos:
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]
            
            # Different line styles for different relationship types
            line_style = {
                'HAS_CHUNK': dict(color='rgba(125,125,125,0.5)', width=2),
                'SIMILAR_TO': dict(color='rgba(255,107,107,0.7)', width=1, dash='dash')
            }.get(edge['type'], dict(color='rgba(125,125,125,0.3)', width=1))
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=line_style,
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=[node_trace] + edge_traces)
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Knowledge Graph ({graph_data['total_nodes']} nodes, {graph_data['total_edges']} edges)",
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Documents (red) connected to Chunks (teal). Dashed lines show similarity relationships.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=10)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )
    
    return fig


def get_query_graph_data(query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create graph data for visualizing query results and their relationships.
    
    Args:
        query_results: List of retrieved chunks with metadata
        
    Returns:
        Dictionary containing nodes and edges for the query result graph
    """
    nodes = []
    edges = []
    
    # Add query node
    nodes.append({
        'id': 'query',
        'entity_id': 'query',
        'label': 'Query',
        'title': 'User Query',
        'size': 30
    })
    
    # Add result nodes and connections
    for i, result in enumerate(query_results):
        chunk_id = f"chunk_{i}"
        nodes.append({
            'id': chunk_id,
            'entity_id': result.get('chunk_id', f'unknown_{i}'),
            'label': 'Result',
            'title': result.get('content', 'No content')[:50] + "...",
            'size': 20 + result.get('similarity', 0) * 20  # Size based on similarity
        })
        
        # Connect query to result
        edges.append({
            'source': 'query',
            'target': chunk_id,
            'type': 'RETRIEVED',
            'weight': result.get('similarity', 0)
        })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'total_nodes': len(nodes),
        'total_edges': len(edges)
    }


def create_query_result_graph(query_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a graph visualization for query results.
    
    Args:
        query_results: List of retrieved chunks with metadata
        
    Returns:
        Plotly figure showing query relationships
    """
    graph_data = get_query_graph_data(query_results)
    
    if not graph_data['nodes'] or len(graph_data['nodes']) <= 1:
        # Return empty figure if no meaningful data
        fig = go.Figure()
        fig.update_layout(
            title="No query results to visualize",
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
        )
        return fig
    
    # Create NetworkX graph for layout
    G = create_networkx_graph(graph_data)
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create node trace
    node_trace = go.Scatter(
        x=[], y=[], text=[], textposition='middle center',
        mode='markers+text', hoverinfo='text', marker=dict(size=[], color=[])
    )
    
    # Color map for different node types
    color_map = {
        'Query': '#FF6B6B',
        'Result': '#4ECDC4'
    }
    
    for node in graph_data['nodes']:
        node_id = node['id']
        if node_id in pos:
            x, y = pos[node_id]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node['title'][:20],)
            node_trace['marker']['size'] += (node['size'],)
            node_trace['marker']['color'] += (color_map.get(node['label'], '#95A5A6'),)
    
    # Create edge traces
    edge_traces = []
    for edge in graph_data['edges']:
        source_id, target_id = edge['source'], edge['target']
        if source_id in pos and target_id in pos:
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]
            
            # Line width based on similarity/weight
            line_width = max(1, edge['weight'] * 5)
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(color=f'rgba(78,205,196,{edge["weight"]})', width=line_width),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=[node_trace] + edge_traces)
    
    fig.update_layout(
        title={
            'text': f"Query Results Graph ({len(query_results)} results)",
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=400
    )
    
    return fig