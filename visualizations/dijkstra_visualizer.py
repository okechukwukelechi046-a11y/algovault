"""
Interactive Dijkstra algorithm visualization using PyGame.
"""

import pygame
import sys
import math
from typing import List, Tuple, Dict, Optional
import heapq

# Colors
BACKGROUND = (20, 20, 30)
NODE_COLOR = (100, 150, 255)
NODE_SELECTED = (255, 100, 100)
NODE_VISITED = (100, 255, 100)
NODE_PATH = (255, 255, 100)
EDGE_COLOR = (150, 150, 150)
EDGE_VISITED = (100, 255, 100)
TEXT_COLOR = (255, 255, 255)
GRID_COLOR = (40, 40, 50)

class Node:
    """Graph node for visualization."""
    
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        self.radius = 20
        self.color = NODE_COLOR
        self.distance = float('inf')
        self.visited = False
        self.in_path = False
        
    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw node on screen."""
        # Draw node circle
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 
                          self.radius)
        
        # Draw node ID
        text = font.render(str(self.id), True, TEXT_COLOR)
        text_rect = text.get_rect(center=(self.x, self.y))
        screen.blit(text, text_rect)
        
        # Draw distance if not infinite
        if self.distance < float('inf'):
            dist_text = font.render(f"{self.distance:.1f}", True, TEXT_COLOR)
            dist_rect = dist_text.get_rect(center=(self.x, self.y + 30))
            screen.blit(dist_text, dist_rect)

class Edge:
    """Graph edge for visualization."""
    
    def __init__(self, node1: Node, node2: Node, weight: float):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.color = EDGE_COLOR
        self.visited = False
        
    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw edge on screen."""
        # Draw line
        pygame.draw.line(screen, self.color, 
                        (self.node1.x, self.node1.y),
                        (self.node2.x, self.node2.y), 3)
        
        # Draw weight
        mid_x = (self.node1.x + self.node2.x) / 2
        mid_y = (self.node1.y + self.node2.y) / 2
        
        # Offset weight text perpendicular to edge
        dx = self.node2.x - self.node1.x
        dy = self.node2.y - self.node1.y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            perp_x = -dy / length * 15
            perp_y = dx / length * 15
            
            text = font.render(f"{self.weight:.1f}", True, TEXT_COLOR)
            text_rect = text.get_rect(
                center=(mid_x + perp_x, mid_y + perp_y)
            )
            
            # Draw background for readability
            pygame.draw.rect(screen, BACKGROUND, text_rect.inflate(10, 5))
            screen.blit(text, text_rect)

class DijkstraVisualizer:
    """Main visualization controller."""
    
    def __init__(self, width: int = 1000, height: int = 700):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dijkstra Algorithm Visualizer")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        
        self.selected_node: Optional[Node] = None
        self.start_node: Optional[Node] = None
        self.end_node: Optional[Node] = None
        
        self.animation_speed = 1.0  # Steps per second
        self.is_animating = False
        self.animation_step = 0
        self.distances: List[float] = []
        self.previous: List[Optional[int]] = []
        self.visited_nodes: List[Node] = []
        self.current_edge: Optional[Edge] = None
        
        self.message = "Click to place nodes, then press S to start"
        self.stats = {}
        
        self._create_sample_graph()
        
    def _create_sample_graph(self) -> None:
        """Create a sample graph for demonstration."""
        # Clear existing graph
        self.nodes.clear()
        self.edges.clear()
        
        # Create nodes in a circle
        n_nodes = 7
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(self.width, self.height) // 3
        
        for i in range(n_nodes):
            angle = 2 * math.pi * i / n_nodes
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.nodes.append(Node(i, x, y))
        
        # Create edges
        edges = [
            (0, 1, 4), (0, 2, 2), (0, 6, 7),
            (1, 2, 1), (1, 3, 5), (1, 6, 3),
            (2, 3, 8), (2, 4, 10), (2, 6, 2),
            (3, 4, 2), (3, 5, 6),
            (4, 5, 3), (4, 6, 4),
            (5, 6, 5)
        ]
        
        for u, v, w in edges:
            self.edges.append(Edge(self.nodes[u], self.nodes[v], w))
        
        # Set start and end
        self.start_node = self.nodes[0]
        self.end_node = self.nodes[5]
        self.start_node.color = NODE_SELECTED
        self.end_node.color = NODE_SELECTED
        
    def run(self) -> None:
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.pos, event.button)
            
            if self.is_animating:
                self._update_animation()
            
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()
    
    def _handle_keydown(self, key: int) -> None:
        """Handle keyboard input."""
        if key == pygame.K_r:
            self._reset()
        elif key == pygame.K_s:
            self._start_algorithm()
        elif key == pygame.K_n:
            self._next_step()
        elif key == pygame.K_c:
            self._create_sample_graph()
            self._reset()
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.animation_speed = min(5.0, self.animation_speed + 0.5)
        elif key == pygame.K_MINUS:
            self.animation_speed = max(0.1, self.animation_speed - 0.5)
        elif key == pygame.K_ESCAPE:
            self.is_animating = False
    
    def _handle_mouse_click(self, pos: Tuple[int, int], button: int) -> None:
        """Handle mouse clicks."""
        if button == 1:  # Left click
            # Check if clicked on node
            for node in self.nodes:
                dx = node.x - pos[0]
                dy = node.y - pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance <= node.radius:
                    if not self.start_node:
                        self.start_node = node
                        node.color = NODE_SELECTED
                        self.message = f"Start node set to {node.id}. Select end node."
                    elif not self.end_node and node != self.start_node:
                        self.end_node = node
                        node.color = NODE_SELECTED
                        self.message = "Press S to start algorithm"
                    else:
                        self.selected_node = node
                    return
            
            # If not clicked on node, create new node
            new_id = len(self.nodes)
            self.nodes.append(Node(new_id, pos[0], pos[1]))
            self.message = f"Created node {new_id}"
        
        elif button == 3:  # Right click
            # Connect nodes if two are selected
            if self.selected_node:
                for node in self.nodes:
                    if node != self.selected_node:
                        dx = node.x - pos[0]
                        dy = node.y - pos[1]
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        if distance <= node.radius:
                            # Calculate distance as weight
                            dx = node.x - self.selected_node.x
                            dy = node.y - self.selected_node.y
                            weight = math.sqrt(dx*dx + dy*dy) / 50  # Scale for visualization
                            
                            self.edges.append(Edge(self.selected_node, node, weight))
                            self.message = f"Connected nodes {self.selected_node.id} and {node.id}"
                            self.selected_node = None
                            return
    
    def _start_algorithm(self) -> None:
        """Initialize Dijkstra algorithm."""
        if not self.start_node or not self.end_node:
            self.message = "Please select both start and end nodes"
            return
        
        # Reset all nodes
        for node in self.nodes:
            node.distance = float('inf')
            node.visited = False
            node.in_path = False
            if node not in [self.start_node, self.end_node]:
                node.color = NODE_COLOR
        
        for edge in self.edges:
            edge.visited = False
            edge.color = EDGE_COLOR
        
        # Initialize Dijkstra
        self.start_node.distance = 0
        self.distances = [float('inf')] * len(self.nodes)
        self.previous = [-1] * len(self.nodes)
        self.distances[self.start_node.id] = 0
        
        # Priority queue: (distance, node_id)
        self.pq = [(0, self.start_node.id)]
        self.visited_nodes = []
        self.current_edge = None
        
        self.is_animating = True
        self.animation_step = 0
        self.message = "Running Dijkstra algorithm..."
        
        # Statistics
        self.stats = {
            'nodes_visited': 0,
            'edges_relaxed': 0,
            'current_distance': 0,
            'queue_size': 1
        }
    
    def _update_animation(self) -> None:
        """Update animation by one step."""
        if not self.pq:
            self.is_animating = False
            self._reconstruct_path()
            return
        
        # Get next node from priority queue
        current_dist, current_id = heapq.heappop(self.pq)
        current_node = self.nodes[current_id]
        
        # Skip if already visited with shorter distance
        if current_dist > self.distances[current_id]:
            return
        
        # Mark as visited
        current_node.visited = True
        current_node.color = NODE_VISITED
        self.visited_nodes.append(current_node)
        self.stats['nodes_visited'] += 1
        
        # Check if we reached the end
        if current_id == self.end_node.id:
            self.is_animating = False
            self._reconstruct_path()
            return
        
        # Relax all outgoing edges
        for edge in self.edges:
            if edge.node1.id == current_id:
                neighbor = edge.node2
                new_dist = current_dist + edge.weight
                
                if new_dist < self.distances[neighbor.id]:
                    self.distances[neighbor.id] = new_dist
                    self.previous[neighbor.id] = current_id
                    heapq.heappush(self.pq, (new_dist, neighbor.id))
                    
                    # Visualize edge relaxation
                    edge.visited = True
                    edge.color = EDGE_VISITED
                    self.current_edge = edge
                    self.stats['edges_relaxed'] += 1
                    self.stats['queue_size'] = len(self.pq)
                    self.stats['current_distance'] = new_dist
                    
                    # Update neighbor distance display
                    neighbor.distance = new_dist
                    
                    return  # One step per frame
        
        # If no edges to relax, continue
        self.stats['queue_size'] = len(self.pq)
    
    def _reconstruct_path(self) -> None:
        """Reconstruct shortest path after algorithm completes."""
        if self.previous[self.end_node.id] == -1:
            self.message = "No path found!"
            return
        
        # Trace back from end to start
        path = []
        current = self.end_node.id
        
        while current != -1:
            self.nodes[current].in_path = True
            self.nodes[current].color = NODE_PATH
            path.append(current)
            current = self.previous[current]
        
        # Color path edges
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for edge in self.edges:
                if (edge.node1.id == u and edge.node2.id == v) or \
                   (edge.node1.id == v and edge.node2.id == u):
                    edge.color = NODE_PATH
        
        path_length = self.distances[self.end_node.id]
        self.message = f"Shortest path found! Length: {path_length:.2f}"
        
        self.stats['path_length'] = path_length
        self.stats['path_nodes'] = len(path)
    
    def _reset(self) -> None:
        """Reset visualization to initial state."""
        for node in self.nodes:
            node.distance = float('inf')
            node.visited = False
            node.in_path = False
            node.color = NODE_COLOR
        
        for edge in self.edges:
            edge.visited = False
            edge.color = EDGE_COLOR
        
        if self.start_node:
            self.start_node.color = NODE_SELECTED
        if self.end_node:
            self.end_node.color = NODE_SELECTED
        
        self.is_animating = False
        self.animation_step = 0
        self.message = "Click to place nodes, then press S to start"
        self.stats = {}
    
    def _draw(self) -> None:
        """Draw everything on screen."""
        # Clear screen
        self.screen.fill(BACKGROUND)
        
        # Draw grid
        grid_size = 50
        for x in range(0, self.width, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.width, y), 1)
        
        # Draw edges
        for edge in self.edges:
            edge.draw(self.screen, self.font)
        
        # Draw nodes
        for node in self.nodes:
            node.draw(self.screen, self.font)
        
        # Draw UI
        self._draw_ui()
    
    def _draw_ui(self) -> None:
        """Draw user interface."""
        # Draw control instructions
        controls = [
            "CONTROLS:",
            "Left Click: Place/Select Node",
            "Right Click: Connect Nodes",
            "S: Start Algorithm",
            "N: Next Step",
            "R: Reset",
            "C: Create Sample Graph",
            "+/-: Adjust Speed",
            "ESC: Stop Animation"
        ]
        
        for i, text in enumerate(controls):
            rendered = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(rendered, (10, 10 + i * 25))
        
        # Draw message
        message_text = self.large_font.render(self.message, True, TEXT_COLOR)
        self.screen.blit(message_text, (self.width // 2 - message_text.get_width() // 2, 10))
        
        # Draw statistics
        if self.stats:
            stats_y = self.height - 200
            
            stats_text = self.large_font.render("STATISTICS", True, TEXT_COLOR)
            self.screen.blit(stats_text, (self.width - 200, stats_y))
            
            for i, (key, value) in enumerate(self.stats.items()):
                text = f"{key.replace('_', ' ').title()}: {value}"
                rendered = self.font.render(text, True, TEXT_COLOR)
                self.screen.blit(rendered, (self.width - 200, stats_y + 30 + i * 25))
        
        # Draw animation speed
        speed_text = self.font.render(f"Speed: {self.animation_speed:.1f}x", True, TEXT_COLOR)
        self.screen.blit(speed_text, (self.width - 150, 10))
        
        # Draw legend
        legend = [
            ("Start/End", NODE_SELECTED),
            ("Visited", NODE_VISITED),
            ("Path", NODE_PATH),
            ("Normal", NODE_COLOR),
            ("Active Edge", EDGE_VISITED)
        ]
        
        for i, (label, color) in enumerate(legend):
            # Color box
            pygame.draw.rect(self.screen, color, (10, self.height - 150 + i * 30, 20, 20))
            
            # Label
            label_text = self.font.render(label, True, TEXT_COLOR)
            self.screen.blit(label_text, (40, self.height - 150 + i * 30))

if __name__ == "__main__":
    visualizer = DijkstraVisualizer()
    visualizer.run()
