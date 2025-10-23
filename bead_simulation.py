import pygame
import math
from dataclasses import dataclass
from typing import List


@dataclass
class Vector2:
    """2D vector with common operations."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self):
        length = self.length()
        if length == 0:
            return Vector2(0, 0)
        return Vector2(self.x / length, self.y / length)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def copy(self):
        return Vector2(self.x, self.y)


class Bead:
    """Represents a bead constrained to move on a wire."""

    def __init__(self, radius: float, mass: float, position: Vector2):
        self.radius = radius
        self.mass = mass
        self.pos = position.copy()
        self.prev_pos = position.copy()
        self.vel = Vector2(0, 0)

    def integrate(self, dt: float, gravity: Vector2):
        """Perform velocity Verlet integration step."""
        self.vel = self.vel + gravity * dt
        self.prev_pos = self.pos.copy()
        self.pos = self.pos + self.vel * dt

    def constrain_to_wire(self, center: Vector2, radius: float):
        """Keep bead on circular wire constraint."""
        direction = self.pos - center
        distance = direction.length()

        if distance == 0:
            return

        direction = direction.normalized()
        correction = radius - distance
        self.pos = self.pos + direction * correction

    def update_velocity(self, dt: float):
        """Update velocity based on position change."""
        self.vel = (self.pos - self.prev_pos) * (1.0 / dt)


class CollisionHandler:
    """Handles collision detection and response between beads."""

    @staticmethod
    def handle_bead_collision(bead1: Bead, bead2: Bead, restitution: float = 1.0):
        """Resolve collision between two beads with elastic collision."""
        direction = bead2.pos - bead1.pos
        distance = direction.length()

        # Check if beads are colliding
        if distance == 0 or distance > bead1.radius + bead2.radius:
            return

        direction = direction.normalized()

        # Position correction
        overlap = bead1.radius + bead2.radius - distance
        correction = overlap / 2.0
        bead1.pos = bead1.pos - direction * correction
        bead2.pos = bead2.pos + direction * correction

        # Velocity update using elastic collision formula
        v1 = bead1.vel.dot(direction)
        v2 = bead2.vel.dot(direction)
        m1 = bead1.mass
        m2 = bead2.mass

        new_v1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * restitution) / (m1 + m2)
        new_v2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)

        bead1.vel = bead1.vel + direction * (new_v1 - v1)
        bead2.vel = bead2.vel + direction * (new_v2 - v2)


class PhysicsSimulation:
    """Manages the physics simulation."""

    def __init__(self, wire_center: Vector2, wire_radius: float):
        self.gravity = Vector2(0, -10.0)
        self.dt = 1.0 / 60.0
        self.substeps = 100
        self.wire_center = wire_center
        self.wire_radius = wire_radius
        self.beads: List[Bead] = []
        self.collision_handler = CollisionHandler()

    def add_bead(self, bead: Bead):
        """Add a bead to the simulation."""
        self.beads.append(bead)

    def simulate_step(self):
        """Perform one simulation step with substeps for stability."""
        sub_dt = self.dt / self.substeps

        for _ in range(self.substeps):
            # Integration
            for bead in self.beads:
                bead.integrate(sub_dt, self.gravity)

            # Apply constraints
            for bead in self.beads:
                bead.constrain_to_wire(self.wire_center, self.wire_radius)

            # Update velocities
            for bead in self.beads:
                bead.update_velocity(sub_dt)

            # Handle collisions
            for i in range(len(self.beads)):
                for j in range(i):
                    self.collision_handler.handle_bead_collision(
                        self.beads[i], self.beads[j]
                    )


class Renderer:
    """Handles all drawing operations."""

    def __init__(self, screen: pygame.Surface, scale: float):
        self.screen = screen
        self.scale = scale
        self.screen_height = screen.get_height()

    def world_to_screen(self, pos: Vector2) -> tuple:
        """Convert world coordinates to screen coordinates."""
        x = int(pos.x * self.scale)
        y = int(self.screen_height - pos.y * self.scale)
        return (x, y)

    def draw_wire(self, center: Vector2, radius: float, color=(255, 0, 0)):
        """Draw the circular wire constraint."""
        screen_pos = self.world_to_screen(center)
        screen_radius = int(radius * self.scale)
        pygame.draw.circle(self.screen, color, screen_pos, screen_radius, 2)

    def draw_bead(self, bead: Bead, color=(255, 0, 0)):
        """Draw a bead."""
        screen_pos = self.world_to_screen(bead.pos)
        screen_radius = int(bead.radius * self.scale)
        pygame.draw.circle(self.screen, color, screen_pos, screen_radius)

    def clear(self, color=(255, 255, 255)):
        """Clear the screen."""
        self.screen.fill(color)


class BeadSimulationApp:
    """Main application class."""

    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Constrained Dynamics - Beads on Wire")
        self.clock = pygame.time.Clock()

        # Calculate simulation space
        self.sim_min_width = 2.0
        self.scale = min(width, height) / self.sim_min_width
        self.sim_width = width / self.scale
        self.sim_height = height / self.scale

        self.renderer = Renderer(self.screen, self.scale)
        self.setup_simulation()
        self.running = True

    def setup_simulation(self):
        """Initialize the physics simulation with beads."""
        wire_center = Vector2(self.sim_width / 2, self.sim_height / 2)
        wire_radius = self.sim_min_width * 0.4

        self.simulation = PhysicsSimulation(wire_center, wire_radius)

        # Create beads arranged around the wire
        num_beads = 5
        angle = 0.0
        angle_step = math.pi / num_beads

        for i in range(num_beads):
            radius = 0.05 + (i % 3) * 0.03  # Varying sizes
            mass = math.pi * radius ** 2

            pos = Vector2(
                wire_center.x + wire_radius * math.cos(angle),
                wire_center.y + wire_radius * math.sin(angle)
            )

            self.simulation.add_bead(Bead(radius, mass, pos))
            angle += angle_step

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.setup_simulation()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def update(self):
        """Update simulation."""
        self.simulation.simulate_step()

    def render(self):
        """Render the scene."""
        self.renderer.clear((255, 255, 255))
        self.renderer.draw_wire(
            self.simulation.wire_center,
            self.simulation.wire_radius,
            (200, 200, 200)
        )

        for bead in self.simulation.beads:
            self.renderer.draw_bead(bead, (255, 0, 0))

        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    app = BeadSimulationApp()
    app.run()
