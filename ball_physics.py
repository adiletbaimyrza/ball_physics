import pygame
import sys
import math
import random
import pygame.gfxdraw
from enum import Enum

pygame.init()

# Window setup -----------------------------------------------------
WIDTH, HEIGHT = 1600, 600  # Double width for two screens
SCREEN_WIDTH = WIDTH // 2  # Each screen is half
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ball Physics Simulation - Left: With Friction | Right: Without Friction")

clock = pygame.time.Clock()

# Simulation setup -------------------------------------------------
SIM_MIN_WIDTH = 20.0
C_SCALE = min(SCREEN_WIDTH, HEIGHT) / SIM_MIN_WIDTH
SIM_WIDTH = SCREEN_WIDTH / C_SCALE
SIM_HEIGHT = HEIGHT / C_SCALE


def inv_cY(y): return (HEIGHT - y) / C_SCALE


# Wind Direction Enum ----------------------------------------------
class WindDirection(Enum):
    NONE = (0, 0)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)
    DOWN = (0, -1)


# Physics parameters -----------------------------------------------
GRAVITY = {'x': 0.0, 'y': -10.0}
TIME_STEP = 1.0 / 60.0
AIR_FRICTION = 0.98  # Damping coefficient (0-1, where 1 = no friction, 0.98 = 2% velocity loss per second)
WIND_STRENGTH = 3.0  # Wind force strength
current_wind = WindDirection.NONE

# Ball Constants (NEW) ---------------------------------------------
BALL_RADIUS = 0.5
BALL_MIN_POS = {'x': 0.2, 'y': 0.2}
BALL_MAX_POS = {'x': 1.0, 'y': 2.0}
BALL_MIN_VEL = {'x': 6.0, 'y': 10.0}
BALL_MAX_VEL = {'x': 12.0, 'y': 18.0}


# Multiple balls ---------------------------------------------------
def make_ball():
    # Calculate safe spawning bounds, ensuring the ball's center is at least
    # one radius away from the simulation edges (0 and SIM_WIDTH/HEIGHT).

    # X-position bounds
    x_min_bound = max(BALL_MIN_POS['x'], BALL_RADIUS)
    x_max_bound = min(BALL_MAX_POS['x'], SIM_WIDTH - BALL_RADIUS)

    # Y-position bounds
    y_min_bound = max(BALL_MIN_POS['y'], BALL_RADIUS)
    y_max_bound = min(BALL_MAX_POS['y'], SIM_HEIGHT - BALL_RADIUS)

    # Note: If x_min_bound > x_max_bound, random.uniform will error.
    # This design assumes the user-defined min/max positions are safe.
    # We will use x_min_bound if the calculated range is invalid to prevent crash.

    safe_x_start = x_min_bound
    safe_x_end = max(x_min_bound, x_max_bound)

    safe_y_start = y_min_bound
    safe_y_end = max(y_min_bound, y_max_bound)

    return {
        'radius': BALL_RADIUS,
        'pos': {
            'x': random.uniform(safe_x_start, safe_x_end),
            'y': random.uniform(safe_y_start, safe_y_end)
        },
        'vel': {
            'x': random.uniform(BALL_MIN_VEL['x'], BALL_MAX_VEL['x']),
            'y': random.uniform(BALL_MIN_VEL['y'], BALL_MAX_VEL['y'])
        },
        'color': [random.randint(100, 255) for _ in range(3)]
    }


def copy_ball(ball):
    """Create a copy of a ball"""
    return {
        'radius': ball['radius'],
        'pos': {'x': ball['pos']['x'], 'y': ball['pos']['y']},
        'vel': {'x': ball['vel']['x'], 'y': ball['vel']['y']},
        'color': ball['color'].copy()
    }


balls_with_friction = [make_ball() for _ in range(3)]  # Left screen
balls_without_friction = [copy_ball(b) for b in balls_with_friction]  # Right screen (copies)

drawn_lines_left = []  # Store drawn pen strokes for left screen
drawn_lines_right = []  # Store drawn pen strokes for right screen
PEN_WIDTH = 15  # Width of the pen stroke

# UI Buttons -------------------------------------------------------
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 18)
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 40
SMALL_BUTTON_WIDTH = 60


def draw_button(x, y, w, h, text, active, text_color=(255, 255, 255)):
    color = (0, 180, 0) if active else (180, 0, 0)
    pygame.draw.rect(screen, color, (x, y, w, h), border_radius=8)
    txt = font.render(text, True, text_color)
    screen.blit(txt, (x + (w - txt.get_width()) // 2,
                      y + (h - txt.get_height()) // 2))


def draw_wind_button(x, y, text, is_active):
    color = (50, 150, 200) if is_active else (120, 120, 120)
    pygame.draw.rect(screen, color, (x, y, SMALL_BUTTON_WIDTH, 30), border_radius=5)
    txt = small_font.render(text, True, (255, 255, 255))
    screen.blit(txt, (x + (SMALL_BUTTON_WIDTH - txt.get_width()) // 2,
                      y + (30 - txt.get_height()) // 2))


start_button_rect = pygame.Rect(20, 20, BUTTON_WIDTH, BUTTON_HEIGHT)
pause_button_rect = pygame.Rect(140, 20, BUTTON_WIDTH, BUTTON_HEIGHT)
add_ball_button_rect = pygame.Rect(260, 20, BUTTON_WIDTH + 20, BUTTON_HEIGHT)
restart_button_rect = pygame.Rect(400, 20, BUTTON_WIDTH, BUTTON_HEIGHT)

# Wind control buttons
wind_none_rect = pygame.Rect(20, 80, SMALL_BUTTON_WIDTH, 30)
wind_left_rect = pygame.Rect(20, 120, SMALL_BUTTON_WIDTH, 30)
wind_right_rect = pygame.Rect(90, 120, SMALL_BUTTON_WIDTH, 30)
wind_up_rect = pygame.Rect(90, 80, SMALL_BUTTON_WIDTH, 30)
wind_down_rect = pygame.Rect(90, 160, SMALL_BUTTON_WIDTH, 30)

running_sim = False
drawing_left = False
drawing_right = False
current_stroke_left = []
current_stroke_right = []


# Helper ------------------------------------------------------------
def reflect_velocity(vel, normal):
    # CORRECTION: Apply COR (e.g., 0.8) for better realism and energy loss
    COR = 0.8
    dot = vel['x'] * normal[0] + vel['y'] * normal[1]

    # Calculate reflected velocity
    # v_reflected = v_incoming - (1 + COR) * (v_incoming . n) * n
    impulse_magnitude = -(1 + COR) * dot

    vel['x'] += impulse_magnitude * normal[0]
    vel['y'] += impulse_magnitude * normal[1]


# Collision detection with line segments ---------------------------
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Returns the shortest distance from point (px, py) to line segment (x1,y1)-(x2,y2)"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2), (x1, y1)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2), (closest_x, closest_y)


def check_line_collision(ball_pos, ball_radius, stroke, screen_offset=0):
    """Check collision between ball and a pen stroke"""
    if len(stroke) < 2:
        return False, None

    for i in range(len(stroke) - 1):
        x1, y1 = stroke[i]
        x2, y2 = stroke[i + 1]

        # Convert to simulation coordinates (adjust for screen offset)
        sx1, sy1 = (x1 - screen_offset) / C_SCALE, inv_cY(y1)
        sx2, sy2 = (x2 - screen_offset) / C_SCALE, inv_cY(y2)

        dist, (cx, cy) = point_to_segment_distance(
            ball_pos['x'], ball_pos['y'], sx1, sy1, sx2, sy2
        )

        collision_dist = ball_radius + (PEN_WIDTH / 2) / C_SCALE

        if dist < collision_dist:
            # Calculate normal
            dx = ball_pos['x'] - cx
            dy = ball_pos['y'] - cy
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0:
                # Resolve penetration by moving the ball away from the line
                # Push back amount is the penetration depth
                penetration = collision_dist - dist

                # Normalize the direction vector to get the collision normal
                nx, ny = dx / d, dy / d

                # Apply separation (to avoid jitter/sticking)
                ball_pos['x'] += nx * penetration
                ball_pos['y'] += ny * penetration

                return True, (nx, ny)  # Return the normal vector

    return False, None


# Simulation --------------------------------------------------------
def simulate_balls(balls, drawn_lines, apply_friction, screen_offset=0):
    # Get wind force
    wind_x, wind_y = current_wind.value
    wind_force_x = wind_x * WIND_STRENGTH
    wind_force_y = wind_y * WIND_STRENGTH

    for ball in balls:
        # 1. Apply Forces (Gravity + Wind)
        ball['vel']['x'] += GRAVITY['x'] * TIME_STEP
        ball['vel']['y'] += GRAVITY['y'] * TIME_STEP

        ball['vel']['x'] += wind_force_x * TIME_STEP
        ball['vel']['y'] += wind_force_y * TIME_STEP

        # 2. Apply Air Friction (Exponential Damping)
        if apply_friction:
            damping = AIR_FRICTION ** TIME_STEP
            ball['vel']['x'] *= damping
            ball['vel']['y'] *= damping

        # 3. Update position (Velocity Integration - Semi-Implicit Euler)
        ball['pos']['x'] += ball['vel']['x'] * TIME_STEP
        ball['pos']['y'] += ball['vel']['y'] * TIME_STEP

        # 4. Border collisions with coefficient of restitution (COR=0.8)
        COR_WALL = 0.8

        # Bottom edge
        if ball['pos']['y'] < ball['radius']:
            ball['pos']['y'] = ball['radius']
            ball['vel']['y'] = -ball['vel']['y'] * COR_WALL

        # Top edge (not usually relevant with downward gravity, but for completeness)
        if ball['pos']['y'] > SIM_HEIGHT - ball['radius']:
            ball['pos']['y'] = SIM_HEIGHT - ball['radius']
            ball['vel']['y'] = -ball['vel']['y'] * COR_WALL

        # Left edge
        if ball['pos']['x'] < ball['radius']:
            ball['pos']['x'] = ball['radius']
            ball['vel']['x'] = -ball['vel']['x'] * COR_WALL

        # Right edge
        if ball['pos']['x'] > SIM_WIDTH - ball['radius']:
            ball['pos']['x'] = SIM_WIDTH - ball['radius']
            ball['vel']['x'] = -ball['vel']['x'] * COR_WALL

        # 5. Line collisions
        for stroke in drawn_lines:
            # check_line_collision now also handles separation
            collided, normal = check_line_collision(ball['pos'], ball['radius'], stroke, screen_offset)
            if collided:
                reflect_velocity(ball['vel'], normal)  # reflect_velocity now applies COR

    # 6. Ball-to-ball collisions (still assuming equal mass, but added COR)
    COR_BALL = 0.95

    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            b1, b2 = balls[i], balls[j]

            dx = b2['pos']['x'] - b1['pos']['x']
            dy = b2['pos']['y'] - b1['pos']['y']
            dist2 = dx * dx + dy * dy
            min_dist = b1['radius'] + b2['radius']

            if dist2 < min_dist * min_dist:
                dist = math.sqrt(dist2) if dist2 > 0 else 0.0001
                nx, ny = dx / dist, dy / dist

                # Separate overlapping balls
                overlap = (min_dist - dist)
                b1['pos']['x'] -= nx * overlap * 0.5
                b1['pos']['y'] -= ny * overlap * 0.5
                b2['pos']['x'] += nx * overlap * 0.5
                b2['pos']['y'] += ny * overlap * 0.5

                # Relative velocity along the normal
                dvx = b2['vel']['x'] - b1['vel']['x']
                dvy = b2['vel']['y'] - b1['vel']['y']
                vn = dvx * nx + dvy * ny

                if vn < 0:  # only collide if moving toward each other
                    # Calculate impulse (assumes equal mass, applies COR)
                    j_impulse = -(1 + COR_BALL) * vn / 2

                    b1['vel']['x'] -= j_impulse * nx
                    b1['vel']['y'] -= j_impulse * ny
                    b2['vel']['x'] += j_impulse * nx
                    b2['vel']['y'] += j_impulse * ny


def draw_balls(balls, screen_offset=0):
    """Draw balls with given screen offset - with fake 3D shading"""
    for ball in balls:
        px = ball['pos']['x'] * C_SCALE + screen_offset
        py = HEIGHT - ball['pos']['y'] * C_SCALE
        pr = math.ceil(ball['radius'] * C_SCALE)

        # Draw shadow (offset down and right)
        shadow_color = (80, 80, 80)
        pygame.gfxdraw.filled_circle(screen, int(px + 2), int(py + 2), pr, shadow_color)

        # Draw darker base (bottom shading)
        dark_color = tuple(max(0, int(c * 0.5)) for c in ball['color'])
        pygame.gfxdraw.filled_circle(screen, int(px), int(py), pr, dark_color)

        # Draw main ball color
        pygame.gfxdraw.filled_circle(screen, int(px), int(py), pr, ball['color'])
        pygame.gfxdraw.aacircle(screen, int(px), int(py), pr, ball['color'])

        # Create radial gradient from center - lighter in middle, darker at edges
        num_layers = max(5, pr // 2)
        for i in range(num_layers):
            factor = 1 - (i / num_layers)  # 1 at center, 0 at edge
            # Darken towards the edges
            shade_color = tuple(max(0, int(c * (0.6 + 0.4 * factor))) for c in ball['color'])
            layer_radius = int(pr * factor)
            if layer_radius > 1:
                pygame.gfxdraw.filled_circle(screen, int(px), int(py), layer_radius, shade_color)

        # Add subtle highlight (top-left) for glossy sphere effect
        highlight_radius = max(2, pr // 4)
        highlight_x = int(px - pr * 0.35)
        highlight_y = int(py - pr * 0.35)

        # Soft highlight with gradient - no white, just lighter version of ball color
        for i in range(3):
            alpha_factor = 1 - (i / 3)
            highlight_color = tuple(min(255, int(c + (255 - c) * alpha_factor * 0.5)) for c in ball['color'])
            h_radius = highlight_radius - i
            if h_radius > 0:
                pygame.gfxdraw.filled_circle(screen, highlight_x, highlight_y, h_radius, highlight_color)

        # Very small, subtle specular highlight at center
        spec_radius = max(1, pr // 8)
        spec_color = (255, 255, 255)
        pygame.gfxdraw.filled_circle(screen, int(px - pr * 0.3), int(py - pr * 0.3), spec_radius, spec_color)

        # Edge outline for definition
        edge_color = tuple(max(0, c - 50) for c in ball['color'])
        pygame.gfxdraw.aacircle(screen, int(px), int(py), pr, edge_color)


# Main loop --------------------------------------------------------
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_button_rect.collidepoint(event.pos):
                running_sim = True
            elif pause_button_rect.collidepoint(event.pos):
                running_sim = False
            elif restart_button_rect.collidepoint(event.pos):
                # Restart: clear everything and create new balls
                running_sim = False
                balls_with_friction = [make_ball() for _ in range(3)]
                balls_without_friction = [copy_ball(b) for b in balls_with_friction]
                drawn_lines_left = []
                drawn_lines_right = []
                current_wind = WindDirection.NONE
            elif add_ball_button_rect.collidepoint(event.pos):
                new_ball = make_ball()
                balls_with_friction.append(new_ball)
                balls_without_friction.append(copy_ball(new_ball))
            elif wind_none_rect.collidepoint(event.pos):
                current_wind = WindDirection.NONE
            elif wind_left_rect.collidepoint(event.pos):
                current_wind = WindDirection.LEFT
            elif wind_right_rect.collidepoint(event.pos):
                current_wind = WindDirection.RIGHT
            elif wind_up_rect.collidepoint(event.pos):
                current_wind = WindDirection.UP
            elif wind_down_rect.collidepoint(event.pos):
                current_wind = WindDirection.DOWN
            elif not running_sim:
                # Start drawing on left or right screen
                if event.pos[0] < SCREEN_WIDTH:
                    drawing_left = True
                    current_stroke_left = [event.pos]
                else:
                    drawing_right = True
                    current_stroke_right = [event.pos]

        elif event.type == pygame.MOUSEMOTION:
            if drawing_left:
                current_stroke_left.append(event.pos)
            elif drawing_right:
                current_stroke_right.append(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if drawing_left:
                if len(current_stroke_left) > 1:
                    drawn_lines_left.append(current_stroke_left)
                current_stroke_left = []
                drawing_left = False
            elif drawing_right:
                if len(current_stroke_right) > 1:
                    drawn_lines_right.append(current_stroke_right)
                current_stroke_right = []
                drawing_right = False

    if running_sim:
        simulate_balls(balls_with_friction, drawn_lines_left, apply_friction=True, screen_offset=0)
        simulate_balls(balls_without_friction, drawn_lines_right, apply_friction=False, screen_offset=SCREEN_WIDTH)

    # Drawing -------------------------------------------------------
    screen.fill((255, 255, 255))

    # Draw divider line
    pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH, 0), (SCREEN_WIDTH, HEIGHT), 3)

    # Labels
    label_left = font.render("WITH AIR FRICTION", True, (0, 100, 0))
    label_right = font.render("WITHOUT AIR FRICTION", True, (100, 0, 0))
    screen.blit(label_left, (SCREEN_WIDTH // 2 - label_left.get_width() // 2, HEIGHT - 30))
    screen.blit(label_right, (SCREEN_WIDTH + SCREEN_WIDTH // 2 - label_right.get_width() // 2, HEIGHT - 30))

    # Main control buttons
    draw_button(start_button_rect.x, start_button_rect.y, BUTTON_WIDTH, BUTTON_HEIGHT,
                "Start", running_sim)
    draw_button(pause_button_rect.x, pause_button_rect.y, BUTTON_WIDTH, BUTTON_HEIGHT,
                "Pause", not running_sim)
    draw_button(add_ball_button_rect.x, add_ball_button_rect.y, BUTTON_WIDTH + 20, BUTTON_HEIGHT,
                "Add Ball", False)
    draw_button(restart_button_rect.x, restart_button_rect.y, BUTTON_WIDTH, BUTTON_HEIGHT,
                "Restart", False)

    # Wind control label
    wind_label = small_font.render("Wind:", True, (0, 0, 0))
    screen.blit(wind_label, (160, 85))

    # Wind direction buttons
    draw_wind_button(wind_none_rect.x, wind_none_rect.y, "None", current_wind == WindDirection.NONE)
    draw_wind_button(wind_left_rect.x, wind_left_rect.y, "Left", current_wind == WindDirection.LEFT)
    draw_wind_button(wind_right_rect.x, wind_right_rect.y, "Right", current_wind == WindDirection.RIGHT)
    draw_wind_button(wind_up_rect.x, wind_up_rect.y, "Up", current_wind == WindDirection.UP)
    draw_wind_button(wind_down_rect.x, wind_down_rect.y, "Down", current_wind == WindDirection.DOWN)

    # Draw left screen strokes
    for stroke in drawn_lines_left:
        if len(stroke) > 1:
            pygame.draw.lines(screen, (0, 0, 0), False, stroke, PEN_WIDTH)
    if drawing_left and len(current_stroke_left) > 1:
        pygame.draw.lines(screen, (100, 100, 100), False, current_stroke_left, PEN_WIDTH)

    # Draw right screen strokes
    for stroke in drawn_lines_right:
        if len(stroke) > 1:
            pygame.draw.lines(screen, (0, 0, 0), False, stroke, PEN_WIDTH)
    if drawing_right and len(current_stroke_right) > 1:
        pygame.draw.lines(screen, (100, 100, 100), False, current_stroke_right, PEN_WIDTH)

    # Draw balls
    draw_balls(balls_with_friction, screen_offset=0)
    draw_balls(balls_without_friction, screen_offset=SCREEN_WIDTH)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
