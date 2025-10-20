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
TIME_STEP = 1.0 / 30.0
AIR_FRICTION = 0.98  # Damping coefficient (0-1, where 1 = no friction, 0.98 = 2% velocity loss per second)
WIND_STRENGTH = 3.0  # Wind force strength
current_wind = WindDirection.NONE

# NEW: Threshold for putting balls to sleep to prevent jitter
# We use the squared value to avoid costly square root operations
SLEEP_VELOCITY_SQ = 0.01 * 0.01

# Ball Constants (UPDATED with variable Radius) --------------------
# Mass range
BALL_MIN_MASS = 1.0
BALL_MAX_MASS = 10.0

# Radius range (linked to mass for visual size)
BALL_MIN_RADIUS = 0.3
BALL_MAX_RADIUS = 0.9

# Position and Velocity bounds
BALL_MIN_POS = {'x': 0.2, 'y': 0.2}
BALL_MAX_POS = {'x': 1.0, 'y': 2.0}
BALL_MIN_VEL = {'x': 6.0, 'y': 10.0}
BALL_MAX_VEL = {'x': 12.0, 'y': 18.0}

# Multiple balls ---------------------------------------------------
def make_ball():
    # 1. Generate Mass
    mass = random.uniform(BALL_MIN_MASS, BALL_MAX_MASS)

    # 2. Calculate Radius (using M proportional to R^3 normalized scaling)
    # This ensures R_min -> M_min and R_max -> M_max using volumetric relationship
    mass_range = BALL_MAX_MASS - BALL_MIN_MASS
    if mass_range > 0:
        # Normalized mass position (0 to 1)
        norm_m = (mass - BALL_MIN_MASS) / mass_range
        # Normalized radius position (R proportional to M^(1/3))
        norm_r = norm_m ** (1 / 3)

        radius_range = BALL_MAX_RADIUS - BALL_MIN_RADIUS
        radius = BALL_MIN_RADIUS + norm_r * radius_range
    else:
        # Handle case where min and max mass are the same
        radius = BALL_MIN_RADIUS

    # 3. Calculate safe spawning bounds, ensuring the ball's center is at least
    # one radius away from the simulation edges.

    # X-position bounds
    x_min_bound = max(BALL_MIN_POS['x'], radius)
    x_max_bound = min(BALL_MAX_POS['x'], SIM_WIDTH - radius)

    # Y-position bounds
    y_min_bound = max(BALL_MIN_POS['y'], radius)
    y_max_bound = min(BALL_MAX_POS['y'], SIM_HEIGHT - radius)

    safe_x_start = x_min_bound
    safe_x_end = max(x_min_bound, x_max_bound)

    safe_y_start = y_min_bound
    safe_y_end = max(y_min_bound, y_max_bound)

    # If the calculated range is invalid (e.g., SIM_WIDTH too small), ensure start <= end
    if safe_x_start > safe_x_end: safe_x_end = safe_x_start
    if safe_y_start > safe_y_end: safe_y_end = safe_y_start

    return {
        'radius': radius,  # Dynamic radius
        'mass': mass,  # Dynamic mass
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
        'mass': ball['mass'],
        'pos': {'x': ball['pos']['x'], 'y': ball['pos']['y']},
        'vel': {'x': ball['vel']['x'], 'y': ball['vel']['y']},
        'color': ball['color'].copy()
    }


balls_with_friction = [make_ball() for _ in range(3)]  # Left screen
balls_without_friction = [copy_ball(b) for b in balls_with_friction]  # Right screen (copies)

drawn_lines = []  # Store drawn pen strokes (shared across all screens)
PEN_WIDTH = 5  # Width of the pen stroke

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
drawing = False
current_stroke = []

# --- NEW: Ball holding state ---
held_ball_left = -1
held_ball_right = -1
last_ball_x = 0.0
last_ball_y = 0.0
last_update_time = 0.0

# Helper for Ball Grab ----------------------------------------
def check_ball_click(pos_x, pos_y, balls, screen_offset=0):
    """Checks if mouse is over a ball and returns the ball index if true."""
    for i, ball in enumerate(balls):
        # Convert mouse screen position to simulation coordinates
        sim_x = (pos_x - screen_offset) / C_SCALE
        # Correct Sim Y conversion (Y is inverted in Pygame)
        sim_y = (HEIGHT - pos_y) / C_SCALE

        dx = ball['pos']['x'] - sim_x
        dy = ball['pos']['y'] - sim_y
        dist = math.sqrt(dx * dx + dy * dy)

        # Check if distance is less than the ball radius + a small tolerance for clicking
        if dist < ball['radius'] + (2 / C_SCALE):
            return i
    return -1

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

def line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Check if line segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4) intersect"""
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return False, None, None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return True, ix, iy
    return False, None, None

def check_line_collision_swept(old_pos, new_pos, ball_radius, stroke, screen_offset=0):
    """Check collision between ball trajectory and pen stroke using swept collision"""
    if len(stroke) < 2:
        return False, None, None, None

    collision_happened = False
    best_t = 2.0  # Beyond the range [0,1]
    best_normal = None
    best_point = None

    for i in range(len(stroke) - 1):
        # stroke now contains normalized coordinates (0-1)
        rel_x1, rel_y1 = stroke[i]
        rel_x2, rel_y2 = stroke[i + 1]

        # Convert normalized coordinates to absolute screen coordinates
        x1 = rel_x1 * SCREEN_WIDTH + screen_offset
        y1 = rel_y1 * HEIGHT
        x2 = rel_x2 * SCREEN_WIDTH + screen_offset
        y2 = rel_y2 * HEIGHT

        # Convert to simulation coordinates
        sx1, sy1 = (x1 - screen_offset) / C_SCALE, (HEIGHT - y1) / C_SCALE
        sx2, sy2 = (x2 - screen_offset) / C_SCALE, (HEIGHT - y2) / C_SCALE

        # Check distance at start and end positions
        dist_old, (cx_old, cy_old) = point_to_segment_distance(
            old_pos['x'], old_pos['y'], sx1, sy1, sx2, sy2
        )
        dist_new, (cx_new, cy_new) = point_to_segment_distance(
            new_pos['x'], new_pos['y'], sx1, sy1, sx2, sy2
        )

        collision_dist = ball_radius + (PEN_WIDTH / 2) / C_SCALE

        # If we're penetrating at the new position, we need to handle it
        if dist_new < collision_dist:
            # Check if we crossed the line during this frame
            # Use ray-casting from old to new position
            intersects, ix, iy = line_segment_intersection(
                old_pos['x'], old_pos['y'], new_pos['x'], new_pos['y'],
                sx1, sy1, sx2, sy2
            )

            # Calculate when collision occurred (parametric t along movement path)
            dx_move = new_pos['x'] - old_pos['x']
            dy_move = new_pos['y'] - old_pos['y']
            move_len = math.sqrt(dx_move * dx_move + dy_move * dy_move)

            if move_len > 1e-6:
                # Find the closest point along the trajectory to the line segment
                # This handles both direct intersections and grazing collisions
                t_collision = 0.0

                if intersects:
                    # Direct intersection - calculate t
                    dx_to_intersection = ix - old_pos['x']
                    dy_to_intersection = iy - old_pos['y']
                    t_collision = math.sqrt(dx_to_intersection ** 2 + dy_to_intersection ** 2) / move_len
                else:
                    # Grazing collision - estimate t from distance changes
                    if dist_old > collision_dist:
                        # We were outside, now inside - approximate t
                        t_collision = (dist_old - collision_dist) / (dist_old - dist_new + 1e-6)
                    else:
                        t_collision = 0.0

                t_collision = max(0.0, min(1.0, t_collision))

                # Only update if this is the earliest collision
                if t_collision < best_t:
                    best_t = t_collision

                    # Calculate collision point on trajectory
                    collision_x = old_pos['x'] + t_collision * dx_move
                    collision_y = old_pos['y'] + t_collision * dy_move

                    # Find closest point on line segment at collision time
                    _, (cpx, cpy) = point_to_segment_distance(
                        collision_x, collision_y, sx1, sy1, sx2, sy2
                    )

                    # Calculate normal
                    dx = collision_x - cpx
                    dy = collision_y - cpy
                    d = math.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        best_normal = (dx / d, dy / d)
                        best_point = (cpx, cpy)
                        collision_happened = True

    return collision_happened, best_normal, best_point, best_t

# Simulation --------------------------------------------------------
def simulate_balls(balls, drawn_lines, apply_friction, screen_offset=0):
    # Get wind force (This is a force F, not an acceleration)
    wind_x, wind_y = current_wind.value
    wind_force_x = wind_x * WIND_STRENGTH
    wind_force_y = wind_y * WIND_STRENGTH

    for ball in balls:
        # Store old position for swept collision detection
        old_pos = {'x': ball['pos']['x'], 'y': ball['pos']['y']}

        # 1. Apply Forces (Gravity + Wind)
        ball['vel']['x'] += GRAVITY['x'] * TIME_STEP
        ball['vel']['y'] += GRAVITY['y'] * TIME_STEP

        # Wind: Wind is a force (F). Acceleration is F/m.
        mass = ball['mass']
        ball['vel']['x'] += (wind_force_x / mass) * TIME_STEP
        ball['vel']['y'] += (wind_force_y / mass) * TIME_STEP

        # 2. Apply Air Friction (Exponential Damping)
        if apply_friction:
            damping = AIR_FRICTION ** TIME_STEP
            ball['vel']['x'] *= damping
            ball['vel']['y'] *= damping

        # 3. Update position (Velocity Integration - Semi-Implicit Euler)
        ball['pos']['x'] += ball['vel']['x'] * TIME_STEP
        ball['pos']['y'] += ball['vel']['y'] * TIME_STEP

        # 4. Line collisions with swept collision detection
        for stroke in drawn_lines:
            collided, normal, collision_point, t = check_line_collision_swept(
                old_pos, ball['pos'], ball['radius'], stroke, screen_offset
            )
            if collided and normal is not None:
                # Move ball back to collision point
                ball['pos']['x'] = old_pos['x'] + t * (ball['pos']['x'] - old_pos['x'])
                ball['pos']['y'] = old_pos['y'] + t * (ball['pos']['y'] - old_pos['y'])

                # Push ball away from line to prevent sticking
                penetration_fix = ball['radius'] + (PEN_WIDTH / 2) / C_SCALE + 0.01
                ball['pos']['x'] = collision_point[0] + normal[0] * penetration_fix
                ball['pos']['y'] = collision_point[1] + normal[1] * penetration_fix

                # Reflect velocity
                reflect_velocity(ball['vel'], normal)

        # 5. Border collisions with coefficient of restitution (COR=0.8)
        COR_WALL = 0.8

        # Bottom edge
        if ball['pos']['y'] < ball['radius']:
            ball['pos']['y'] = ball['radius']
            ball['vel']['y'] = -ball['vel']['y'] * COR_WALL

        # Top edge
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

    # 6. Ball-to-ball collisions
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
                    # Calculate impulse for unequal masses
                    mass_sum_inv = 1.0 / b1['mass'] + 1.0 / b2['mass']
                    j_impulse = -(1 + COR_BALL) * vn / mass_sum_inv

                    # Apply impulse change to velocity
                    b1['vel']['x'] -= (j_impulse / b1['mass']) * nx
                    b1['vel']['y'] -= (j_impulse / b1['mass']) * ny
                    b2['vel']['x'] += (j_impulse / b2['mass']) * nx
                    b2['vel']['y'] += (j_impulse / b2['mass']) * ny

    # FINAL CHECK TO PREVENT JITTER
    # This loop must be separate, after all collisions are resolved
    for ball in balls:
        # Put balls to sleep if they are barely moving
        speed_sq = ball['vel']['x'] ** 2 + ball['vel']['y'] ** 2
        if speed_sq < SLEEP_VELOCITY_SQ:
            ball['vel']['x'] = 0.0
            ball['vel']['y'] = 0.0

def draw_balls(balls, screen_offset=0):
    """Draw balls with given screen offset - with fake 3D shading and holding highlight"""
    for i, ball in enumerate(balls):
        px = ball['pos']['x'] * C_SCALE + screen_offset
        py = HEIGHT - ball['pos']['y'] * C_SCALE
        pr = math.ceil(ball['radius'] * C_SCALE)

        # Draw shadow
        shadow_color = (80, 80, 80)
        pygame.gfxdraw.filled_circle(screen, int(px + 2), int(py + 2), pr, shadow_color)

        # Draw darker base
        dark_color = tuple(max(0, int(c * 0.5)) for c in ball['color'])
        pygame.gfxdraw.filled_circle(screen, int(px), int(py), pr, dark_color)

        # Draw main ball color
        pygame.gfxdraw.filled_circle(screen, int(px), int(py), pr, ball['color'])
        pygame.gfxdraw.aacircle(screen, int(px), int(py), pr, ball['color'])

        # Create radial gradient
        num_layers = max(5, pr // 2)
        for j in range(num_layers):
            factor = 1 - (j / num_layers)
            shade_color = tuple(max(0, int(c * (0.6 + 0.4 * factor))) for c in ball['color'])
            layer_radius = int(pr * factor)
            if layer_radius > 1:
                pygame.gfxdraw.filled_circle(screen, int(px), int(py), layer_radius, shade_color)

        # Add highlight
        highlight_radius = max(2, pr // 4)
        highlight_x = int(px - pr * 0.35)
        highlight_y = int(py - pr * 0.35)

        for j in range(3):
            alpha_factor = 1 - (j / 3)
            highlight_color = tuple(min(255, int(c + (255 - c) * alpha_factor * 0.5)) for c in ball['color'])
            h_radius = highlight_radius - j
            if h_radius > 0:
                pygame.gfxdraw.filled_circle(screen, highlight_x, highlight_y, h_radius, highlight_color)

        # Specular highlight
        spec_radius = max(1, pr // 8)
        spec_color = (255, 255, 255)
        pygame.gfxdraw.filled_circle(screen, int(px - pr * 0.3), int(py - pr * 0.3), spec_radius, spec_color)

        # Edge outline
        edge_color = tuple(max(0, c - 50) for c in ball['color'])
        pygame.gfxdraw.aacircle(screen, int(px), int(py), pr, edge_color)

        # Highlight held ball
        is_held = (screen_offset == 0 and i == held_ball_left) or \
                  (screen_offset == SCREEN_WIDTH and i == held_ball_right)

        if is_held:
            ring_color = (255, 255, 255)
            for r_offset in range(2, 5):
                pygame.gfxdraw.aacircle(screen, int(px), int(py), pr + r_offset, ring_color)

# Main loop --------------------------------------------------------
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    current_time = pygame.time.get_ticks() / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos

            is_left_screen = mouse_x < SCREEN_WIDTH
            ball_list = balls_with_friction if is_left_screen else balls_without_friction
            screen_x_offset = 0 if is_left_screen else SCREEN_WIDTH

            idx = check_ball_click(mouse_x, mouse_y, ball_list, screen_offset=screen_x_offset)

            if idx != -1:
                held_ball_left = idx if is_left_screen else -1
                held_ball_right = idx if not is_left_screen else -1
                running_sim = False

                sim_x = (mouse_x - screen_x_offset) / C_SCALE
                sim_y = (HEIGHT - mouse_y) / C_SCALE

                ball_list[idx]['pos']['x'] = sim_x
                ball_list[idx]['pos']['y'] = sim_y
                ball_list[idx]['vel']['x'] = 0.0
                ball_list[idx]['vel']['y'] = 0.0

                last_ball_x = sim_x
                last_ball_y = sim_y
                last_update_time = current_time

            elif held_ball_left == -1 and held_ball_right == -1:
                if start_button_rect.collidepoint(event.pos):
                    running_sim = True
                elif pause_button_rect.collidepoint(event.pos):
                    running_sim = False
                elif restart_button_rect.collidepoint(event.pos):
                    running_sim = False
                    balls_with_friction = [make_ball() for _ in range(3)]
                    balls_without_friction = [copy_ball(b) for b in balls_with_friction]
                    drawn_lines = []
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
                    # Start drawing - store relative position (normalized 0-1)
                    drawing = True
                    # Normalize x position to 0-1 range within one screen
                    if is_left_screen:
                        rel_x = mouse_x / SCREEN_WIDTH
                    else:
                        rel_x = (mouse_x - SCREEN_WIDTH) / SCREEN_WIDTH
                    rel_y = mouse_y / HEIGHT
                    current_stroke = [(rel_x, rel_y)]

        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos

            ball_list = None
            ball_idx = -1
            screen_x_offset = 0

            if held_ball_left != -1:
                ball_list = balls_with_friction
                ball_idx = held_ball_left
            elif held_ball_right != -1:
                ball_list = balls_without_friction
                ball_idx = held_ball_right
                screen_x_offset = SCREEN_WIDTH

            if ball_list is not None:
                dt = current_time - last_update_time

                sim_x = (mouse_x - screen_x_offset) / C_SCALE
                sim_y = (HEIGHT - mouse_y) / C_SCALE

                if dt > 0.001:
                    ball_list[ball_idx]['vel']['x'] = (sim_x - last_ball_x) / dt
                    ball_list[ball_idx]['vel']['y'] = (sim_y - last_ball_y) / dt

                ball_list[ball_idx]['pos']['x'] = sim_x
                ball_list[ball_idx]['pos']['y'] = sim_y

                last_ball_x = sim_x
                last_ball_y = sim_y
                last_update_time = current_time

            elif drawing:
                # Store normalized coordinates
                is_left_screen = mouse_x < SCREEN_WIDTH
                if is_left_screen:
                    rel_x = mouse_x / SCREEN_WIDTH
                else:
                    rel_x = (mouse_x - SCREEN_WIDTH) / SCREEN_WIDTH
                rel_y = mouse_y / HEIGHT
                current_stroke.append((rel_x, rel_y))

        elif event.type == pygame.MOUSEBUTTONUP:
            if held_ball_left != -1 or held_ball_right != -1:
                held_ball_left = -1
                held_ball_right = -1
                running_sim = True
            elif drawing:
                if len(current_stroke) > 1:
                    drawn_lines.append(current_stroke)
                current_stroke = []
                drawing = False

    if running_sim:
        if held_ball_left == -1 and held_ball_right == -1:
            simulate_balls(balls_with_friction, drawn_lines, apply_friction=True, screen_offset=0)
            simulate_balls(balls_without_friction, drawn_lines, apply_friction=False, screen_offset=SCREEN_WIDTH)

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
    for stroke in drawn_lines:
        if len(stroke) > 1:
            # Convert normalized coordinates to left screen absolute coordinates
            absolute_stroke_left = [(x * SCREEN_WIDTH, y * HEIGHT) for x, y in stroke]
            pygame.draw.lines(screen, (0, 0, 0), False, absolute_stroke_left, PEN_WIDTH)
    if drawing and len(current_stroke) > 1:
        # Draw current stroke being drawn
        mouse_x = mouse_pos[0]
        if mouse_x < SCREEN_WIDTH:  # Drawing on left screen
            absolute_current = [(x * SCREEN_WIDTH, y * HEIGHT) for x, y in current_stroke]
            pygame.draw.lines(screen, (100, 100, 100), False, absolute_current, PEN_WIDTH)

    # Draw right screen strokes
    for stroke in drawn_lines:
        if len(stroke) > 1:
            # Convert normalized coordinates to right screen absolute coordinates
            absolute_stroke_right = [(x * SCREEN_WIDTH + SCREEN_WIDTH, y * HEIGHT) for x, y in stroke]
            pygame.draw.lines(screen, (0, 0, 0), False, absolute_stroke_right, PEN_WIDTH)
    if drawing and len(current_stroke) > 1:
        # Draw current stroke being drawn
        mouse_x = mouse_pos[0]
        if mouse_x >= SCREEN_WIDTH:  # Drawing on right screen
            absolute_current = [(x * SCREEN_WIDTH + SCREEN_WIDTH, y * HEIGHT) for x, y in current_stroke]
            pygame.draw.lines(screen, (100, 100, 100), False, absolute_current, PEN_WIDTH)

    # Draw balls
    draw_balls(balls_with_friction, screen_offset=0)
    draw_balls(balls_without_friction, screen_offset=SCREEN_WIDTH)

    pygame.display.flip()
    clock.tick(60)


pygame.quit()
sys.exit()