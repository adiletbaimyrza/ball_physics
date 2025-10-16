import pygame
import sys
import math
import random
import pygame.gfxdraw

pygame.init()

# Window setup -----------------------------------------------------
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cannonball with Multiple Balls")

clock = pygame.time.Clock()

# Simulation setup -------------------------------------------------
SIM_MIN_WIDTH = 20.0
C_SCALE = min(WIDTH, HEIGHT) / SIM_MIN_WIDTH
SIM_WIDTH = WIDTH / C_SCALE
SIM_HEIGHT = HEIGHT / C_SCALE

def inv_cY(y): return (HEIGHT - y) / C_SCALE

# Physics parameters
GRAVITY = {'x': 0.0, 'y': -10.0}
TIME_STEP = 1.0 / 60.0

# Multiple balls ---------------------------------------------------
def make_ball():
    return {
        'radius': 1.2,
        'pos': {'x': random.uniform(0.2, 1.0), 'y': random.uniform(0.2, 2.0)},
        'vel': {'x': random.uniform(6.0, 12.0), 'y': random.uniform(10.0, 18.0)},
        'color': [random.randint(100, 255) for _ in range(3)]
    }

balls = [make_ball() for _ in range(3)]  # start with 3 balls
obstacles = []

# UI Buttons -------------------------------------------------------
font = pygame.font.SysFont(None, 24)
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 40

def draw_button(x, y, text, active):
    color = (0, 180, 0) if active else (180, 0, 0)
    pygame.draw.rect(screen, color, (x, y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=8)
    txt = font.render(text, True, (255, 255, 255))
    screen.blit(txt, (x + (BUTTON_WIDTH - txt.get_width()) // 2,
                      y + (BUTTON_HEIGHT - txt.get_height()) // 2))

start_button_rect = pygame.Rect(20, 20, BUTTON_WIDTH, BUTTON_HEIGHT)
pause_button_rect = pygame.Rect(140, 20, BUTTON_WIDTH, BUTTON_HEIGHT)
add_ball_button_rect = pygame.Rect(260, 20, BUTTON_WIDTH + 20, BUTTON_HEIGHT)

running_sim = False
drawing = False
draw_start = None

# Helper ------------------------------------------------------------
def reflect_velocity(vel, normal):
    dot = vel['x'] * normal[0] + vel['y'] * normal[1]
    vel['x'] -= 2 * dot * normal[0]
    vel['y'] -= 2 * dot * normal[1]

# Collision detection -----------------------------------------------
def check_collision(ball_pos, ball_radius, rect):
    rect_x = rect.x / C_SCALE
    rect_y = inv_cY(rect.y + rect.height)
    rect_w = rect.width / C_SCALE
    rect_h = rect.height / C_SCALE

    closest_x = max(rect_x, min(ball_pos['x'], rect_x + rect_w))
    closest_y = max(rect_y, min(ball_pos['y'], rect_y + rect_h))

    dx = ball_pos['x'] - closest_x
    dy = ball_pos['y'] - closest_y
    dist2 = dx * dx + dy * dy

    if dist2 <= ball_radius * ball_radius:
        dist = math.sqrt(dist2) if dist2 != 0 else 0.0001
        nx, ny = dx / dist, dy / dist
        return True, (nx, ny)
    return False, None

# Simulation --------------------------------------------------------
def simulate():
    for ball in balls:
        ball['vel']['x'] += GRAVITY['x'] * TIME_STEP
        ball['vel']['y'] += GRAVITY['y'] * TIME_STEP
        ball['pos']['x'] += ball['vel']['x'] * TIME_STEP
        ball['pos']['y'] += ball['vel']['y'] * TIME_STEP

        # Border collisions
        if ball['pos']['x'] < ball['radius']:
            ball['pos']['x'] = ball['radius']
            ball['vel']['x'] = -ball['vel']['x']
        if ball['pos']['x'] > SIM_WIDTH - ball['radius']:
            ball['pos']['x'] = SIM_WIDTH - ball['radius']
            ball['vel']['x'] = -ball['vel']['x']
        if ball['pos']['y'] < ball['radius']:
            ball['pos']['y'] = ball['radius']
            ball['vel']['y'] = -ball['vel']['y']

        # Obstacle collisions
        for rect in obstacles:
            collided, normal = check_collision(ball['pos'], ball['radius'], rect)
            if collided:
                reflect_velocity(ball['vel'], normal)
                ball['pos']['x'] += normal[0] * 0.02
                ball['pos']['y'] += normal[1] * 0.02

        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                b1, b2 = balls[i], balls[j]

                dx = b2['pos']['x'] - b1['pos']['x']
                dy = b2['pos']['y'] - b1['pos']['y']
                dist2 = dx * dx + dy * dy
                min_dist = b1['radius'] + b2['radius']

                if dist2 < min_dist * min_dist:
                    dist = math.sqrt(dist2) if dist2 != 0 else 0.0001
                    nx, ny = dx / dist, dy / dist

                    # Separate overlapping balls
                    overlap = 0.5 * (min_dist - dist)
                    b1['pos']['x'] -= nx * overlap
                    b1['pos']['y'] -= ny * overlap
                    b2['pos']['x'] += nx * overlap
                    b2['pos']['y'] += ny * overlap

                    # Relative velocity along the normal
                    dvx = b2['vel']['x'] - b1['vel']['x']
                    dvy = b2['vel']['y'] - b1['vel']['y']
                    vn = dvx * nx + dvy * ny

                    if vn < 0:  # only collide if moving toward each other
                        # Elastic collision (equal mass)
                        impulse = -2 * vn / 2
                        b1['vel']['x'] -= impulse * nx
                        b1['vel']['y'] -= impulse * ny
                        b2['vel']['x'] += impulse * nx
                        b2['vel']['y'] += impulse * ny

# Main loop --------------------------------------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_button_rect.collidepoint(event.pos):
                running_sim = True
            elif pause_button_rect.collidepoint(event.pos):
                running_sim = False
            elif add_ball_button_rect.collidepoint(event.pos):
                balls.append(make_ball())
            elif not running_sim:
                draw_start = event.pos
                drawing = True

        elif event.type == pygame.MOUSEBUTTONUP and drawing:
            draw_end = event.pos
            x1, y1 = draw_start
            x2, y2 = draw_end
            rect = pygame.Rect(min(x1, x2), min(y1, y2),
                               abs(x2 - x1), abs(y2 - y1))
            obstacles.append(rect)
            drawing = False

    if running_sim:
        simulate()

    # Drawing -------------------------------------------------------
    screen.fill((255, 255, 255))
    draw_button(start_button_rect.x, start_button_rect.y, "Start", running_sim)
    draw_button(pause_button_rect.x, pause_button_rect.y, "Pause", not running_sim)

    pygame.draw.rect(screen, (0, 0, 180), add_ball_button_rect, border_radius=8)
    txt = font.render("Add Ball", True, (255, 255, 255))
    screen.blit(txt, (add_ball_button_rect.x + (add_ball_button_rect.width - txt.get_width()) // 2,
                      add_ball_button_rect.y + (add_ball_button_rect.height - txt.get_height()) // 2))

    # Obstacles
    for rect in obstacles:
        pygame.draw.rect(screen, (0, 0, 0), rect)

    # While drawing
    if drawing:
        mx, my = pygame.mouse.get_pos()
        x1, y1 = draw_start
        temp_rect = pygame.Rect(min(x1, mx), min(y1, my), abs(mx - x1), abs(my - y1))
        pygame.draw.rect(screen, (150, 150, 150), temp_rect, 2)

    # Balls
    for ball in balls:
        px = ball['pos']['x'] * C_SCALE
        py = HEIGHT - ball['pos']['y'] * C_SCALE
        pr = round(ball['radius'] * C_SCALE)
        pygame.gfxdraw.filled_circle(screen, int(px), int(py), pr, ball['color'])
        pygame.gfxdraw.aacircle(screen, int(px), int(py), pr, ball['color'])

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
