import os
import time
import random
import msvcrt
import numpy as np
from colorama import init, Fore, Back, Style

# Initialize colorama for colored output and force it to work on Windows
init(convert=True)

# ANSI escape codes for cursor movement and screen control
MOVE_TO_TOP = '\x1b[H'
HIDE_CURSOR = '\x1b[?25l'
SHOW_CURSOR = '\x1b[?25h'
CLEAR_SCREEN = '\x1b[2J'
CLEAR_LINE = '\x1b[2K'

POSITIONS = ['* _ _','_ * _', '_ _ *']
INITIAL_FRAMES = 5
Dead = False
frames = []
player = 0  # center column
player_positions = [f'{Fore.GREEN}^{Style.RESET_ALL}    ', f'{Fore.GREEN}  ^  {Style.RESET_ALL}', f'{Fore.GREEN}    ^{Style.RESET_ALL}']
gradient_steps = 50
timeout = np.linspace(1,0.5, gradient_steps)

# Replace stars with colored blocks for better visibility
POSITIONS = [pos.replace('*', f'{Fore.RED}■{Style.RESET_ALL}').replace('_', '─') for pos in POSITIONS]

score = 0 
for i in range(INITIAL_FRAMES):
    frames.append(random.choice(POSITIONS))

# Welcome screen with minimal design
welcome_screen = f'''
{Fore.YELLOW}OBSTACLE DODGER{Style.RESET_ALL}

Instructions:
► Dodge the {Fore.RED}■{Style.RESET_ALL} to stay alive
► Controls: {Fore.GREEN},{Style.RESET_ALL} for LEFT, {Fore.GREEN}.{Style.RESET_ALL} for RIGHT

Press Enter to start...
'''

print(CLEAR_SCREEN + HIDE_CURSOR + welcome_screen)
input()

print(CLEAR_SCREEN + "Enter your name: ")
player_name = input(f'{Fore.GREEN}> {Style.RESET_ALL}')

print(f"\n{Fore.YELLOW}Starting in...{Style.RESET_ALL}")
for i in range(3):
    print(f"{Fore.GREEN}{3-i}{Style.RESET_ALL}")
    time.sleep(1)

def get_user_input(timeout):
    start_time = time.time()
    input_char = ''
    while True:
        if msvcrt.kbhit():
            input_char = msvcrt.getch().decode()
            break
        elif time.time() - start_time > timeout:
            break
    return input_char

def create_game_frame(frames, player_pos, current_score):
    frame_parts = []
    
    # Add score
    frame_parts.append(f"Score: {Fore.YELLOW}{current_score}{Style.RESET_ALL}")
    
    # Add obstacle frames
    for k in range(INITIAL_FRAMES):
        frame_parts.append(f"           {frames[INITIAL_FRAMES - 1 - k]}")
    
    # Add player position
    frame_parts.append(f"           {player_pos}")
    
    return "\n".join(frame_parts)

# Clear screen once at start and create initial empty space
print(CLEAR_SCREEN)
print("\n" * (INITIAL_FRAMES + 2))

# Game loop
while not Dead:
    new_obs = random.randint(0, 2) - 1
    frames.append(POSITIONS[new_obs + 1])
    frames.pop(0)
    obstacle_index = POSITIONS.index(frames[0])-1
    
    # Move cursor to top and print new frame
    print(MOVE_TO_TOP + create_game_frame(frames, player_positions[player + 1], score), end='', flush=True)

    inp = get_user_input(timeout[min(score,gradient_steps-1)])
    if inp == ',' and player >= 0:
        player -= 1
    elif inp == '.' and player <= 0:
        player += 1

    if obstacle_index == player:
        Dead = True
        break
    
    score += 1

if Dead:
    game_over_screen = f'''
{Fore.RED}GAME OVER{Style.RESET_ALL}
{Fore.YELLOW}Final Score: {score}{Style.RESET_ALL}
'''
    print(SHOW_CURSOR + game_over_screen)

# Saving Scores in .txt files
file = r"C://Users//LENOVO//OneDrive//Desktop//projects//obstacle_dodger//scores//" + player_name + '.txt'
with open(file, "a+") as f:
    f.write(f"{score}\n")
