# Obstacle Dodger (Console Version)

A fast-paced console arcade game where you dodge falling obstacles using ASCII graphics and colors.

## Features

- Colorful console-based interface
- Smooth animations using ANSI escape codes
- Progressive difficulty system
- Score tracking and saving
- 3-2-1 countdown timer
- Non-blocking keyboard input

## Requirements

```bash
pip install colorama numpy
```

## How to Play

1. Run the game:
```bash
python obstacle_dodger.py
```

2. Controls:
- Use `,` key to move left
- Use `.` key to move right
- Try to dodge the falling red blocks (■)
- Your score increases with each successful dodge

## Game Elements

```
Score: 42
           ─ ■ ─    # Obstacles falling from top
           ─ ─ ■    # ■ = Obstacle
           ■ ─ ─    # ─ = Empty space
           ─ ■ ─
           ■ ─ ─
             ^      # ^ = Your player
```

## Features in Detail

1. **Smooth Movement**
   - Optimized screen updates
   - No flickering animations
   - Responsive controls

2. **Progressive Difficulty**
   - Game starts at a comfortable pace
   - Speed gradually increases as you score higher
   - Tests your reflexes and concentration

3. **Score System**
   - Points awarded for each obstacle dodged
   - Scores saved to individual player files
   - Located in `scores/[player_name].txt`

4. **Game Flow**
   1. Welcome screen
   2. Player name entry
   3. 3-2-1 countdown
   4. Game starts
   5. Game over screen with final score

## Technical Implementation

- Uses ANSI escape codes for screen manipulation
- Colorama for cross-platform color support
- Non-blocking input handling with msvcrt
- Efficient frame-based obstacle movement
- Minimal screen clearing for smooth performance

## File Structure

```
Command_line_game/
├── obstacle dodger.py    # Main game file
├── README.md             # This file
└── scores/              # Score directory
    └── [player_name].txt # Individual score files
```


