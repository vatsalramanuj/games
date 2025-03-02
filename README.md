# Face Tracker Game

A fun and interactive game that uses your webcam to control a character using face movements. Collect coins, dodge obstacles, and use powerups to achieve the highest score possible!

## Features

- 🎮 Face-controlled gameplay
- 💰 Collectible coins system
- 🛡️ Multiple powerups:
  - Shield: Temporary invincibility
  - Slow: Reduces game speed
  - Magnet: Attracts nearby coins
- 🏪 Upgrade store system
- 💾 Progress saving
- 🎯 High score tracking
- ✨ Visual effects and UI enhancements

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Webcam

## Installation

1. Make sure you have Python installed on your system
2. Install the required packages:
```bash
pip install opencv-python numpy
```

## How to Play

1. Run the game:
```bash
python game.py
```

2. Controls:
   - Move your face left/right to control the green square
   - Press 'U' during gameplay to access the upgrade menu
   - Press 'SPACE' to start the game
   - Press 'Q' to quit

3. Gameplay:
   - Collect yellow coins to increase your score and currency
   - Avoid red obstacles
   - Collect powerups:
     - Orange: Shield (temporary invincibility)
     - Purple: Slow (reduces game speed)
     - Dark Purple: Magnet (attracts nearby coins)

4. Store/Upgrades:
   - Use collected coins to upgrade powerups
   - Each upgrade increases the duration of the powerup
   - Access the store from the main menu or during gameplay

## Game Mechanics

- The game speed increases as your score gets higher
- Powerups have limited duration but can be upgraded
- Coins spawn more frequently in the center of the screen
- Collision detection prevents objects from overlapping
- Progress is automatically saved after collecting coins or ending the game

## Tips

- Use the magnet powerup to collect coins more easily
- Upgrade the shield duration for better survival
- The slow powerup is useful when the game speed gets too fast
- Stay in the middle of the screen to collect more coins

## Files

- `game.py`: Main game file
- `game_data.json`: Saves your progress (coins, high score, powerup levels)

## Contributing

Feel free to fork this project and submit pull requests with improvements or new features! 