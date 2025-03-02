import cv2
import numpy as np
import random
import json
import os

# Load or create game data
def load_game_data():
    default_data = {
        'total_coins': 0,
        'high_score': 0,
        'powerup_levels': {
            'shield': 1,
            'speed': 1,
            'magnet': 1
        }
    }
    try:
        if os.path.exists('game_data.json'):
            with open('game_data.json', 'r') as f:
                return json.load(f)
        return default_data
    except:
        return default_data

def save_game_data(data):
    with open('game_data.json', 'w') as f:
        json.dump(data, f)

# Load saved game data
game_data = load_game_data()
total_coins = game_data['total_coins']
high_score = game_data['high_score']
powerup_levels = game_data['powerup_levels']
if 'speed' in powerup_levels:  # Convert old 'speed' powerup to 'slow'
    powerup_levels['slow'] = powerup_levels.pop('speed')

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize game parameters
frame_width = 640
frame_height = 480
player_width = 50
player_height = 50
player_x = frame_width // 2 - player_width // 2
player_y = frame_height - player_height - 10
obstacle_width = 50
obstacle_height = 50
coin_size = 30
base_speed = 8  # Renamed from obstacle_speed
speed_increase_rate = 0.1  # Speed increase per 10 points
obstacle_frequency = 20
coin_frequency = 60  # Decreased from 100 to make coins appear more often
powerup_frequency = 250  # Increased from 200 to make powerups slightly rarer

# Game economy and upgrades
upgrade_costs = {
    'shield': 10,
    'slow': 15,  # Changed from speed to slow
    'magnet': 20
}

# Define powerup types and their effects
POWERUP_TYPES = {
    'shield': {'color': (255, 165, 0), 'duration': 5},  # Orange
    'slow': {'color': (255, 0, 255), 'duration': 3},    # Purple - now slows down
    'magnet': {'color': (128, 0, 128), 'duration': 4}   # Dark purple
}

# Define the central region for coin spawning
center_x_range = (frame_width // 4, 3 * frame_width // 4 - coin_size)

# Create lists to store game objects
obstacles = []
coins = []
powerups = []
active_powerups = {}

# Initialize score and game state
score = 0
shield_active = False
slow_active = False  # Changed from speed_multiplier
magnet_active = False

def check_object_collision(x1, y1, width1, height1, x2, y2, width2, height2):
    """Check if two rectangular objects overlap"""
    return (x1 < x2 + width2 and
            x1 + width1 > x2 and
            y1 < y2 + height2 and
            y1 + height1 > y2)

def is_position_safe(x, y, width, height, obstacles, coins, powerups):
    """Check if a position is safe to spawn a new object"""
    # Check collision with obstacles
    for obstacle in obstacles:
        if check_object_collision(x, y, width, height,
                                obstacle[0], obstacle[1], obstacle_width, obstacle_height):
            return False
            
    # Check collision with coins
    for coin in coins:
        if check_object_collision(x, y, width, height,
                                coin[0], coin[1], coin_size, coin_size):
            return False
            
    # Check collision with powerups
    for powerup in powerups:
        if check_object_collision(x, y, width, height,
                                powerup[0], powerup[1], coin_size, coin_size):
            return False
            
    return True

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Show instructions with enhanced UI
instructions = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
# Add a gradient background
for y in range(frame_height):
    color = (int(40 * (y/frame_height)), int(20 * (y/frame_height)), int(60 * (y/frame_height)))
    cv2.line(instructions, (0, y), (frame_width, y), color)

cv2.putText(instructions, "Face Detection Game Instructions:", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(instructions, "1. Move your face left and right to control the green square", (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(instructions, "2. Collect coins and powerups:", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
# Draw example powerups
cv2.circle(instructions, (80, 190), 15, (0, 255, 255), -1)  # Gold coin
cv2.putText(instructions, "- Gold coins: +1 point", (100, 195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.circle(instructions, (80, 220), 15, (255, 165, 0), -1)  # Shield powerup
cv2.putText(instructions, "- Shield powerup: Temporary invincibility", (100, 225),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.circle(instructions, (80, 250), 15, (255, 0, 255), -1)  # Speed powerup
cv2.putText(instructions, "- Speed powerup: Move faster", (100, 255),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.circle(instructions, (80, 280), 15, (128, 0, 128), -1)  # Magnet powerup
cv2.putText(instructions, "- Magnet powerup: Attract nearby coins", (100, 285),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(instructions, "3. Avoid red obstacles", (50, 315),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(instructions, "4. Press 'SPACE' to start", (50, 365),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(instructions, "5. Press 'Q' to quit", (50, 415),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def show_main_menu():
    menu = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    # Add a gradient background
    for y in range(frame_height):
        color = (int(40 * (y/frame_height)), int(20 * (y/frame_height)), int(60 * (y/frame_height)))
        cv2.line(menu, (0, y), (frame_width, y), color)

    cv2.putText(menu, "Face Detection Game", (frame_width//2 - 150, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.putText(menu, f"Total Coins: {total_coins}", (frame_width//2 - 100, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(menu, f"High Score: {high_score}", (frame_width//2 - 100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Menu options
    cv2.putText(menu, "1. Start Game", (frame_width//2 - 100, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(menu, "2. Store", (frame_width//2 - 100, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(menu, "Q. Quit", (frame_width//2 - 100, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return menu

def show_store_menu():
    menu = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    # Add a gradient background
    for y in range(frame_height):
        color = (int(40 * (y/frame_height)), int(20 * (y/frame_height)), int(60 * (y/frame_height)))
        cv2.line(menu, (0, y), (frame_width, y), color)

    cv2.putText(menu, "Store", (frame_width//2 - 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(menu, f"Your Coins: {total_coins}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    y_offset = 150
    for powerup, level in powerup_levels.items():
        cost = upgrade_costs[powerup] * level
        duration = POWERUP_TYPES[powerup]['duration'] + level - 1
        cv2.putText(menu, f"{powerup.title()} (Level {level})", (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, POWERUP_TYPES[powerup]['color'], 2)
        cv2.putText(menu, f"Duration: {duration}s | Upgrade Cost: {cost} coins", (50, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 80

    cv2.putText(menu, "1-3. Upgrade (1:Shield, 2:Slow, 3:Magnet)", (50, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(menu, "SPACE. Return to Main Menu", (50, y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return menu

def try_upgrade_powerup(powerup_index):
    global total_coins
    powerups = ['shield', 'slow', 'magnet']
    if powerup_index < len(powerups):
        powerup = powerups[powerup_index]
        cost = upgrade_costs[powerup] * powerup_levels[powerup]
        if total_coins >= cost:
            total_coins -= cost
            powerup_levels[powerup] += 1
            # Save game data after upgrade
            save_game_data({
                'total_coins': total_coins,
                'high_score': high_score,
                'powerup_levels': powerup_levels
            })
            return True
    return False

def run_game():
    global total_coins, high_score
    score = 0
    shield_active = False
    slow_active = False  # Changed from speed_multiplier
    magnet_active = False
    
    # Create lists to store game objects
    obstacles = []
    coins = []
    powerups = []
    active_powerups = {}
    
    # Start the webcam feed
    cap = cv2.VideoCapture(0)

    # Main game loop
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the player's face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            player_x = x + w // 2 - player_width // 2
        
        # Add new obstacles at random intervals
        if random.randint(0, obstacle_frequency) == 0:
            max_attempts = 10  # Maximum attempts to find a safe position
            for _ in range(max_attempts):
                obstacle_x = random.randint(0, frame_width - obstacle_width)
                if is_position_safe(obstacle_x, 0, obstacle_width, obstacle_height,
                                  obstacles, coins, powerups):
                    obstacles.append([obstacle_x, 0])
                    break
        
        # Add new coins at random intervals
        if random.randint(0, coin_frequency) == 0:
            max_attempts = 10
            for _ in range(max_attempts):
                coin_x = random.randint(*center_x_range)
                if is_position_safe(coin_x, 0, coin_size, coin_size,
                                  obstacles, coins, powerups):
                    coins.append([coin_x, 0, 1])  # All coins worth 1 point
                    break
            
        # Add new powerups at random intervals
        if random.randint(0, powerup_frequency) == 0:
            max_attempts = 10
            for _ in range(max_attempts):
                powerup_x = random.randint(*center_x_range)
                if is_position_safe(powerup_x, 0, coin_size, coin_size,
                                  obstacles, coins, powerups):
                    powerup_type = random.choice(list(POWERUP_TYPES.keys()))
                    powerups.append([powerup_x, 0, powerup_type])
                    break
        
        # Calculate current game speed based on score
        current_speed = base_speed * (1 + (score // 10) * speed_increase_rate)
        if slow_active:
            current_speed *= 0.5  # Slow down to half speed when slow powerup is active
        
        # Move game objects
        for obstacle in obstacles:
            obstacle[1] += current_speed
        for coin in coins:
            coin[1] += current_speed
        for powerup in powerups:
            powerup[1] += current_speed
            
        # Apply magnet effect
        if magnet_active:
            for coin in coins:
                if abs(coin[0] - player_x) < 150:  # Magnet range
                    coin[0] += (player_x - coin[0]) * 0.1
        
        # Remove off-screen objects
        obstacles = [obs for obs in obstacles if obs[1] < frame_height]
        coins = [coin for coin in coins if coin[1] < frame_height]
        powerups = [powerup for powerup in powerups if powerup[1] < frame_height]
        
        # Check for collisions with obstacles
        if not shield_active:
            for obstacle in obstacles:
                if (obstacle[0] < player_x < obstacle[0] + obstacle_width or
                    obstacle[0] < player_x + player_width < obstacle[0] + obstacle_width):
                    if (obstacle[1] < player_y < obstacle[1] + obstacle_height or
                        obstacle[1] < player_y + player_height < obstacle[1] + obstacle_height):
                        print(f"Game Over! Your Score: {score}")
                        if score > high_score:
                            high_score = score
                        # Save game data when game ends
                        save_game_data({
                            'total_coins': total_coins,
                            'high_score': high_score,
                            'powerup_levels': powerup_levels
                        })
                        cap.release()
                        cv2.destroyAllWindows()
                        return  # Return to main menu instead of exit
        
        # Check for collisions with coins and powerups
        coins_to_remove = []
        # Calculate player center coordinates
        player_center_x = player_x + player_width // 2
        player_center_y = player_y + player_height // 2
        
        for coin in coins:
            coin_center_x = coin[0] + coin_size // 2
            coin_center_y = coin[1] + coin_size // 2
            
            distance = np.sqrt((coin_center_x - player_center_x)**2 + (coin_center_y - player_center_y)**2)
            if distance < (coin_size // 2 + (player_width + player_height) // 4):
                score += coin[2]  # Add coin value to score
                total_coins += coin[2]  # Add to total coins
                coins_to_remove.append(coin)
                # Save game data when collecting coins
                save_game_data({
                    'total_coins': total_coins,
                    'high_score': high_score,
                    'powerup_levels': powerup_levels
                })
        
        powerups_to_remove = []
        for powerup in powerups:
            powerup_center_x = powerup[0] + coin_size // 2
            powerup_center_y = powerup[1] + coin_size // 2
            distance = np.sqrt((powerup_center_x - player_center_x)**2 + (powerup_center_y - player_center_y)**2)
            
            if distance < (coin_size // 2 + (player_width + player_height) // 4):
                powerup_type = powerup[2]
                base_duration = POWERUP_TYPES[powerup_type]['duration']
                level_bonus = powerup_levels[powerup_type] - 1
                active_powerups[powerup_type] = (base_duration + level_bonus) * 30  # 30 frames per second
                if powerup_type == 'shield':
                    shield_active = True
                elif powerup_type == 'slow':  # Changed from speed to slow
                    slow_active = True
                elif powerup_type == 'magnet':
                    magnet_active = True
                powerups_to_remove.append(powerup)
        
        # Update powerup durations and effects
        for powerup_type in list(active_powerups.keys()):
            active_powerups[powerup_type] -= 1
            if active_powerups[powerup_type] <= 0:
                del active_powerups[powerup_type]
                if powerup_type == 'shield':
                    shield_active = False
                elif powerup_type == 'slow':  # Changed from speed to slow
                    slow_active = False
                elif powerup_type == 'magnet':
                    magnet_active = False
        
        # Remove collected items
        for coin in coins_to_remove:
            if coin in coins:
                coins.remove(coin)
        for powerup in powerups_to_remove:
            if powerup in powerups:
                powerups.remove(powerup)
        
        # Draw the player with shield effect if active
        player_color = (0, 255, 0)
        if shield_active:
            cv2.circle(frame, (int(player_x + player_width//2), int(player_y + player_height//2)),
                      int(max(player_width, player_height) * 0.7), (255, 165, 0), 2)
        cv2.rectangle(frame, (int(player_x), int(player_y)),
                     (int(player_x + player_width), int(player_y + player_height)), player_color, -1)
        
        # Draw obstacles
        for obstacle in obstacles:
            cv2.rectangle(frame, (int(obstacle[0]), int(obstacle[1])),
                         (int(obstacle[0] + obstacle_width), int(obstacle[1] + obstacle_height)), (0, 0, 255), -1)
        
        # Draw coins with sparkle effect
        for coin in coins:
            # Draw main coin circle
            center_x = int(coin[0] + coin_size // 2)
            center_y = int(coin[1] + coin_size // 2)
            cv2.circle(frame, (center_x, center_y),
                      coin_size // 2, (0, 255, 255), -1)
            # Add sparkle effect
            sparkle_points = [
                (int(coin[0] + coin_size // 4), int(coin[1] + coin_size // 4)),
                (int(coin[0] + 3 * coin_size // 4), int(coin[1] + coin_size // 4)),
                (int(coin[0] + coin_size // 2), int(coin[1] + coin_size // 2))
            ]
            for point in sparkle_points:
                cv2.circle(frame, point, 2, (255, 255, 255), -1)
                
        # Draw powerups
        for powerup in powerups:
            powerup_type = powerup[2]
            powerup_center_x = int(powerup[0] + coin_size // 2)
            powerup_center_y = int(powerup[1] + coin_size // 2)
            cv2.circle(frame, (powerup_center_x, powerup_center_y),
                      coin_size // 2, POWERUP_TYPES[powerup_type]['color'], -1)
            
        # Display score, coins, and active powerups
        cv2.putText(frame, f"Score: {score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Coins: {total_coins}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        y_offset = 90
        for powerup_type, time_left in active_powerups.items():
            cv2.putText(frame, f"{powerup_type.title()}: {time_left//30}s", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, POWERUP_TYPES[powerup_type]['color'], 2)
            y_offset += 25
        
        # Show upgrade menu hint
        cv2.putText(frame, "Press 'U' for upgrades", (frame_width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Face Detection Game', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            # Show upgrade menu
            while True:
                menu = show_store_menu()
                cv2.imshow('Face Detection Game', menu)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    try_upgrade_powerup(0)  # Shield
                elif key == ord('2'):
                    try_upgrade_powerup(1)  # Slow
                elif key == ord('3'):
                    try_upgrade_powerup(2)  # Magnet
                elif key == ord(' '):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if score > high_score:
        high_score = score
        save_game_data({
            'total_coins': total_coins,
            'high_score': high_score,
            'powerup_levels': powerup_levels
        })

def main():
    global total_coins, high_score
    cap = cv2.VideoCapture(0)
    
    # Main menu loop
    while True:
        menu = show_main_menu()
        cv2.imshow('Face Detection Game', menu)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):  # Start Game
            # Show instructions first
            while True:
                cv2.imshow('Face Detection Game', instructions)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    run_game()
                    break
                elif key == ord('q'):
                    break
        elif key == ord('2'):  # Store
            while True:
                store = show_store_menu()
                cv2.imshow('Face Detection Game', store)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    try_upgrade_powerup(0)  # Shield
                elif key == ord('2'):
                    try_upgrade_powerup(1)  # Slow
                elif key == ord('3'):
                    try_upgrade_powerup(2)  # Magnet
                elif key == ord(' '):
                    break
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
