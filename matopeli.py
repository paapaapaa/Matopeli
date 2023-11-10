import random
import numpy as np
import time
import tensorflow
import os
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.backend import clear_session
from collections import deque
from keras.models import load_model
import math
import datetime


mapsize = 10
# 10x10 map
# 0 = empty
# 2 = point
# 1 = snek


def refreshMap(map, snake, points):
    """Refreshes the map with new information

    Returns:
        map: the state of the game with new position given by parameters
    """

    for i in range(mapsize):

        for j in range(mapsize):

            if [j, i] in points:

                map[i][j] = 2
            else:
                map[i][j] = 0

    # To place a different symbol for head of snake for the agent
    Head_placed = False

    for cords in snake:

        if Head_placed:
            map[cords[1]][cords[0]] = 1
        else:
            map[cords[1]][cords[0]] = 3
            Head_placed = True

    return map


def generateSnake(x=mapsize, y=mapsize):
    """
    At the start of the game generates snake in the middle left
    """

    snake = []

    for i in range(4):

        cords = []
        cords.append(int(x/2+i))
        cords.append(int(y/2))
        snake.append(cords)

    return snake


def generatePoint(map):
    """Generates a new point for the map where the snake isn't currently

    Returns:
        [x,y]: coordinates for the generated point
    """

    x = random.randint(0, mapsize-1)
    y = random.randint(0, mapsize-1)

    for i in range(mapsize):

        for j in range(mapsize):

            if (map[(x+i) % mapsize][(y+j) % mapsize] == 0):

                return [(y+j) % mapsize, (x+i) % mapsize]

    return [-1, -1]


def drawMap(map):
    """
    Prints the gamestate
    """
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    RESET = '\x1b[0m'
    s = f""
    for row in map:

        for col in row:

            if col == 0:
                s += "0"
            elif col == 1 or col == 3:
                s += f"{GREEN}S{RESET}"
            else:
                s += f"{RED}*{RESET}"
        s += "\n"

    print(s, end='\r')


def moveSnake(map, snake, direction, points):
    """This function handles the movement of the snake and the reward given for the action.

        This is the reward structure for the agent:
            If the snake runs into a wall or into its own body reward -10 is given
            If the snake eats a point, it grows and a reward 20 is given

            For movement the snake is rewarded based on euclidean distance between the head of the snake and the point.
            The closer the snake is the more reward is given for moving toward the point and vice versa if moving away.

    Returns:
        snake: new coordinates for the snake
        game_active: Bool whether the game is ended or not
        points: An updated list of points on the map that contain coordinates for the points (There may be multiple if configured that way)
        reward: reward given to the agent for the action, ranges from -10 to 20
    """

    game_active = True

    reward = 0

    # 0 = f
    # 1 = r
    # 2 = l

    head = snake[0]
    second = snake[1]
    point = points[0]

    if second == [head[0]-1, head[1]]:
        previous = "l"
    elif second == [head[0]+1, head[1]]:
        previous = "r"
    elif second == [head[0], head[1]-1]:
        previous = "u"
    else:
        previous = "d"

    if direction not in [0, 1, 2]:    # if invalid input is chosen somehow, choose forward
        direction = 0

    if direction == 0:              # Direction 0 : Forward
        if previous == "l":
            new_direction = "r"
        elif previous == "r":
            new_direction = "l"
        elif previous == "u":
            new_direction = "d"
        else:
            new_direction = "u"
    elif direction == 1:            # Direction 1 : Right
        if previous == "l":
            new_direction = "d"
        elif previous == "r":
            new_direction = "u"
        elif previous == "u":
            new_direction = "l"
        else:
            new_direction = "r"
    else:                           # Direction 2 : Left
        if previous == "l":
            new_direction = "u"
        elif previous == "r":
            new_direction = "d"
        elif previous == "u":
            new_direction = "r"
        else:
            new_direction = "l"

    if new_direction == "r":
        newx = head[0]+1
        newy = head[1]
    elif new_direction == "l":
        newx = head[0]-1
        newy = head[1]
    elif new_direction == "u":
        newx = head[0]
        newy = head[1]-1
    elif new_direction == "d":
        newx = head[0]
        newy = head[1]+1

    if newx >= mapsize or newx < 0 or newy >= mapsize or newy < 0:
        game_active = False
        reward = -10
    elif map[newy][newx] == 1:
        game_active = False
        reward = -10
    elif map[newy][newx] == 2:
        snake.insert(0, [newx, newy])
        points.pop(points.index([newx, newy]))
        reward = 20

    else:

        snake.pop()

        old_dist = math.sqrt((head[0]-point[0])**2 + (head[1]-point[1])**2)
        snake.insert(0, [newx, newy])

        dist = math.sqrt((newx-point[0])**2 + (newy-point[1])**2)

        if dist != 0:

            if dist >= old_dist:
                reward = (1-(1/(dist)))*-0.5
            else:
                reward = (1/dist)

        else:
            reward = 0

    return snake, game_active, points, reward


def cycle(map, snake, points, direction, point_count=1):
    """Advances the game by one gamecycle


    Returns:
        map: the gamestate as a numpy ndarray
        snake: new coordinates for the snake
        game_active: Bool whether the game is ended or not
        points: An updated list of points on the map that contain coordinates for the points (There may be multiple if configured that way)
        reward: reward given to the agent for the action, ranges from -10 to 20
    """

    snake, game_active, points, reward = moveSnake(
        map, snake, direction, points)

    if (len(points) < point_count):
        points.append(generatePoint(map))

    map = refreshMap(map, snake, points)

    return map, points, snake, reward, game_active


def initGame(starting_points=1):
    """Initializes the game by calling the necessary functions

    Returns:
        map: ndarray of the initial gamestate
        snake: coordinates of the snake
        points: coordinates for the points generated
    """

    map = np.zeros((mapsize, mapsize), dtype=int)

    snake = generateSnake()

    points = []

    for i in range(starting_points):
        map = refreshMap(map, snake, points)
        points.append(generatePoint(map))

    map = refreshMap(map, snake, points)

    return map, snake, points


def main():
    """
    Main function for deciding whether to train or test
    """

    while True:

        command = input("Do you want to Train or Test [Train/Test/Q]: ")

        if command == "Train" or command == "train":

            while True:
                newold = input(
                    "Do you want to Train a new model or continue training a saved one [New/Old]: ")
                if newold == "New" or newold == "new":
                    trainNN(True)
                    break

                elif newold == "Old" or newold == "old":
                    trainNN()
                    break

                else:
                    print("Invalid Input >:c")

        elif command == "Test" or command == "test":

            testNN()
        elif command == "q" or command == "Q":
            break

        else:
            print("Invalid Input")

    return


def trainNN(new=False):
    """
    Trains a Deep Q-Network model (DQN) with a Convoluted Neural Network (CNN) using epsilon-greedy exploration

    The game is a custom snake game.

    Parameters:
        new: Boolean whether to start training a new network or continue training an old one
    """

    num_of_interactions = []

    stucks = 0
    previous_episodes = 0
    starting_points = 1
    episode_points = 0

    num_of_episodes = 10000  # number of episodes to run
    alpha = 0.001       # Learning rate
    gamma = 0.99        # Discoun_factor
    epsilon = 1
    epsilon_min = 0.2
    epsilon_decay = 0.9999

    batch = 32          # Experience replay batch-size
    fit_period = 2      # number of actions after which experience replay is initiated
    counter = 0         # counter for fit period
    target_update_period = 10  # number of episodes between target network updates

    # Buffer that stores maxlen gamestates, actions,rewards, new gamestates and whether the game is running for experience replay
    buffer = deque(maxlen=10000)

    total_interactions = 0

    game_average_interactions = 0

    if new:
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model = Sequential()

        model.add(InputLayer(input_shape=(10, 10, 1)))  # Starts with 2d layers
        model.add(Conv2D(128, 4, activation='relu'))
        model.add(Conv2D(128, 2, activation='relu'))
        model.add(Conv2D(64, 1, activation='relu'))
        model.add(Flatten())  # Dense layers
        model.add(Dense(200, activation='relu'))
        # The snake can take 3 actions
        model.add(Dense(3, activation='linear'))

    else:

        loginp = input("Give logdir eg. 20231031-234735: ")
        previnp = input("Give previous episodes ran (multiple of 50): ")
        prevepsilon = input("Give Starting epsilon: ")
        try:
            previous_episodes = int(previnp)
            epsilon = int(prevepsilon)
        except:
            print("Invalid Episodes or Epsilon")
            return
        log_dir = f"logs/{loginp}"
        model = load_model("cache", compile=False)

    summary_writer = tensorflow.summary.create_file_writer(
        log_dir)  # logwriter for tensorboard

    target_model = tensorflow.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=alpha)
    lossfn = tensorflow.keras.losses.MeanSquaredError()

    time_stamp1 = time.time()

    for episode in range(num_of_episodes):

        if episode % 50 == 0:

            # model is saved to a cache periodically in case the training is interrupted
            model.save("cache")

            time_stampfinal = time.time()
            print(
                f"\n\n\n Average Points: {episode_points/50} & Interactions: {game_average_interactions/50} past 50 games played at episode {episode+1}\nEpsilon {epsilon}\n")
            print(f"Total states: {total_interactions}")
            print(f"time taken {time_stampfinal-time_stamp1}\n\n")

            game_average_interactions = 0
            episode_points = 0
            time_stamp1 = time.time()

        # Target model is updated every 10 episodes
        if episode % target_update_period == 0:
            target_model.set_weights(model.get_weights())

        # Epsilon decay limiters, to control exploration vs exploitation
        if episode + previous_episodes >= 2000 and episode + previous_episodes < 4000:
            epsilon_min = 0.1
        elif episode + previous_episodes >= 4000 and episode + previous_episodes < 8500:
            epsilon_min = 0.01
        else:
            epsilon_min = 0
            epsilon = 0

        total_reward = 0
        num_of_interactions = 0

        state, snake, points = initGame(starting_points)
        if episode % 500 == 0:
            drawMap(state)
        flatstate = state

        point_cache = points.copy()
        point_counter = 0

        running = True

        while running:

            counter += 1

            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 3)

            else:

                action = np.argmax(model.predict(
                    flatstate.reshape(1, 10, 10), verbose=0))

            new_state, points, snake, reward, running = cycle(
                state, snake, points, action, starting_points)

            # Shows every 500th episode training run on the console
            if episode % 500 == 0:

                drawMap(new_state)
                time.sleep(.1)

            new_flatstate = new_state

            num_of_interactions += 1

            total_reward += reward

            buffer.append((flatstate, action, reward, new_flatstate, running))

            # Experience replay
            if len(buffer) >= batch and counter >= fit_period:

                counter = 0
                minibatch = random.sample(buffer, batch)

                mini_state, mini_action, mini_reward, mini_new_state, mini_running = zip(
                    *minibatch)
                mini_state = np.array(mini_state)
                mini_action = np.array(mini_action)
                mini_reward = np.array(mini_reward)
                mini_new_state = np.array(mini_new_state)
                mini_running = np.array(mini_running)

                target_q_values = target_model.predict(mini_state, verbose=0)
                max_next_q_values = np.max(target_model.predict(
                    mini_new_state, verbose=0), axis=1)

                for i in range(batch):
                    if mini_running[i]:
                        target_q_values[i][mini_action[i]
                                           ] = mini_reward[i] + gamma * max_next_q_values[i]
                    else:
                        target_q_values[i][mini_action[i]] = mini_reward[i]

                with tensorflow.GradientTape() as tape:
                    predicted_q_values = model(mini_state)
                    loss = lossfn(target_q_values, predicted_q_values)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))
                clear_session()     # this is not probably efficient, but the tensors are not ever released without this. removing would lead to huge memory leakage

                with summary_writer.as_default():

                    tensorflow.summary.scalar(
                        "Loss", loss, step=episode+previous_episodes)

            if episode % 500 == 0:
                print(
                    f"reward: {total_reward}\interactions: {num_of_interactions}")

            if episode >= 100:
                epsilon *= epsilon_decay
                if epsilon < epsilon_min:
                    epsilon = epsilon_min

            if points == point_cache:
                point_counter += 1

                if point_counter >= 500:
                    stucks += 1
                    break
            else:
                point_cache = points.copy()
                point_counter = 0
                episode_points += 1

            state = new_state
            flatstate = new_flatstate

        # data that is stored in the logs for tensorboard
        with summary_writer.as_default():
            tensorflow.summary.scalar(
                "Episode Cycles", num_of_interactions, step=episode+previous_episodes)
            tensorflow.summary.scalar(
                "Episode Reward", total_reward, step=episode+previous_episodes)
            tensorflow.summary.scalar(
                "Epsilon", epsilon, step=episode+previous_episodes)
            tensorflow.summary.scalar(
                "Points Eaten", episode_points, step=episode+previous_episodes)
        total_interactions += num_of_interactions
        game_average_interactions += num_of_interactions

    model.summary()
    print(f"\nStuck loops terminated: {stucks}\n")
    summary_writer.close()

    while True:
        inp = input(
            "Do you want to overwrite the previous model with this one? [Y/N]: ")

        if inp == 'Y' or inp == 'y':
            model.save("model")
            print("Model saved.")
            break
        elif inp == 'N' or inp == 'n':
            print("Model discarded.")
            break
        else:
            print("Invalid input")


def testNN():
    """
    Tests a saved model by running 50 games, first 5 of which are shown on the console (they might take long)
    """

    model = load_model("model", compile=False)
    sum_of_rewards = 0
    sum_of_actions = 0
    sum_of_points = 0

    model.summary()

    for i in range(50):
        state, snake, points = initGame()
        flatState = state
        tot_reward = 0
        counter = 0
        points_collected = 0
        running = True
        while running:

            action = np.argmax(model.predict(
                flatState.reshape(1, 10, 10), verbose=0))
            state, points, snake, reward, running = cycle(
                state, snake, points, action)
            flatState = state
            sum_of_actions += 1
            tot_reward += reward

            if reward < 20:
                counter += 1
            else:
                counter = 0
                points_collected += 1

            if counter >= 500:
                print("\nJUMI-KESKEYTYS\n")
                if i <= 5:
                    time.sleep(2)
                running = False

            if (i <= 5):
                # print(reward)
                os
                drawMap(state)
                time.sleep(.15)
            if not running:
                print("Total reward %d" % tot_reward)
                break
        sum_of_rewards += tot_reward
        sum_of_points += points_collected

    print(
        f"After 50 runs \naverage total reward: {sum_of_rewards/50}\naverage number of actions: {sum_of_actions/50}\naverage points collected {sum_of_points/50}")


if __name__ == "__main__":
    main()
