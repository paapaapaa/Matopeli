import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import tensorflow
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from keras.models import load_model
import math
import datetime


mapsize=10

#40x40 map
# 0 = empty
# 2 = point
# 1 = snek
def createMap(x=mapsize,y=mapsize):
    
    rows = []
    
    
    for i in range(x):
        
        cols = []

        
        for j in range(y):
            
            
            cols.append(0)
        
        rows.append(cols)     
            
    #print(len(rows))
    return rows

def refreshMap(map, snake, points):
    
    #[8,0]
    
    for i in range(mapsize):
        
        for j in range(mapsize):
            
            if [j,i] in points:
                
                map[i][j] = 2
            else:
                map[i][j] = 0
    
    #print(point)
    
    Head_placed = False                 # To place a different symbol for head of snake for the agent
    
    for cords in snake:
        
        if Head_placed:
            map[cords[1]][cords[0]] = 1
        else:
            map[cords[1]][cords[0]] = 3
            Head_placed = True
    
    #map[point[1]][point[0]] = 2
    
    
    return map
    
        
        
def generateSnake(x=mapsize,y=mapsize):
    
    snake = []
    
    
    for i in range(4):
        
        cords = []
        cords.append(int(x/2+i))
        cords.append(int(y/2))
        snake.append(cords)
    
    return snake
    
def generatePoint(map):
    
    x = random.randint(0,mapsize-1)
    y = random.randint(0,mapsize-1)
    
    for i in range(mapsize):
        
        for j in range(mapsize):
            
            if(map[(x+j)%mapsize][(y+i)%mapsize] == 0):
                #print(f"generated Point: {[(y+i)%mapsize,(x+j)%mapsize]}")
                return [(y+i)%mapsize,(x+j)%mapsize]
        
    return [-1,-1]

#def fixPoints(map):
    
    
                
            
def drawMap(map):
    
    for row in map:
        
        s = ""
        
        for col in row:
            
            if col == 0:
                s += "0"
            elif col == 1 or col == 3:
                s += "S"
            else:
                s += "*"
                       
        print(s)
    print()
    
def moveSnake(map,snake,direction,points):
    
    game_active = True
    #point_active = True
    
    reward = 0
    
    # 0 = f
    # 1 = r
    # 2 = l
    
    
    
    #print(f"dir: {direction}")

    
    
    
    head = snake[0]
    second = snake[1]
    
    if second == [head[0]-1,head[1]]:
        previous = "l"
    elif second == [head[0]+1,head[1]]:
        previous = "r"
    elif second == [head[0],head[1]-1]:
        previous = "u"
    elif second == [head[0],head[1]+1]:
        previous = "d"
    else:
        print("Second point of snake is in impossible location (THIS SHOULD NOT HAPPEN)")

    if direction not in [0,1,2]:    # if invalid input is chosen somehow, choose forward
        direction = 0
        print("Hello Joona :), somehow directional input that was not 0,1 or 2 was received, Have a nice debugging!")


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
        snake.insert(0,[newx,newy])
        points.pop(points.index([newx,newy]))
        #print(f"Point: {removed} was removed from points")
        #points.index
        reward = 20
        #print("Point Ate")
        
    else:
        #print(snake)
        snake.pop()
        #print(snake)
        #old_dist = math.sqrt((head[0]-point[0])**2 +(head[1]-point[1])**2)
        snake.insert(0,[newx,newy])
       # print(snake)
        #reward = 0
        
        #dist = math.sqrt((newx-points[0][0])**2 +(newy-points[0][1])**2)
        reward = -0.1
        '''
        if dist != 0:
            #print([newx,newy])
            #print(dist)
            #if old_dist > dist:

            reward = -0.1
            #else:
            #    reward = -0.01
        else:
            reward = 0
            print("MITEN ON MAHDOLLISTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        '''
            
    #if not point_active:
    #    print(reward)
        
    return snake, game_active, points, reward

def cycle(map,snake,points,direction,point_count=1):
        
    #map = deflatten(map)
    #print(len(map))
        
    #reward = 0
    
    #inp = input("dir: ")
        
    #print(f"direction: {direction}")
        
    snake, game_active, points, reward = moveSnake(map,snake,direction,points)
    
    if (len(points) < point_count):
        points.append(generatePoint(map))
    
    map= refreshMap(map,snake,points)
    
    
    return oneLine(map), points, snake , reward, game_active

def flatten(map):
    
    state = []
    
    for row in map:
        
        for col in row:
            
            state.append(col)
    
    return state

def deflatten(state):
    
    map = []
    
    for i in range(mapsize):
        
        row = []
        
        for j in range(mapsize):
            
            row.append(state[i*mapsize+j])
        
        map.append(row)
    
    return map
            
                     

def initGame(starting_points=1):
    
    #game_active = True
    #previous_direction = "l"
    #point = [[0,0],[1,3],[4,5],[5,1],[3,0],[7,0],[8,8],[8,1],[9,3],[3,7],[9,5],[6,2],[2,6],[6,8]]

    
    map = createMap()


   
    #reward = 0

    #drawMap(map)
    snake = generateSnake()
    
    points = []
    
    for i in range(starting_points):
        points.append(generatePoint(map))
        map = refreshMap(map,snake,points)
    #print(snake)
    #point = generatePoint(map)
    #print(point)
    map = refreshMap(map,snake,points)
    #drawMap(map)
    #print(game_active)
    
    return map, snake, points
    
def eval_policy(qtable_, num_of_episodes_):
    rewards = []

    
        
    
    for episode in range(num_of_episodes_): # This is out loop over num of episodes
        state,snake,point = initGame()
        total_reward = 0
        running = True
        previous_direction = "l"
        
        
        while running:
            action = np.argmax(qtable_[state,:])
            action = action % 4
            map, point, snake, reward, running, previous_direction = cycle(state,snake,point,action,previous_direction)
            total_reward += reward
            #print(reward)
        
        
        '''
        for step in range(max_steps_):
            action = np.argmax(qtable_[state,:])
            new_state, reward, done, truncated, info = env.step(action)
            num_of_interactions += 1
            total_reward += reward
            if done:
                break
            else:
                state = new_state
        '''
        rewards.append(total_reward)
        
    return sum(rewards)/num_of_episodes_

def encodeState(state):
    
    encoded = ""
    
    for row in state:
        
        for col in row:
            
            encoded = encoded + str(col)
    
    
    return encoded

def oneLine(state):
    
    result = np.zeros((10,10),dtype=int)
    r = 0
    
    
    for row in state:
        
        c = 0
        
        for col in row:
            
            result[r][c] = col
            c += 1
        
        r += 1

    #result =result.flatten()
    #result.append()

    return result #.flatten()
        
            
def play(starting_points = 1):
    #buffer = deque()
    for number in range(10):
        state, snake, point = initGame(starting_points)
        print(state)
        drawMap(state)
        print(point)
        running = True
        previous_direction = "l"
        dir = 0
        total_reward = 0
        while running:
            
            i = input("give direction 0=f, 1=r, 2=l: ")
            if i not in ['0','1','2']:
                i = 0
            new_state, point, snake, reward, running = cycle(state,snake,point,int(i))
            total_reward += reward
            print(reward)
            #dir = 0
            drawMap(new_state)
            print(point)
            #print(oneLine(state).flatten())
            #print(oneLine(new_state).flatten())
            #print(point)
            #print(snake)
            #time.sleep(1)
            #buffer.append((oneLine(state).flatten(),int(i),reward,oneLine(new_state).flatten(),running))
            state = new_state
        print(total_reward)
    #print(len(buffer))
    #with open('expert_buffer.pkl','wb') as f:
    #    pickle.dump(buffer,f)

def correct():
    buffer2 = deque(maxlen=2000)
    with open('expert_buffer.pkl','rb') as f:
        buffer = pickle.load(f)
    
    for mini_state,mini_action,mini_reward,mini_new_state,mini_running in buffer:   
        buffer2.append((mini_state,int(mini_action),mini_reward,mini_new_state,mini_running))

    with open('expert_buffer.pkl','wb') as f:
        pickle.dump(buffer2,f)   

def main():


    #correct()
    while True:
        
        command = input("Do you want to Train or Test [Train/Test/Q]: ")
        
        
        if command == "Train" or command == "train":
            
            while True:
                newold = input("Do you want to Train a new model or continue training a saved one [New/Old]: ")
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
    
    


    num_of_interactions = []
    
    epoch_points = 0
    epoch_interactions = 0
    epoch_length = 0
    stucks = 0
    previous_episodes = 0
    starting_points = 25
    episode_points = 0
    
    fit_check = False
    num_of_episodes = 3000
    alpha = 0.001 # Learning rate
    gamma = 0.95
    epsilon = 1
    epsilon_min = 0.2
    epsilon_decay = 0.9999
    #               0.9998
    batch = 64
    fit_period = 2
    counter = 0
    buffer = deque(maxlen=10000)

    total_interactions = 0
    
    game_average_points = 0
    game_average_interactions = 0
    if new:
        #log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #summary_writer = tensorflow.summary.create_file_writer(log_dir)
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model = Sequential()
        model.add(Dense(100,input_dim=100,activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(3,activation='linear'))
    else:
    
        log_dir = "logs/20231025-235320"
        model = load_model("model",compile=False)
        previous_episodes = 16000
        starting_points = 1
        epsilon = 0.1
        epsilon_min = 0.1
    
    #log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tensorflow.summary.create_file_writer(log_dir)    

    target_model = tensorflow.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    
    #model.compile(loss='mse',optimizer=Adam(learning_rate=alpha),metrics=['mae'])
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=alpha)
    lossfn = tensorflow.keras.losses.MeanSquaredError()
    


    time_stamp1 = time.time()


    for episode in range(num_of_episodes):
        
        episode_points = 0

        if episode % 400 == 0 and starting_points > 1:
            starting_points -= 1
        
        if episode % 500 == 0:
            model.save("cache")
        
        if episode % 50 == 0:
            time_stampfinal = time.time()
            print(f"\n\n\n Average Points: {game_average_points/50} & Interactions: {game_average_interactions/50} past 50 games played at episode {episode+1}\nEpsilon {epsilon}\n")
            print(f"Total states: {total_interactions}")
            print(f"time taken {time_stampfinal-time_stamp1}\n\n")
            game_average_points = 0
            game_average_interactions = 0
            time_stamp1 = time.time()

        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())
            
        epoch_length += 1
        total_reward = 0
        num_of_interactions = 0
        
        state, snake, points = initGame(starting_points)
        if episode % 500 == 0:
            drawMap(state)
        flatstate = oneLine(state).flatten()
        point_cache = points.copy()
        point_counter = 0
 
        running = True
        
        
        while running:
            
            counter += 1
            epoch_interactions += 1

            if np.random.rand() <= epsilon:
                action = np.random.randint(0,3)
            
            else:
                action = np.argmax(model.predict(flatstate.reshape(1,-1),verbose=0))
            
            new_state, points, snake, reward, running = cycle(state,snake,points,action,starting_points)
            
            if episode % 500 == 0:
        
                drawMap(new_state)
                time.sleep(.5)
            
            new_flatstate = new_state.flatten()

            num_of_interactions += 1
            

            total_reward += reward
            
            buffer.append((flatstate,action,reward,new_flatstate,running))
            
            
            if len(buffer) >= batch and counter >= fit_period:

                counter = 0
                minibatch = random.sample(buffer,batch)
                
                mini_state,mini_action,mini_reward,mini_new_state,mini_running = zip(*minibatch)
                mini_state = np.array(mini_state)
                mini_action = np.array(mini_action)
                mini_reward = np.array(mini_reward)
                mini_new_state = np.array(mini_new_state)
                mini_running = np.array(mini_running)
                
                target_q_values = target_model.predict(mini_state,verbose=0)
                max_next_q_values = np.max(target_model.predict(mini_new_state,verbose=0),axis=1)
                
                for i in range(batch):
                    if mini_running[i]:
                        target_q_values[i][mini_action[i]] = mini_reward[i] + gamma * max_next_q_values[i]
                    else:
                        
                        target_q_values[i][mini_action[i]] = mini_reward[i]

                
                with tensorflow.GradientTape() as tape:
                    predicted_q_values = model(mini_state)
                    loss = lossfn(target_q_values,predicted_q_values)
                gradients = tape.gradient(loss,model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                
                with summary_writer.as_default():
                
                    tensorflow.summary.scalar("Loss", loss, step=episode+previous_episodes)


            if episode % 500 == 0:
                print(f"reward: {total_reward}\interactions: {num_of_interactions}")
            
                        
            if  episode >= 100:
                epsilon *= epsilon_decay
                if epsilon < epsilon_min:
                    epsilon = epsilon_min
            
            if episode +previous_episodes >= 4000:
                epsilon_min = 0.1  
            
            if points == point_cache:
                point_counter += 1
                
                if point_counter >= 100:
                    stucks +=1
                    break
            else:
                point_cache = points.copy()
                point_counter = 0
                episode_points += 1
              
            state = new_state.copy()
            flatstate = new_flatstate.copy()
        with summary_writer.as_default():
            tensorflow.summary.scalar("Episode Cycles", num_of_interactions, step=episode+previous_episodes)
            tensorflow.summary.scalar("Episode Reward", total_reward, step=episode+previous_episodes)
            tensorflow.summary.scalar("Epsilon", epsilon, step=episode+previous_episodes)
            tensorflow.summary.scalar("Points Eaten", episode_points, step=episode+previous_episodes)
        epoch_points += total_reward  
        total_interactions += num_of_interactions
        game_average_points += total_reward
        game_average_interactions += num_of_interactions
        
    model.summary()
    print(f"\nStuck loops terminated: {stucks}\n")
    summary_writer.close()
    
    while True:
        inp = input("Do you want to overwrite the previous model with this one? [Y/N]: ")
        
        if inp == 'Y' or inp =='y':
            model.save("model")
            print("Model saved.")
            break
        elif inp == 'N' or inp == 'n':
            print("Model discarded.")
            break
        else:
            print("Invalid input")

    
def testNN():
    
    # Testing
    model = load_model("model",compile=False)
    sum_of_rewards = 0
    sum_of_actions = 0
    model.summary()

    for i in range(50):
        state, snake, points = initGame()
        state = oneLine(state)
        flatState = state.flatten()
        tot_reward = 0
        running = True
        previous_direction = "l"
        counter = 0
        while running:

            
            #print(np.identity(10)[statenum:statenum + 1])
            
            #action = np.argmax(model.predict(np.identity(1000)[statenum:statenum+1],verbose=0))
            action = np.argmax(model.predict(flatState.reshape(1,-1),verbose=0))
            #state, reward, done, truncated, info = env.step(action)
            state, points, snake, reward, running = cycle(state,snake,points,action)
            flatState = state.flatten()
            sum_of_actions += 1
            tot_reward += reward
            #print(tot_reward)
            
            if reward != 50:
                counter += 1
            else:
                counter = 0
                
            if counter >= 100:
                print("\nJUMI-KESKEYTYS\n")
                if i<= 5:
                    time.sleep(2)
                running = False
                #reward = -10
            if (i <= 5):
                print(reward)
                drawMap(state)
                time.sleep(.2)
            if not running:
                print(f"Total reward {tot_reward}")
                break
        sum_of_rewards += tot_reward

    print(f"After 50 runs \naverage total reward: {sum_of_rewards/50}\naverage number of actions: {sum_of_actions/50}")
    

   

if __name__ == "__main__":
    main()









