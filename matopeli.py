import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle


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

def refreshMap(map, snake, point):
    
    
    for i in range(mapsize):
        
        for j in range(mapsize):
            
            if [j,i] == point:
                
                map[j][i] = 2
            else:
                map[j][i] = 0
    
    #print(point)
    
    for cords in snake:
        
        map[cords[1]][cords[0]] = 1
    
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
            
            if(map[(y+i)%mapsize][(x+j)%mapsize] == 0):
            
                return [(x+j)%mapsize,(y+i)%mapsize]
        
    return [-1,-1]

#def fixPoints(map):
    
    
                
            
def drawMap(map):
    
    for row in map:
        
        s = ""
        
        for col in row:
            
            if col == 0:
                s += "0"
            elif col == 1:
                s += "S"
            else:
                s += "*"
                       
        print(s)
    print()
    
def moveSnake(map,snake,direction):
    
    game_active = True
    point_active = True
    
    reward = 0
    
    
    #print(f"dir: {direction}")

    
    head = snake[0]
    
    if direction == "r":
        newx = head[0]+1
        newy = head[1]
    elif direction == "l":
        newx = head[0]-1
        newy = head[1]
    elif direction == "u":
        newx = head[0]
        newy = head[1]-1
    elif direction == "d":
        newx = head[0]
        newy = head[1]+1
   
        
    if newx >= mapsize or newx < 0 or newy >= mapsize or newy < 0:
        game_active = False
    elif map[newy][newx] == 1:
        game_active = False
    elif map[newy][newx] == 2:
        snake.insert(0,[newx,newy])
        point_active = False
        reward = 10
        
    else:
        #print(snake)
        snake.pop()
        #print(snake)
        snake.insert(0,[newx,newy])
       # print(snake)
        reward = -1
        
    return snake, game_active, point_active, reward

def cycle(map,snake,point,inp,previous_direction="l"):
        
    #map = deflatten(map)
    #print(len(map))
        
    #reward = 0
    
    #inp = input("dir: ")
        
    if inp == 0:
        inp = "r"
    elif inp == 1:
        inp = "l"
    elif inp == 2:
        inp = "u"
    elif inp == 3:
        inp = "d"
    else:
        inp = "wrong"
    
    if inp == "q":
        return
    elif inp == "r" and previous_direction == "l":
        direction = previous_direction
    elif inp == "l" and previous_direction == "r":
        direction = previous_direction
    elif inp == "u" and previous_direction == "d":
        direction = previous_direction
    elif inp == "d" and previous_direction == "u":
        direction = previous_direction
    elif inp == "wrong":
        direction = previous_direction
    else:
        direction = inp  
        previous_direction = inp
        
    #print(f"direction: {direction}")
        
    snake, game_active, point_active, reward = moveSnake(map,snake,direction)
    
    if not point_active:
        point = generatePoint(map)
    
    map= refreshMap(map,snake,point)
    
    
    return map, point, snake , reward, game_active, previous_direction

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
            
                     

def initGame():
    
    game_active = True
    #previous_direction = "l"
    #point = [[0,0],[1,3],[4,5],[5,1],[3,0],[7,0],[8,8],[8,1],[9,3],[3,7],[9,5],[6,2],[2,6],[6,8]]

    
    map = createMap()
    point = generatePoint(map)

   
    reward = 0

    #drawMap(map)
    snake = generateSnake()
    #print(snake)
    #point = generatePoint(map)
    #print(point)
    map = refreshMap(map,snake,point)
    #drawMap(map)
    #print(game_active)
    
    return map, snake, point
    
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
            
def play():
    
    state, snake, points = initGame()
    print(state)
    drawMap(state)
    print(points)
    running = True
    previous_direction = "l"
    while running:
        new_state, point, snake, reward, running, previous_direction = cycle(state,snake,points,0)
        drawMap(new_state)
    
    

def main():

    #play()
    
    
    previous_direction = "l"
    
    alpha = 0.9 # Learning rate
    gamma = 0.5    
    epsilon = 0.1
    best_reward = -1000
    num_of_interactions = []
    num_of_evals = 50
    episode_nums = []
    best_tot_rewards=[]
    best_tot_rewards_real = []
    num_of_episodes = 100000
    num_of_steps = 50000
    states = {}
    statecount = 0

    q_table = np.zeros((1000000,4))

    for episode in range(num_of_episodes):
        
        actions = []
        rewards = []
        
        state, snake, point = initGame()
        if str(state) in states:
            statenum = states.get(str(state))
        else:
            states[str(state)] = statecount
            statenum = statecount
            statecount += 1
        
        running = True
        
        if np.random.uniform() < epsilon:
            action = np.argmax(q_table[statenum,:])
            action = action % 4
        else:
            action = np.random.randint(0,4)
            
        for step in range(num_of_steps):
            
            #print(f"begin action: {action}")
            
            new_state, point, snake, reward, running, previous_direction = cycle(state,snake,point,action,previous_direction)
            #print(len(state))
            
            if str(new_state) in states:
                new_statenum = states.get(str(new_state))
            else:
                states[str(new_state)] = statecount
                new_statenum = statecount
                statecount += 1
            
            if np.random.uniform() < epsilon:
                new_action = np.argmax(q_table[statenum,:])
                new_action = new_action % 4
            else:
                new_action = np.random.randint(0,4)
                
            #print(f"new action: {new_action}")
            #print(f"action action: {action}")
            #print(state)
            if not running:
                #print(state)
                q_table[statenum,action] = reward
               # print(f"break action: {new_action}")
                #print(f"action action2: {action}")
                break
            else:
                #print(f"new action1: {action}")
                #print(new_state == state)
                #print(statenum)
                q_table[statenum,action] = q_table[statenum,action] + alpha*(reward+gamma*np.max(q_table[new_statenum,:])-q_table[statenum,action])
                #print(q_table[statenum,action])
                #print(f"new action2: {action}")
                state = new_state
                action = new_action
                #print(f"end action: {action}")
                statenum = new_statenum
        if episode % 10000 == 0:
            #print(reward)
            #print(statenum)
            #print(q_table[statenum,action])
            #print(np.argmax(q_table[state,:]))
            episode_nums.append(episode)
            #print(q_table[5])
            #print(type(q_table))
            #print(type(num_of_evals*10))
            eval_reward = eval_policy(q_table,num_of_evals*10)
            best_tot_rewards_real.append(eval_reward)
            print(f'Reward after episode {episode+1} is {eval_reward}')
            
            
    print (f'Tot reward of the found policy: {best_reward}')
    print(q_table)
   # plt.plot(num_of_interactions, best_tot_rewards_real,label='Q-learning')
   # plt.legend()
   # plt.xlabel('Env. interactions')
   # plt.ylabel('Best reward')
   # plt.show()            



    # Testing
    
    sum_of_rewards = 0
    sum_of_actions = 0

    for i in range(10):
        state, snake, point = initGame()
        tot_reward = 0
        running = True
        previous_direction = "l"
        while running:

            statenum = states.get(str(state))


            action = np.argmax(q_table[statenum,:])
            #state, reward, done, truncated, info = env.step(action)
            state, point, snake, reward, running, previous_direction = cycle(state,snake,point,action,previous_direction)
            
            sum_of_actions += 1
            tot_reward += reward
            drawMap(state)
            time.sleep(0.5)
            if not running:
                print("Total reward %d" %tot_reward)
                break
        sum_of_rewards += tot_reward

    print(f"After 10 runs \naverage total reward: {sum_of_rewards/10}\naverage number of actions: {sum_of_actions/10}")
    
    '''
    while True:
        inp = input("train or test? or q to quit [train/test/q]: ")
        if inp == "train":
            train()
        elif inp == "test":
            test()
        elif inp == "q":
            return
        else:
            play()
            print("invalid input")
    '''

def train():
    
    previous_direction = "l"
    
    alpha = 0.9 # Learning rate
    gamma = 0.5    
    epsilon = 0.1
    best_reward = -1000
    num_of_interactions = []
    num_of_evals = 50
    episode_nums = []
    best_tot_rewards=[]
    best_tot_rewards_real = []
    num_of_episodes = 100000
    num_of_steps = 50000
    
    


    states = {}
    #with open('Found_States.pkl','rb') as f:
    #    states = pickle.load(f)
    statecount = len(states)
    
    #q_table = np.loadtxt('data.csv',delimiter=',')
    q_table = np.zeros((1000000,4))

    for episode in range(num_of_episodes):
        
        actions = []
        rewards = []
        
        state, snake, point = initGame()
        encState = encodeState(state)
        #print(encState)
        if encState in states.keys():
            #print("encstaet foubnd")
            statenum = states.get(encState)
        else:
            
            states[encState] = statecount
            
            statenum = statecount
            statecount = len(states)
            
        
        running = True
        
        if np.random.uniform() < epsilon:
            action = np.argmax(q_table[statenum,:])
            action = action % 4
        else:
            action = np.random.randint(0,4)
            
        for step in range(num_of_steps):
            
            #print(f"begin action: {action}")
            
            new_state, point, snake, reward, running, previous_direction = cycle(state,snake,point,action,previous_direction)
            #print(len(state))
            new_encState = encodeState(new_state)
            
            if new_encState in states.keys():
                new_statenum = states.get(new_encState)
            else:
                #print(len(states))
                states[new_encState] = statecount
                new_statenum = statecount
                statecount = len(states)
            
            if np.random.uniform() < epsilon:
                new_action = np.argmax(q_table[statenum,:])
                new_action = new_action % 4
            else:
                new_action = np.random.randint(0,4)
                
            #print(f"new action: {new_action}")
            #print(f"action action: {action}")
            #print(state)
            if not running:
                #print(state)
                q_table[statenum,action] = reward
               # print(f"break action: {new_action}")
                #print(f"action action2: {action}")
                break
            else:
                #print(f"new action1: {action}")
                #print(new_state)
                #print(state)
                #print(new_state == state)
                #print(new_encState == encState)
                #print(encState)
                #print(new_encState)
                #break
                #print(statenum)
                q_table[statenum,action] = q_table[statenum,action] + alpha*(reward+gamma*np.max(q_table[new_statenum,:])-q_table[statenum,action])
                #print(q_table[statenum,action])
                #print(f"new action2: {action}")
                state = new_state
                action = new_action
                #print(f"end action: {action}")
                statenum = new_statenum
                encState = new_encState
        if episode % 10000 == 0:
            #print(reward)
            #print(statenum)
            #print(q_table[statenum,action])
            #print(np.argmax(q_table[state,:]))
            episode_nums.append(episode)
            #print(q_table[5])
            #print(type(q_table))
            #print(type(num_of_evals*10))
            eval_reward = eval_policy(q_table,num_of_evals*10)
            best_tot_rewards_real.append(eval_reward)
            print(f'Reward after episode {episode+1} is {eval_reward}')
            
            
    print (f'Tot reward of the found policy: {best_reward}')
    print(q_table)
    print(f"States found: {len(states)}")
    #print(states)
    test(q_table,states)
    
    np.savetxt('data.csv',q_table,delimiter=',')
    
    with open('Found_States.pkl','wb') as f:
        pickle.dump(states,f)


def test(q_table,states):
    
    #q_table = np.loadtxt('data.csv',delimiter=',')
    #with open('Found_States.pkl','rb') as f:
    #    states = pickle.load(f)
    sum_of_rewards = 0
    sum_of_actions = 0

    for i in range(10):




        state, snake, point = initGame()
        encState = encodeState(state)
        tot_reward = 0
        running = True
        previous_direction = "l"
        while running:

            
            statenum = states.get(encState)


            action = np.argmax(q_table[statenum,:])
            print(f"Chosen Action: {action}")
            print(f"Current Statenum: {statenum}")
            #state, reward, done, truncated, info = env.step(action)
            state, point, snake, reward, running, previous_direction = cycle(state,snake,point,action,previous_direction)
            encState = states.get(encodeState(state))
            
            sum_of_actions += 1
            tot_reward += reward
            drawMap(state)
            time.sleep(0.5)
            if not running:
                print("Total reward %d" %tot_reward)
                break
        sum_of_rewards += tot_reward

    print(f"After 10 runs \naverage total reward: {sum_of_rewards/10}\naverage number of actions: {sum_of_actions/10}")              
        
    

   

if __name__ == "__main__":
    main()









