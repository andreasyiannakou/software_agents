# packages to import
import pygame
import numpy as np
from numpy import genfromtxt
f = r"./S.S. ANNE_evt.jpg"
import matplotlib.pyplot as plt
states = []

# general variables
random_state = np.random.RandomState(1)
n_episodes = 100

# actions
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# colours
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#finds the set of all letters next to the letter passed in        
def possible_next_states(state, states):
    actions = []
    nb = []
    both = []    
    for s in states:
        if state.row > 0:
            if state.col == s.col and (state.row - 1) == s.row:
                actions.append(UP)
                nb.append(s)
                both.append((UP, s)) 
        if state.row < 5:
            if state.col == s.col and (state.row + 1) == s.row:
                actions.append(DOWN)
                nb.append(s)
                both.append((DOWN, s))
        if state.col > 0:
            if (state.col - 1) == s.col and state.row == s.row:
                actions.append(LEFT)
                nb.append(s)
                both.append((LEFT, s))
        if state.col < 5:
            if (state.col + 1) == s.col and state.row == s.row:
                actions.append(RIGHT)
                nb.append(s)
                both.append((RIGHT, s))
    return actions, nb, both

#-------------------------DEFINE STATE CLASS-----------------------------------#
class State:
    # initialise the state object
    def __init__(self, row, col, num):
        self.row = row
        self.col = col
        self.num = num
        self.actions = []
        self.next_state = []
        self.both = []    

    # populate the list of next_states from this state    
    def populate_next_states(self):
        self.actions, self.next_state, self.both = possible_next_states(self, states)
        

def perform_action(state, action):
    for ns in state.both:
        if ns[0] == action:
            return ns[1]
    return state

#-------------------------DEFINE Agent CLASS-----------------------------------#
class Agent:
    # initialise the Agent object
    def __init__(self, row, col, num, start_state):
        self.row = row
        self.col = col
        self.num = num
        self.draw_loc = (col*50+25, row*50+125)
        self.current_state = start_state
        self.reward = 0
        self.reached_goal = False
    # reset the agent after each episode
    def reset(self, row, col, num, start_state):
        self.row = row
        self.col = col
        self.num = num
        self.draw_loc = (col*50+25, row*50+125)
        self.current_state = start_state
        self.reward = 0
        self.reached_goal = False
    # updates the state, reward and checks whether the agent has reached the goal state    
    def update_state(self, new_state):
        self.reward += r[(self.num-1), (new_state.num-1)]
        self.current_state = new_state
        self.row = new_state.row
        self.col = new_state.col
        self.num = new_state.num
        self.draw_loc = (new_state.col*50+25, new_state.row*50+125)
        if self.num == 27:
            self.reached_goal = True
        
# defines the reward/connection graph
r1 = genfromtxt(r'./Rewards.csv', delimiter=',')
#r2 = genfromtxt(r'C:\Users\Andreas\Documents\Data Science MSc\8 Software Agents\3 Coursework\Rewards2.csv', delimiter=',')
r = r1
# initialise the q matrix
q = np.zeros_like(r1)

# finds the action which provides the highest q reward
def max_Q(state, qt):
    val = -1000
    for s in state.next_state:
        q_t = qt[(state.num-1), (s.num-1)]
        if q_t > val:
            val = q_t
    return val

# udpates the q matrix for double q learning
def update_q(state, next_state, alpha, gamma, q):
    r_sa = r[(state.num-1), (next_state.num-1)]
    q_sa = q[(state.num-1), (next_state.num-1)]
    max_q = max_Q(next_state, q)
    new_q = q_sa + alpha * (r_sa + (gamma * max_q) - q_sa)
    q[(state.num-1), (next_state.num-1)] = new_q
    return q

# udpates the relevant q matrix for double q learning
def update_double_q(state, next_state, alpha, gamma, q_a, q_b):
    r_sa = r[(state.num-1), (next_state.num-1)]
    q_sa = q_a[(state.num-1), (next_state.num-1)]
    max_q = max_Q(next_state, q_b)
    new_q = q_sa + alpha * (r_sa + (gamma * max_q) - q_sa)
    q_a[(state.num-1), (next_state.num-1)] = new_q
    return q_a, q_b

# takes the results from all the episodes and plots them in a line graph
def draw_result_graph(results):
    fig, ax1 = plt.subplots()
    ax1.plot( results[:, 0], results[:, 1], color='skyblue', linewidth=1)
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('steps', color='blue')
    ax1.tick_params('y', colors='blue')
    ax2 = ax1.twinx()
    ax2.plot( results[:, 0], results[:, 2], color='red', linewidth=1)
    ax2.set_ylabel('cumulative reward', color='red')
    ax2.tick_params('y', colors='red')
    fig.tight_layout()
    plt.show()
    return
    
# sets the colour of the circle sprite on the map
def colour(i):
    if i%3 == 0:
        return RED
    elif i%3 == 1:
        return GREEN
    else:
        return BLUE

# initalise all the states
s1 = State(0, 0, 1)
s2 = State(0, 1, 2)
s3 = State(0, 2, 3)
s4 = State(0, 3, 4)
s5 = State(0, 4, 5)
s6 = State(1, 0, 6)
s7 = State(1, 1, 7)
s8 = State(1, 2, 8)
s9 = State(1, 4, 9)
s10 = State(2, 1, 10)
s11 = State(2, 2, 11)
s12 = State(2, 3, 12)
s13 = State(2, 4, 13)
s14 = State(2, 5, 14)
s15 = State(3, 0, 15)
s16 = State(3, 1, 16)
s17 = State(3, 2, 17)
s18 = State(3, 3, 18)
s19 = State(3, 5, 19)
s20 = State(4, 0, 20)
s21 = State(4, 3, 21)
s22 = State(4, 5, 22)
s23 = State(5, 0, 23)
s24 = State(5, 2, 24)
s25 = State(5, 3, 25)
s26 = State(5, 4, 26)
s27 = State(5, 5, 27)

# store the states in a list
states = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14
          , s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27]

# for all states find all their possible next states
for s in states:
    s.populate_next_states()

# initalise the Agent
myAgent = Agent(0, 2, 3, s3)

# q learning functions
def run_case(alpha, gamma, epsilon, reward, q, myAgent):
    episode = 1
    case_results = []
    for ep in range(n_episodes):
        eps = epsilon
        i = 0
        myAgent.reset(0, 2, 3, s3)

        while myAgent.reached_goal is False:
            current_state = myAgent.current_state
            if random_state.rand() < epsilon:
                #random
                actions = current_state.actions
                if type(actions) is int:
                    actions = [actions]
                random_state.shuffle(actions)
                action = actions[0]
                for ns in current_state.both:
                    if ns[0] == action:
                        next_state = ns[1]
                myAgent.update_state(next_state)            
            else:
                #greedy
                options = []
                for s in current_state.next_state:
                    options.append((s.num - 1))
                op = np.argmax(q[(myAgent.num-1), options])
                for ns in current_state.both:
                    if ns[1].num == (options[op]+1):
                        action = ns[0]
                if action == 0:
                    # take a random move if move not assigned
                    actions = current_state.actions
                    random_state.shuffle(actions)
                    action = actions[0]
                for ns in current_state.both:
                    if ns[0] == action:
                        next_state = ns[1]
                myAgent.update_state(next_state) 
                
            q = update_q(current_state, next_state, alpha, gamma, q)

            i += 1
            if eps >= 0.5:
                eps *= 0.99999
            else:
                eps *= 0.9999
            
            if episode == 1:
                if i == 1:
                    q1 = np.copy(q)
                elif i == 2:
                    q2 = np.copy(q)
                
        case_results.append([episode, i, myAgent.reward])
        episode += 1
    i = 0
    myAgent.reset(0, 2, 3, s3)
    # visualise the problem
    pygame.init()
    size = [300, 400]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Software Agents")
    image_surf = pygame.image.load(f).convert()
    screen.blit(image_surf,(0,0))
    pygame.display.flip()
    pygame.draw.circle(screen, colour(i-1), myAgent.draw_loc, 15)
    pygame.display.flip()
        
    while myAgent.reached_goal is False:
        current_state = myAgent.current_state   
        options = []
        for s in current_state.next_state:
            options.append((s.num - 1))
        
        op = np.argmax(q[(myAgent.num-1), options])
        for ns in current_state.both:
            if ns[1].num == (options[op]+1):
                action = ns[0]
        if action == 0:
            # take a random move
            actions = current_state.actions
            random_state.shuffle(actions)
            action = actions[0]
        for ns in current_state.both:
            if ns[0] == action:
                next_state = ns[1]
        myAgent.update_state(next_state)
        # draw circle/sprite in new locations
        pygame.draw.circle(screen, colour(i), myAgent.draw_loc, 15)
        pygame.display.flip()
        pygame.time.wait(50)
        i += 1
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.unicode == 'q':
                break
        pygame.display.flip()
    
    print('steps:', i, 'cumulative reward:', myAgent.reward)
    pygame.quit()   
    # draw the steps/rewards graph
    draw_result_graph(np.array(case_results))
    return np.array(case_results), q1, q2, q

# double q learning function
def run_advanced_case(alpha, gamma, epsilon, reward, q1, q2, myAgent):
    episode = 1
    case_results = []
    for ep in range(n_episodes):
        eps = epsilon
        i = 0
        myAgent.reset(0, 2, 3, s3)
        rd = random_state.rand()
        while myAgent.reached_goal is False:
            current_state = myAgent.current_state
            if random_state.rand() < epsilon:
                #random
                actions = current_state.actions
                if type(actions) is int:
                    actions = [actions]
                random_state.shuffle(actions)
                action = actions[0]
                for ns in current_state.both:
                    if ns[0] == action:
                        next_state = ns[1]
                myAgent.update_state(next_state)            
            else:
                #greedy
                options = []
                for s in current_state.next_state:
                    options.append((s.num - 1))
                if rd < 0.5:
                    op = np.argmax(q1[(myAgent.num-1), options])
                else:
                   op = np.argmax(q2[(myAgent.num-1), options]) 
                for ns in current_state.both:
                    if ns[1].num == (options[op]+1):
                        action = ns[0]
                if action == 0:
                    # take a random move if no move assigned
                    actions = current_state.actions
                    random_state.shuffle(actions)
                    action = actions[0]
                for ns in current_state.both:
                    if ns[0] == action:
                        next_state = ns[1]
                myAgent.update_state(next_state) 
            # update relevant q matrix
            if random_state.rand() < 0.5:
                q1, q2 = update_double_q(current_state, next_state, alpha, gamma, q1, q2)
            else:
                q2, q1 = update_double_q(current_state, next_state, alpha, gamma, q2, q1)
            
            i += 1
            if eps >= 0.5:
                eps *= 0.99999
            else:
                eps *= 0.9999
                
        case_results.append([episode, i, myAgent.reward])
        episode += 1
        
    i = 0
    myAgent.reset(0, 2, 3, s3)
    # visualise the problem
    pygame.init()
    size = [300, 400]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Software Agents")
    image_surf = pygame.image.load(f).convert()
    screen.blit(image_surf,(0,0))
    pygame.display.flip()
    pygame.draw.circle(screen, colour(i-1), myAgent.draw_loc, 15)
    pygame.display.flip()
    # loop until the goal is reached    
    while myAgent.reached_goal is False:
        current_state = myAgent.current_state   
        options = []
        for s in current_state.next_state:
            options.append((s.num - 1))
        rd = random_state.rand()
        if rd < 0.5:
            op = np.argmax(q1[(myAgent.num-1), options])
        else:
           op = np.argmax(q2[(myAgent.num-1), options]) 
        for ns in current_state.both:
            if ns[1].num == (options[op]+1):
                action = ns[0]
        if action == 0:
            # take a random move
            actions = current_state.actions
            random_state.shuffle(actions)
            action = actions[0]
        for ns in current_state.both:
            if ns[0] == action:
                next_state = ns[1]
        myAgent.update_state(next_state)
        # draw circle/sprite in new locations
        pygame.draw.circle(screen, colour(i), myAgent.draw_loc, 15)
        pygame.display.flip()
        pygame.time.wait(50)
        i += 1
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.unicode == 'q':
                break
        pygame.display.flip()
    
    print('steps:', i, 'cumulative reward:', myAgent.reward)
    pygame.quit()   
    # draw the steps/rewards graph
    draw_result_graph(np.array(case_results))
    return np.array(case_results), q1, q2

###epsilon 0.5
#case 1
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.5, 0.5  
results1, q1, q2, q_final1 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 2
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.2, 0.5      
results2, q1, q2, q_final2 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 3
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.5, 0.5
results3, q1, q2, q_final3 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 4
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.8, 0.5
results4, q1, q2, q_final4 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 5
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.2, 0.5
results5, q1, q2, q_final5 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 6
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.8, 0.5
results6, q1, q2, q_final6 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 7
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.2, 0.5      
results7, q1, q2, q_final7 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 8
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.5, 0.5
results8, q1, q2, q_final8 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 9
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.8, 0.5
results9, q1, q2, q_final9 = run_case(alpha, gamma, epsilon, r1, q, myAgent)

"""
###epsilon 0.1
#case 10
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.2, 0.1     
results10, q1, q2, q_final10 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 11
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.5, 0.1
results11, q1, q2, q_final11 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 12
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.8, 0.1
results12, q1, q2, q_final12 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 13
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.2, 0.1
results13, q1, q2, q_final13 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 14
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.5, 0.1
results14, q1, q2, q_final14 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 15
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.8, 0.1
results15, q1, q2, q_final15 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 16
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.2, 0.1      
results16, q1, q2, q_final16 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 17
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.5, 0.1
results17, q1, q2, q_final17 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 18
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.8, 0.1
results18, q1, q2, q_final18 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
"""

"""
###epsilon 0.9
#case 19
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.2, 0.9
results19, q1, q2, q_final19 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 20
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.5, 0.9
results20, q1, q2, q_final20 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 21
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.2, 0.8, 0.9
results21, q1, q2, q_final21 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 22
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.2, 0.9
results22, q1, q2, q_final22 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 23
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.5, 0.9
results23, q1, q2, q_final23 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 24
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.8, 0.9
results24, q1, q2, q_final24 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 25
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.2, 0.9    
results25, q1, q2, q_final125 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 26
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.5, 0.9
results26, q1, q2, q_final26 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
#case 27
q = np.zeros_like(r1)
alpha, gamma, epsilon = 0.8, 0.8, 0.9
results27, q1, q2, q_final27 = run_case(alpha, gamma, epsilon, r1, q, myAgent)
"""



# initialise q1 and q2 matrices for double q learning
q_a1 = np.zeros_like(r1)
q_a2 = np.zeros_like(r1)
alpha, gamma, epsilon = 0.5, 0.5, 0.5
results_a1, q_a1, q_a2 = run_advanced_case(alpha, gamma, epsilon, r1, q1, q2, myAgent)











