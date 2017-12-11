import numpy as np
from math import floor
import random
import copy
import heapq

# state representation
GRID_WIDTH = 12
GRID_HEIGHT = 12
PADDLE_HEIGHT = 0.2
PADDLE_X = 1
NUMBER_EXPLORED_CONST = 25

DISCOUNT_FACTOR = 0.3
LR_CONSTANT = 20

def discrete_x(location,item_height = 0):
	if floor(location) == 1.0:
		return GRID_WIDTH - 1
	else: 
		return int(floor(GRID_WIDTH*location))

def discrete_y(location,item_height = 0):
	if floor(location+item_height) == 1.0:
		return GRID_HEIGHT - 1
	else: 
		return int(floor(GRID_HEIGHT*location / (1 - item_height)))

def discrete_paddle_y(location):
	return discrete_y(location,item_height=PADDLE_HEIGHT)

def discrete_x_velocity(x_velocity):
	if (x_velocity<0):
		return -1
	elif (x_velocity==0):
		return 0
	elif (x_velocity>0):
		return 1

def discrete_y_velocity(y_velocity):
	if (y_velocity>-0.015 and y_velocity<0.015):
		return 0
	elif (y_velocity>0.015):
		return 1
	elif (y_velocity<=0.015):
		return -1

def discretizing_values(ball_x,ball_y,velocity_x,velocity_y, paddle_y):
	return (discrete_x(ball_x),discrete_y(ball_y), discrete_x_velocity(velocity_x), discrete_y_velocity(velocity_y), discrete_paddle_y(paddle_y))


class TestingState:

	def __init__(self,gamestate):
		self.ball_x, self.ball_y, self.velocity_x, self.velocity_y, self.paddle_y=discretizing_values(gamestate.ball_x,gamestate.ball_y,gamestate.velocity_x,gamestate.velocity_y, gamestate.paddle_y)
		self.terminal = gamestate.is_terminal()

	def vector(self):
		return (self.ball_x,self.ball_y,self.velocity_x,self.velocity_y,self.paddle_y)

class GameState:
	initial_vector = (0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2)

	def __init__(self, ball_x=0.5, ball_y=0.5, velocity_x=0.03,velocity_y=0.01, paddle_y=0.5 - PADDLE_HEIGHT / 2):
		self.ball_x = ball_x
		self.ball_y = ball_y
		self.velocity_x = velocity_x
		self.velocity_y = velocity_y
		self.paddle_y = paddle_y

	def reset(self):
		self.ball_x, self.ball_y, self.velocity_x, self.velocity_y, self.paddle_y = self.initial_vector

	def is_terminal(self):
		if self.ball_x >= 1 and (self.ball_y < self.paddle_y or self.ball_y > self.paddle_y + PADDLE_HEIGHT):
			return True
		return False

	def check_hit(self):
		if self.ball_x >= 1 and self.ball_y >= self.paddle_y and self.ball_y <= self.paddle_y + PADDLE_HEIGHT:
			return True
		return False

	def move_ball(self):
		self.ball_x += self.velocity_x
		self.ball_y += self.velocity_y

	def move_paddle(self, direction):
		new_y = self.paddle_y
		if direction == 0:
			new_y -= 0.04
		elif direction == 2:
			new_y += 0.04

		new_y = max(0,new_y)
		new_y = min(1-PADDLE_HEIGHT,new_y)
		self.paddle_y = new_y

class FunctionMatrix:

	def __init__(self):
		self.function = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2, 3, 12, 3))
		self.terminal = [0,0,0]

	def update(self,ms,action,val):
		if ms.terminal:
			self.terminal[action] += val
		else:
			self.function[ms.vector()][action] += val

	def get(self,ms,action):
		if ms.terminal:
			return self.terminal[action]
		else:
			return self.function[ms.vector()][action]

class QModel:
	def __init__(self):
		self.n_function = FunctionMatrix()
		self.q_function = FunctionMatrix()
		self.gamestate = GameState()
		self.prev_state = None
		self.prev_action = None
		self.prev_reward = None

	def train(self, iterations):
		for i in range(iterations):
			score = self.training_step()
			if (i % 1000 == 0):
				print("Num iterations: " + str(i))
		print ("Num iterations: " + str(iterations))

	def test(self, iterations):
		score_sum = 0.0
		for i in range(iterations):
			score = self.training_step()
			score_sum += score
		return float(score_sum) / iterations

	# This is for the bouncing mechanism
	def bounce(self):
		paddle_bounce = False
		terminal_state = False
		if self.gamestate.ball_y < 0: # ball is off top of screen - check
			self.gamestate.ball_y = -self.gamestate.ball_y
			self.gamestate.velocity_y = -self.gamestate.velocity_y
		elif self.gamestate.ball_y > 1: # ball if off bottom of screen - check
			self.gamestate.ball_y = 2 - self.gamestate.ball_y
			self.gamestate.velocity_y = -self.gamestate.velocity_y
		elif self.gamestate.ball_x < 0: # ball is off left edge of screen - check
			self.gamestate.ball_x = -self.gamestate.ball_x
			self.gamestate.velocity_x = -self.gamestate.velocity_x
		elif self.gamestate.ball_x >= 1: # check right edge

			# bounce of paddle
			if self.gamestate.ball_x >= 1 and self.gamestate.ball_y >= self.gamestate.paddle_y and \
				self.gamestate.ball_y <= self.gamestate.paddle_y + PADDLE_HEIGHT:

				ball_x = 2 * PADDLE_X - self.gamestate.ball_x

				U = random.uniform(-0.015, 0.015)
				V = random.uniform(-0.03, 0.03)

				velocity_x = -self.gamestate.velocity_x + U
				velocity_y = self.gamestate.velocity_y + V

				while abs(velocity_x) >= 1 or abs(velocity_y) >= 1 or abs(velocity_x) <= 0.03:
					U = random.uniform(-0.015, 0.015)
					V = random.uniform(-0.03, 0.03)

					velocity_x = -self.gamestate.velocity_x + U
					velocity_y = self.gamestate.velocity_y + V

				self.gamestate.velocity_x = velocity_x
				self.gamestate.velocity_y = velocity_y
				paddle_bounce = True

			# not bounce - terminal state
			else:
				terminal_state = True

		return (paddle_bounce,terminal_state)

	def exploration_function(self,ms):
		candidates = []
		max_score = -1000
		best_action = -1
		for i in range(3):
			if self.n_function.get(ms,i) < NUMBER_EXPLORED_CONST:
				candidates.append(i)
			else:
				if self.q_function.get(ms,i) > max_score:
					best_action = i
					max_score = self.q_function.get(ms,i)
		if len(candidates) > 0:
			return random.choice(candidates)
		return best_action

	def q_learning_agent(self,ms,R_prime):
		action = self.exploration_function(ms)
		if not self.prev_state is None:
			self.update_q(ms)
		self.prev_state = ms
		self.prev_action = action
		self.prev_reward = R_prime
		return action

	def update_q(self,ms):
		self.n_function.update(self.prev_state,self.prev_action,1)
		lr = float(LR_CONSTANT) / (LR_CONSTANT + self.n_function.get(self.prev_state,self.prev_action))
		# print [self.q_function.get(ms,i) for i in xrange(3)]
		qnewsa = max([self.q_function.get(ms,i) for i in range(3)])
		update_sum = self.prev_reward + DISCOUNT_FACTOR*(qnewsa - self.q_function.get(self.prev_state,self.prev_action))
		q_increment_value = lr*update_sum
		self.q_function.update(self.prev_state,self.prev_action,q_increment_value)

	def training_step(self):
		self.gamestate.reset()
		self.prev_state = None
		self.prev_action = None
		self.prev_reward = None
		score = 0
		prev_terminal = False
		next_action = None
		num_iterations = 0
		while not prev_terminal:
		
			self.gamestate.move_ball()
			self.gamestate.move_paddle(next_action)

			current_state = TestingState(self.gamestate)

			paddle_bounce, terminal_state = self.bounce()

			if self.prev_state is not None:
				prev_terminal = self.prev_state.terminal

			R_prime = 0
			if paddle_bounce: 
				R_prime = 1
				score += 1
			elif terminal_state: 
				R_prime = -1

			# Q-learning
			next_action = self.q_learning_agent(current_state, R_prime)

			num_iterations += 1

		return score

model = QModel()
model.train(100000)
for i in range(10):
	avg_hits = model.test(1000)
	print ("Avg. hits(" +str(i+1)+")"+"=" + str(avg_hits))




