import pygame
import numpy as np
from pygame import time
import sys
import math
from random import randint

def think(inputs, syn0, syn1):
	# Pass inputs through our neural network.
	result = nonlin(np.dot(inputs, syn0))
	return nonlin(np.dot(result, syn1))

def nonlin(x, deriv=False):
	if(deriv==True):
		return (x*(1-x))
		
	return 1/(1+np.exp(-x))


#input data
# [ballup, balldown, above, on the paddle, below]
x = np.array(
[
[1,0,1,0,0],
[1,0,0,1,0],
[1,0,0,0,1],
[0,1,1,0,0],
[0,1,0,1,0],
[0,1,0,0,1]

]

)
# [up, down]
y = np.array(
[
[1,0],
[1,0],
[0,1],
[1,0],
[0,1],
[0,1]

]
)

#seed
#np.random.seed(1)




#synapses

syn0 = 2*np.random.random((5,7)) - 1
syn1 = 2*np.random.random((7,2)) - 1


#training





def main(x,y,syn0,syn1):
	pygame.init()
	pygame.font.init()
	size = width, height = 500, 400
	ballspeed = [1, 1]
	paddlestart = [width -64, randint(0,height-256)]
	black = 0, 0, 0
	reset_ball = False
	screen = pygame.display.set_mode(size)
	score = 0
	missed = 0
	ball = pygame.image.load("ball.bmp")
	ballrect = ball.get_rect()
	#paddle = pygame.image.load("leftpaddle.bmp")
	paddle = pygame.image.load("smallpaddle.bmp")
	padrect = paddle.get_rect()
	padrect = padrect.move(paddlestart)
	padspeed = [0,1]
	prcnt = 0
	iters = 0
	lastscore = 0
	font = pygame.font.SysFont('Bitstream Vera Sans Mono', 20)
	
	
	while 1:
		for event in pygame.event.get():
			if event.type == pygame.QUIT: sys.exit(0)
		
		if reset_ball == True:
			ballrect = ball.get_rect()
			ballrect = ballrect.move((randint(0,height-16),randint(0,width-256)))
			ballspeed = [1,1*math.pow(-1,randint(0,1))]
			reset_ball = False
		
		ballspeed,ballrect,score,missed,reset_ball,syn0,syn1,prcnt,iters = move_ball(iters,prcnt,ballrect,ballspeed,width,height,padrect,score,missed,reset_ball,x,y,syn0,syn1)
		
		padrect = move_paddle(paddle, padrect, padspeed,ballrect,ballspeed, width, height)
		if (iters%100)==0:
			lastscore = score
		text = font.render('iterations: '+str(iters),False,(0,255,0))
		Score = font.render('score: '+str(score)+'% out of 100',False,(0,0,255))
		LastScore = font.render('last score of 100: '+str(lastscore)+'%',False,(255,0,0))
		screen.fill(black)
		screen.blit(text,(0,0))
		screen.blit(Score,(200,0))
		screen.blit(LastScore,(200,30))
		screen.blit(ball, ballrect)
		screen.blit(paddle,padrect)
		
		
		#time.delay(2)
		
		pygame.display.flip()

def get_action(ballrect, ballspeed, padrect):
	inputs = [0,0,0,0,0]
	if ballspeed[1]>0:
		inputs[0] = 1
	else:
		inputs[1] = 1
	
	if ballrect.top>padrect.bottom and ballrect.bottom>padrect.top:
		inputs[2] = 1
	
	if ballrect.top>padrect.bottom and ballrect.bottom<padrect.top:
		inputs[3] = 1
	
	if ballrect.top<padrect.bottom and ballrect.bottom<padrect.top:
		inputs[4] = 1
	
	return inputs

def move_ball(iters,prcnt,ballrect,ballspeed,width,height,padrect,score,missed,reset_ball,x,y,syn0,syn1):
	ballr = ballrect.move(ballspeed)
	if ballr.right > width:
		missed += 1
		print 'missed: '+str(missed)
		prcnt = score/(score+missed)*100
		iters = iters + 1
		if iters >= 100:
			if score==0:
				missed -= 1
			else:
				score -= 1
		
		reset_ball = True
		l0 = x
		l1 = nonlin(np.dot(l0,syn0))
		l2 = nonlin(np.dot(l1,syn1))
		
		#backpropagation
		l2_error = y - l2
		
		
		#calculate delta
		l2_delta = l2_error*nonlin(l2, deriv=True)*.5
		
		l1_error = l2_delta.dot(syn1.T)
		
		l1_delta = l1_error*nonlin(l1, deriv=True)*.5
		
		#print 'Error: '+str(l1_error)
		#update synapses
		syn1+= l1.T.dot(l2_delta)
		syn0+= l0.T.dot(l1_delta)
		#print 'missed:'+str(missed)
	#collision with the walls
	if ballr.left < 0 or ballr.right > width:
		ballspeed[0] = -ballspeed[0]
		if ballr.left < 0 :
			score +=1
			#print 'score'+str(score)
			iters = iters + 1
			prcnt = score/(score+missed)*100
			print 'correct:'+ str(prcnt)+'%'
			if iters >= 100:
				if missed==0:
					score -= 1
				else:
					missed -= 1
			
	if ballr.top < 0 or ballr.bottom > height:
		ballspeed[1] = -ballspeed[1]
	
	#collision with the paddle
	if ballr.colliderect(padrect):
		#if ballr.left<padrect.left and ballr.right<(padrect.left+1):
		ballspeed[0] = -ballspeed[0]
		
		if ballr.top<padrect.bottom:
			ballspeed[1] = -ballspeed[1]
		if ballr.bottom>padrect.top:
			ballspeed[1] = -ballspeed[1]
	
	
	ballrect = ballr
	return ballspeed,ballrect,score,missed,reset_ball,syn0,syn1,prcnt,iters

def move_paddle(paddle, padrect, padspeed,ballrect,ballspeed, width, height):
	action = think(get_action(ballrect, ballspeed, padrect),syn0,syn1)
	guess = np.random.rand(1)
	if action[0]-guess[0]>0:
		padrec = padrect.move(padspeed)
	else:
		padrec = padrect.move((-padspeed[0],-padspeed[1]))
	padrect = padrec
	if padrect.bottom>height:
		padrec = padrect.move((0,-2))
	if padrect.top<0:
		padrec = padrect.move((0,2))
	padrect = padrec
	
	
	return padrect
	



main(x,y,syn0,syn1)
