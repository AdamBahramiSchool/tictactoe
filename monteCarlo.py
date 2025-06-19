
import copy
import random
import time
import sys
import math
from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support
# Total 40 pt for monteCarlo.py
class MCTS:
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild

    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

        
    def monteCarloPlayer(self, timelimit = 4):
        """Entry point for Monte Carlo tree search"""
        start = time.perf_counter()
        end = start + timelimit
        childNode=None
        """
        Use time.perf_counter() to apply iterative deepening strategy.
         At each iteration we perform 4 stages of MCTS: 
         SELECT, 
         EXPEND, 
         SIMULATE, 
         and BACKUP. 
         
         Once time is up use getChildWithMaxScore() to pick the node to move to
        """
        while time.perf_counter()<end:
            # select phase
            startNode=self.selectNode(self.root)
            childNode=self.expandNode(startNode)
            leaf = childNode if childNode else startNode
            result = self.simulateRandomPlay(leaf)
            self.backPropagation(leaf, result)
        
        print("MCTS: your code goes here. 10pt.")

        winnerNode = self.root.getChildWithMaxScore()
        assert(winnerNode is not None)
        return winnerNode.state.move


    """SELECT stage function. walks down the tree using findBestNodeWithUCT()"""
    def selectNode(self, nd):
        currentNode=nd
        while currentNode.children:
            currentNode=self.findBestNodeWithUCT(currentNode)
        return currentNode
        print("Your code goes here 5pt.")


    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. 
        Parse nd's children and use uctValue() to collect ucts 
        for the children.....
        Make sure to handle the case when uct value of 2 or more children
        nodes are the same."""
        uct_values=[]
        parent_visit=nd.visitCount
        uctVal=None
        childUCT = []
        for child in nd.children:
            childUCTval=self.uctValue(parent_visit,child.winScore,child.visitCount)
            if childUCTval>uctVal or uctVal is None:
                uctVal=childUCTval
                childUCT=[child]
            elif childUCTval==uctVal:
                childUCT.append(child)
        if len(childUCT) == 1:
            return childUCT[0]
        elif len(childUCT)<1:
            return None
        return random.choice(childUCT) 
        
        print("Your code goes here 5pt.")
        return None


    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """compute Upper Confidence Value for a node"""
        # (win_score / visitCount) + exploreFactor * sqrt(ln(parentVisit) / visitCount)
        if nodeVisit == 0:
            return float('inf')
        return (nodeScore/nodeVisit) + self.exploreFactor *math.sqrt(math.log(parentVisit)/nodeVisit)
        print("Your code goes here 3pt.")
        pass

   
   
    def expandNode(self, nd):
        """generate the child nodes for node nd. For convenience, generate
        all the child nodes and attach them to nd."""
        state = nd.state
        print("Your code goes here 5pt.")
        for action in self.game.actions(state):
            new_state=self.game.result(state,action)
            child_node=self.Node(new_state,nd)
            nd.children.append(child_node)
        return nd.children[0] if nd.children else None
    

    """SIMULATE stage function"""
    def simulateRandomPlay(self, nd):
        """
        This function retuns the result of simulating off of node nd a 
        termination node, and returns the winner 'X' or 'O' or 0 if tie.
        Note: pay attention nd may be itself a termination node. Use compute_utility 
        to check for it.
        """
        print("Your code goes here 7pt.")
        state = copy.deepcopy(nd.state)

        # Check if nd is already a terminal node
        if self.game.terminal_test(state):
            root_player = self.root.state.to_move
            utility = self.game.utility(state, root_player)
            if utility > 0:
                return root_player
            elif utility < 0:
                return 'O' if root_player == 'X' else 'X'
            else:
                return 0

        # Otherwise simulate random play
        while not self.game.terminal_test(state):
            move = random_player(self.game, state)
            if move is None:
                break
            # while loop will break also if current state is terminal
            state = self.game.result(state, move)

        # Final evaluation, 
        root_player = self.root.state.to_move
        utility = self.game.utility(state, root_player)
        if utility > 0:
            return root_player
        elif utility < 0:
            return 'O' if root_player == 'X' else 'X'
        else:
            return 0



    def backPropagation(self, nd, winningPlayer):
        """propagate upword to update score and visit count from
        the current leaf node to the root node."""
        while nd is not None:
            nd.visitCount += 1
            # nd.state.to_move is the player who is about to make the move. so if he is not the winning player, the last player 
            # who made the move is the winning player.
            # So we increment the winScore for the last player who made the move.
            if nd.state.to_move != winningPlayer:  
                nd.winScore += 1
            nd = nd.parent
        print("Your code goes here 5pt.")


