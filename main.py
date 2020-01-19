from board import Board
from players import MMPlayer, ABPlayer, ManualPlayer, RandomPlayer, MCTSPlayer, Node
import model
import settings
import time

class Game:

    # Al empezar a jugar le damos los dos tipos de jugadores que vamos a usar
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    # Ciclo principal de la partida
    def play(self, hayJugadorManual):

        board = Board()
        # Para saber quien mueve ahora
        P1Turn = True
        if self.player1.heuristic in (0,99) or self.player2.heuristic in (0,99):
            # simulaciones = int(input('Introduce cuantas simulaciones quieres que haga el algoritmo MCTS para cada decision: '))
            simulaciones = 1000
            node = Node(board = board)
            timeout = 3

        while(True):

            # Busca el movimiento de a quien le toque
            if P1Turn:
                #print ('Turno del jugador 1')
                if self.player1.heuristic in (0, 99):
                    node, move = self.player1.findMove(board, simulaciones, node, timeout, self.player1.nn)
                elif self.player1.heuristic == 666:
                    move = self.player1.findMove(board)
                else:
                    move = self.player1.findMove(board, self.player1.heuristic)
                #print('El jugador 1 ha decidido poner ficha en la columna', move)
            else:
                #print('Turno del jugador 2')
                if self.player2.heuristic in (0, 99):
                    node, move = self.player2.findMove(board, simulaciones, node, timeout, self.player2.nn)
                elif self.player2.heuristic == 666:
                    move = self.player2.findMove(board)
                else:
                    move = self.player2.findMove(board, self.player2.heuristic)

                #print('El jugador 2 ha decidido poner ficha en la columna', move)

            # Hace el movimiento que se ha decidido y lo enseña
            board.makeMove(move)
            if hayJugadorManual:
                board.printBoard()
                board.convertIntoTable()

            if (self.player1.heuristic == 99 and P1Turn == True) or (self.player2.heuristic == 99 and P1Turn == False):
                node = moveToChildNode(node, board, move)

            #Mira el estado actual de la partida
            gameState = board.isTerminal()
            if gameState == 0:
                '''print("Empate")'''
            elif gameState == 1:
               '''print("Gana el Jugador 1, de tipo ")'''
            elif gameState == 2:
                '''print("Gana el jugador 2, de tipo ")'''
            else:
                P1Turn = not P1Turn

            if gameState in (0,1,2):
                #board.printBoard()
                return gameState


# Nos movemos al nodo hijo sobre el que hemos efectuado la accion en el MCTS y sera nuestro nuevo root node
def moveToChildNode(node, board, move):
    for childnode in node.childNodes:
        if childnode.move == move:
            return childnode
    return Node(board=board)


if __name__ == "__main__":
    print('Empieza la partida')

    # numRepeticiones=input("Introduce cuantas veces quieres que se realice la partida entre los jugadores escogidos: ")
    red = 1
    numRepeticiones = 1
    dephts = [1]

    stats = []
    times = []

    # Al crear un juego decidimos que clase de jugador vamos a usar

    for depht in dephts:
        statsOneDepht = []
        start = time.time()
        for i in range(int(numRepeticiones)):
            # Al crear un Game el tercer parametro indica que clase de jugador sera
            # 1,2,11,12 indica un jugador AB con la heuristica correspondiente
            # 666 indica jugador random
            # 99 indica jugador MCTS
            # 0 indica jugador con red

            # game = Game(ABPlayer(5, True, 1), ABPlayer(5,False, 11))
            game = Game(RandomPlayer(0, True, 666, None), MCTSPlayer(0, False, 99, None))
            # game = Game(ABPlayer(depht, True, 11), MCTSPlayer(0, False, 99))
            board = Board()
            if red == 1:
                best_NN = model.Residual_CNN(settings.REG_CONST, settings.LEARNING_RATE, (2,board.HEIGHT, board.WIDTH)
                                             ,   board.HEIGHT*board.WIDTH, settings.HIDDEN_CNN_LAYERS)
                temp = best_NN.read('NNs/')
                best_NN.model.set_weights(temp.get_weights())
                game = Game(ABPlayer(depht, True, 11, None), MCTSPlayer(0, False, 0, best_NN))

            winner = game.play(0)
            statsOneDepht.append(winner)

        print (time.time() - start)
        print ('Fin ejecucion')
        print ('Victorias del random: ', statsOneDepht.count(1))
        print ('Victorias del MCTS: ', statsOneDepht.count(2))
        stats.append((depht, statsOneDepht.count(1), statsOneDepht.count(2), time.time() - start))
    print (stats)
    print ('Profundidad AB, Victorias AB, Victorias MCTS, Tiempo')
