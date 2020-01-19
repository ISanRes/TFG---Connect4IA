import math
import random
from board import Board
import time
import settings
import numpy as np
import model

# En este fichero se definen los tipos de jugadores que puede haber. Todos heredan de una clase central,
# pero hay 6 distintos: Manual, Random, Minimax (MM), Minimax Alfabeta (AB) y MCTS
# La clase MCTS se usa tanto para el puro como para el que incluye la red neuronal (AZ)

# Todos los jugadores tienen la clase base, que les calcula la heuristica
# Cada jugador tiene una clase distinta porque utiliza un metodo distinto para decidir el proximo movimiento
class Player:

    def __init__(self, depthLimit, isPlayerOne, heuristic, nn):

        self.isPlayerOne = isPlayerOne
        self.depthLimit = depthLimit
        self.heuristic = heuristic
        self.nn = nn

    # Aqui se definen las heuristicas. Si hay dos cifras la primera indica el tipo y la segunda la variacion
    # (la 1 y la 11 son la misma pero con una pequeña diferencia, la misma que hay entre la 2 y la 21
    
    # Calcula la heuristica dando mas peso a las cadenas con mas piezas juntas, dando 10 veces mas valor a una serie de 3 que a una de 2 y 
        # 100 veces mas valor a una de 4 que a una de 3 (para asegurar que si hay una serie de 3 se escoja ese hijo
    # Se mira en las 4 direcciones posibles: horizontal, vertical y las dos diagonales
    def heuristic1(self, board):
        h = 0
        tablero = board.board

        for col in range(0, board.WIDTH):
            for row in range(0, board.HEIGHT):
                # Comprobacion horizontal
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    if tablero[col][row] == tablero[col + 1][row] == 0:
                        h += 5
                    elif tablero[col][row] == tablero[col + 1][row] == 1:
                        h -= 5
                        
                    if tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == 0:
                        h += 10
                    elif tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == 1:
                        h -= 10
                        
                    if tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == tablero[col + 3][row] == 0:
                        h += 1000
                    elif tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == tablero[col + 3][row] == 1:
                        h -= 1000
                except IndexError:
                    pass

                # Comprobacion vertical
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    # add player one vertical streaks to heur
                    if tablero[col][row] == tablero[col][row + 1] == 0:
                        h += 5
                    elif tablero[col][row] == tablero[col][row + 1] == 1:
                        h -= 5
                        
                    if tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == 0:
                        h += 10
                    elif tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == 1:
                        h -= 10
                        
                    if tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == tablero[col][row + 3] == 0:
                        h += 1000
                    elif tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == tablero[col][row + 3] == 1:
                        h -= 1000
                except IndexError:
                    pass

                # Comprobacion diagonal hacia la derecha, buscamos desde abajo hacia arriba
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    #Falta añadir que no cuente las veces en las que nos salimos del tablero por la derecha
                    if not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == 0:
                        h += 5
                    elif not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == 1:
                        h -= 5

                    if not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == tablero[col + 2][row + 2] == 0:
                        h += 10
                    elif not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == tablero[col + 2][row + 2] == 1:
                        h -= 10

                    if not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == tablero[col + 2][row + 2] \
                            == tablero[col + 3][row + 3] == 0:
                        h += 1000
                    elif not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == tablero[col + 2][row + 2] \
                            == tablero[col + 3][row + 3] == 1:
                        h -= 1000
                except IndexError:
                    pass

                # Comprobacion diagonal hacia la izquierda, buscamos desde arriba hacia abajo
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    # add  player one streaks
                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == 0:
                        h += 5
                    elif not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == 1:
                        h -= 5

                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][row - 2] == 0:
                        h += 10
                    elif not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][row - 2] == 1:
                        h -= 10

                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][row - 2] \
                            == tablero[col + 3][row - 3] == 0:
                        h += 1000
                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][row - 2] \
                            == tablero[col + 3][row - 3] == 1:
                        h -= 1000
                except IndexError:
                    pass
        return h

    # Igual que la 1 pero da mas peso a las columnas centrales, asi en caso de empate de cadenas juntas escogeremos las columnas que dan mas juego
    def heuristic11(self, board):
        h = 0
        tablero = board.board
        for col in range(0, board.WIDTH):
            for row in range(0, board.HEIGHT):
                # Comprobacion horizontal
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    if tablero[col][row] == tablero[col + 1][row] == 0:
                        h += 10
                    elif tablero[col][row] == tablero[col + 1][row] == 1:
                        h -= 10

                    if tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == 0:
                        h += 100
                    elif tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == 1:
                        h -= 100

                    if tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == tablero[col + 3][
                        row] == 0:
                        h += 10000
                    elif tablero[col][row] == tablero[col + 1][row] == tablero[col + 2][row] == tablero[col + 3][
                        row] == 1:
                        h -= 10000
                except IndexError:
                    pass

                # Comprobacion vertical
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    # add player one vertical streaks to heur
                    if tablero[col][row] == tablero[col][row + 1] == 0:
                        h += 10
                    elif tablero[col][row] == tablero[col][row + 1] == 1:
                        h -= 10

                    if tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == 0:
                        h += 100
                    elif tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == 1:
                        h -= 100

                    if tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == tablero[col][
                        row + 3] == 0:
                        h += 10000
                    elif tablero[col][row] == tablero[col][row + 1] == tablero[col][row + 2] == tablero[col][
                        row + 3] == 1:
                        h -= 10000
                except IndexError:
                    pass

                # Comprobacion diagonal hacia la derecha, buscamos desde abajo hacia arriba
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    # Falta añadir que no cuente las veces en las que nos salimos del tablero por la derecha
                    if not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == 0:
                        h += 10
                    elif not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == 1:
                        h -= 10

                    if not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == \
                            tablero[col + 2][row + 2] == 0:
                        h += 100
                    elif not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == \
                            tablero[col + 2][row + 2] == 1:
                        h -= 100

                    if not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == \
                            tablero[col + 2][row + 2] \
                            == tablero[col + 3][row + 3] == 0:
                        h += 10000
                    elif not row + 3 > board.HEIGHT and tablero[col][row] == tablero[col + 1][row + 1] == \
                            tablero[col + 2][row + 2] \
                            == tablero[col + 3][row + 3] == 1:
                        h -= 10000
                except IndexError:
                    pass

                # Comprobacion diagonal hacia la izquierda, buscamos desde arriba hacia abajo
                # Positivas las buenas posiciones para el jugador 0 y negativas las buenas para el jugador 1
                try:
                    # add  player one streaks
                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == 0:
                        h += 10
                    elif not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == 1:
                        h -= 10

                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][
                        row - 2] == 0:
                        h += 100
                    elif not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][
                        row - 2] == 1:
                        h -= 100

                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][row - 2] \
                            == tablero[col + 3][row - 3] == 0:
                        h += 10000
                    if not row - 3 < 0 and tablero[col][row] == tablero[col + 1][row - 1] == tablero[col + 2][row - 2] \
                            == tablero[col + 3][row - 3] == 1:
                        h -= 10000
                except IndexError:
                    pass

        if board.lastMove[1] == 3 and board.lastMove[0] == 0:
            h =+ 5
        if board.lastMove[1] == 3 and board.lastMove[0] == 1:
            h =- 5
        if board.lastMove[1] in (2,4) and board.lastMove[0] == 0:
            h =+ 3
        if board.lastMove[1] in (2,4) and board.lastMove[0] == 1:
            h =- 3
        if board.lastMove[1] in (1,5) and board.lastMove[0] == 0:
            h =+ 1
        if board.lastMove[1] in (1,5) and board.lastMove[0] == 1:
            h =- 1
        return h

    # Cuenta, para cada casilla del tablero, cuantas posibilidades de cadena de 4 le quedan aun.
    # Si no hay ninguna ficha del contrario cerca una casilla tiene 4 posibilidades. Si se le pone por ejemplo
    # una ficha del contrario dos casillas a la derecha perdemos una posibilidad, ya que no podremos hacer 4 en horizontal
    def heuristic2(self, board):
        h = 0

        maxRowUsed = 0
        for col in range(board.WIDTH):
            if len(board.board[col]) > maxRowUsed:
                maxRowUsed = len(board.board[col])

        for col in range(board.WIDTH):
            for row in range(maxRowUsed):
                h =+ self.countPosWins(col, row, board, 1)
                h =- self.countPosWins(col, row, board, 0)
        return h

    # Igual que la 2 pero da mas peso a las columnas centrales, asi en caso de empate de cadenas juntas escogeremos las columnas que dan mas juego
    def heuristic21(self, board):
        h = 0

        maxRowUsed = 0
        for col in range(board.WIDTH):
            if len(board.board[col]) > maxRowUsed:
                maxRowUsed = len(board.board[col])

        for col in range(board.WIDTH):
            for row in range(maxRowUsed):
                h =+ self.countPosWins(col, row, board, 1)
                h =- self.countPosWins(col, row, board, 0)

        if board.lastMove[1] == 3 and board.lastMove[0] == 0:
            h =+ 5
        if board.lastMove[1] == 3 and board.lastMove[0] == 1:
            h =- 5
        if board.lastMove[1] in (2,4) and board.lastMove[0] == 0:
            h =+ 3
        if board.lastMove[1] in (2,4) and board.lastMove[0] == 1:
            h =- 3
        if board.lastMove[1] in (1,5) and board.lastMove[0] == 0:
            h =+ 1
        if board.lastMove[1] in (1,5) and board.lastMove[0] == 1:
            h =- 1
        return h

    # Se usa en la heuristica 2, cuenta para una casilla concreta el numero de posibles 4 en ralla que aun podemos hacer
    def countPosWins(self, col, row, board, otherPlayer):
        count = 0
        tablero = board.convertIntoTable()

        # Comprobar si se puede ganar en la vertical
        if otherPlayer not in board.board[col]:
            count =+ 1

        # Comprobar si se puede ganar en la horizontal
        if col < (board.WIDTH -1 - 2):
            fila = (tablero[col][row], tablero[col+1][row], tablero[col+2][row], tablero[col+3][row])
            if otherPlayer not in fila:
                count =+ 1

        # Comprobar si se puede ganar en la diagonal hacia la derecha
        if col < (board.WIDTH -1 - 2) and row < (board.HEIGHT -1 -2):
            diagDerecha = (tablero[col][row], tablero[col+1][row+1], tablero[col+2][row+2], tablero[col+3][row+3])
            if otherPlayer not in diagDerecha:
                count += 1

        # Comprobar si se puede ganar en la diagonal hacia la izquierda
        if col > 2 and row < (board.HEIGHT - 1 - 2):
            diagIzquierda = (tablero[col][row], tablero[col-1][row+1], tablero[col-2][row+2], tablero[col-3][row+3])
            if otherPlayer not in diagIzquierda:
                count += 1

        return count

# Jugador manual, para jugar contra los otros jugadores definidos
class ManualPlayer(Player):
    def findMove(self, board):
        col = input("Colocar una  " + ('O' if self.isPlayerOne else 'X') + " en la columna: ")
        wrongNumber = int(col) > 7 or int(col) < 1

        while wrongNumber:
            col = input("Por favor, introduce un numero de columna entre 1 y 7 para poner ficha: ")
            wrongNumber = int(col) > 7 or int(col) < 1
        col = int(col) - 1
        return col

# Jugador aleatorio, hace un movimiento totalmente al azar cada vez que tiene que escoger
class RandomPlayer(Player):
    # escoge al azar un movimiento posible
    def findMove(self, board):
        suitableColumns = []

        for i in range(board.WIDTH):
            if len(board.board[i]) < 7:
                suitableColumns.append(i)

        return random.choice(suitableColumns)

# Jugador miximax
class MMPlayer(Player):
    def __init__(self, depth, isPlayerOne, nn):
        super().__init__(depth, isPlayerOne, nn)

    # Encuentra el mejor movimiento
    def findMove(self, board):
        if board.numMoves == 0:
            move = random.choice(range(7))
        else:
            move = self.miniMax(board, self.depthLimit, self.isPlayerOne)

        # print(self.isPlayerOne, "ha puesto ficha en la columna ", move)
        return move

    # Aplica el algoritmo
    def miniMax(self, board, depth, player):

        if depth == 0: #Si no hay profundidad devolvemos la evaluacion actual para realizar el minimax
            return self.heuristic(board)

        # lambda hace que shouldreplace pase a ser una funcion donde se comparara el valor que le pasemos con bestScore
        if player:
            bestScore = -math.inf
            shouldReplace = lambda x: x > bestScore
        else:
            bestScore = math.inf
            shouldReplace = lambda x: x < bestScore

        bestMove = -1

        children = board.children()
        # Para cada posible movimiento
        for child in children:
            # Columna en la que hemos metido la ficha, tablero que queda
            move, childboard = child

            temp = self.miniMax(childboard, depth-1, not player)
            if shouldReplace(temp):
                bestScore = temp
                bestMove = move
        return bestMove

# Jugador alfabeta
class ABPlayer(Player):

    def __init__(self, depth, isPlayerOne, heuristic, nn):
        super().__init__(depth, isPlayerOne, heuristic, nn)

    # Encuentra el mejor movimiento posible utilizando el algoritmo alfabeta
    def findMove(self, board, heuristic):

        # Los primeros movimientos son aleatorios
        if board.numMoves in (0,1):
            move = random.choice(range(7))
            move = self.alphaBeta(board, self.depthLimit, self.isPlayerOne, -math.inf, math.inf, heuristic)
        else:
            move = self.alphaBeta(board, self.depthLimit, self.isPlayerOne, -math.inf, math.inf, heuristic)
        return move

    # Aplica el algoritmo
    def alphaBeta(self, board, depth, player, alpha, beta, heuristic):

        if depth == 0:  # Si no hay profundidad calculamos la puntuacion de este tablero usando la heuristica que toque
            if heuristic == 1:
                return self.heuristic1(board)
            elif heuristic == 11:
                return self.heuristic11(board)
            elif heuristic == 2:
                return self.heuristic2(board)
            elif heuristic == 21:
                return self.heuristic21(board)

        # lambda hace que shouldreplace pase a ser una funcion donde se comparara el valor que le pasemos con bestScore
        if player:
            bestScore = -math.inf
            shouldReplace = lambda x: x > bestScore
        else:
            bestScore = math.inf
            shouldReplace = lambda x: x < bestScore

        bestMove = -1

        children = board.children()
        for child in children:
            move, childboard = child
            temp = self.alphaBeta(childboard, depth-1, not player, alpha, beta, heuristic)

            if shouldReplace(temp):
                bestScore = temp
                bestMove = move
            if player:
                alpha = max(alpha, temp)
            else:
                beta = min(beta, temp)
            if alpha >= beta:
                break
        return bestMove

# Jugador MCTS, se usa tanto para el MCTS puro como para el de la red neuronal
class MCTSPlayer(Player):

    def __init__(self, depth, isPlayerOne, heuristic, nn):
        super().__init__(depth, isPlayerOne, heuristic, nn)

    # itermax decide cuantas simulaciones hara para cada decision
    # timeout es para capar y que las simulaciones no se vayan de madre, si superamos el tiempo
    # dejara de hacer simulaciones independientemente de cuantas le queden
    def findMove(self, board, itermax, currentNode=None, timeout=2000, nn = None):

        rootnode = Node(board=board)
        # Si nos pasan un nodo ya empezado sera nuestro root
        if currentNode is not None:
            rootnode = currentNode

        # para vigilar que no nos pasemos de tiempo
        start = time.clock()

        # empiezan las simulaciones
        for i in range(itermax):
            node = rootnode
            tablero = board.copiarTablero()

            # Fase 1: Seleccion
            # Si el nodo actual no es hoja seleccionamos el siguiente al que moverse
            while node.untriedMoves == [] and node.childNodes != []:
                # Seleccion para el MCTS puro, sin predicciones de la NN
                if nn == None:
                    node = node.selection(0, board)
                # Seleccion para el MCTS de AZ, usamos las predicciones de la NN
                else:
                    stats = self.get_preds(board, nn)
                    node = node.selection(1, board, stats)
                tablero.makeMove(node.move)

            # Fase 2: Expansion
            # Cuando hemos llegado a un nodo hoja hacemos la expansion de todos los hijos de ese nodo
            if node.untriedMoves != []:
                m = random.choice(node.untriedMoves)
                tablero.makeMove(m)
                node = node.expand(m, tablero)

            # Fase 3: Rollout o simulacion
            # Simulamos aleatoriamente la partida hasta el final
            while tablero.isTerminal() == -1:
                tablero.makeMove(random.choice(tablero.movimientosPosibles()))

            # Fase 4: Backpropagation
            # Actualizamos los valores de los nodos con lo que ha pasado en esta simulacion hasta llegar a la raiz

            # Esto nos dice -1 no ha acabado, 0 empate, 1-2 ha ganado ese jugador
            boardState = tablero.isTerminal()
            while node is not None:
                if (self.isPlayerOne == True and boardState == 1) or (self.isPlayerOne == False and boardState == 2):
                    #Hemos ganado
                    node.update(1)
                elif boardState == 0:
                    node.update(0.5)
                else:
                    node.update(0)
                node = node.parent

            # Vigilamos que no hayamos estado demasiado tiempo
            duration = time.clock() - start
            if duration > timeout: break

        # De todos los nodos miramos en cual hemos ganado mas veces y hacemos esa accion
        bestWinRate = lambda x: x.wins / x.visits
        sortedChildNodes = sorted(rootnode.childNodes, key=bestWinRate)[::-1]
        return rootnode, sortedChildNodes[0].move

    # Nos devuelve las predicciones de la red neuronal para el tablero actual
    def get_preds(self, board, nn):
        inputToModel = board.pasarAInputDelModelo()
        preds = nn.predict(inputToModel)
        logits_array = preds[1]

        logits = logits_array[0]

        allowedActions = board.movimientosPosibles()

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        # Devuelve las probabilidades en una lista de 42 posiciones, donde cada una representa
        # la posibilidad del movimiento 42 - (columna+1)*(fila+1) (por ejemplo, la accion 41 representa
        # poner una ficha en la columna 0 y fila 0
        return (probs)

# Clase que representa cada posible estado de la partida en terminos del algoritmo MCTS
class Node:

    # Cada nodo tiene asociado un tablero, un padre, un movimiento que es el que se ha realizado para llegar a el,
    # una lista de movimientos que puede realizar, una lista de hijos y el numero de visitas y victorias que ha tenido
    def __init__(self, move=None, parent=None, board=None):
        self.board = board.copiarTablero()
        self.parent = parent
        self.move = move
        self.untriedMoves = board.movimientosPosibles()
        self.childNodes = []
        self.wins = 0
        self.visits = 0

    # Devuelve el siguiente nodo a explorar, dependiendo de si es MCTS puro o MCTS de AZ usa una formula u otra
    # Corresponde a la Fase 1 del MCTS
    def selection(self, isAZ, board, pred = None):
        if isAZ:
            # esta formula se llama PUCT y esta extraida de la web
            # https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5

            ucb = lambda x: settings.CPUCT * pred[(42-((x.move + 1)*(len(board.board[x.move])) + 1))] * \
                            (np.sqrt(self.calculateOtherNodesVisits(x)) / (1 + x.visits))
        else:
            # esta formula se llama UCB1 y esta extraida del video
            # https://www.youtube.com/watch?v=UXW2yZndl7U
            ucb = lambda x: x.wins / x.visits + (2 * np.sqrt(np.log(self.visits) / x.visits))
        a = sorted(self.childNodes, key=ucb)[-1]
        return sorted(self.childNodes, key=ucb)[-1]

    # Abre todos los posibles hijos que puede tener un nodo y se usa cuando llegamos a un nodo hoja
    # Corresponde a la Fase 2 del MCTS, expansion
    def expand(self, move, state):
        child = Node(move=move, parent=self, board=state)
        self.untriedMoves.remove(move)
        self.childNodes.append(child)
        return child

    # Actualiza los valores de un nodo con lo que ha pasado al finalizar esa simulacion
    # Corresponde a la Fase 4 del MCTS, backpropagation
    def update(self, result):
        self.wins += result
        self.visits += 1

    # Calcula cuantas veces se han visitado todos los nodos que no son el actual
    # Se usa para calcular la metrica PUCT del MCTS AZ
    def calculateOtherNodesVisits(self, actualNode):

        nb = 0
        for node in self.childNodes:
            if node  != actualNode:
                nb =+ node.visits

        return nb








