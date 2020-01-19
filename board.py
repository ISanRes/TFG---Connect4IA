import numpy as np

# En este fichero se define la clase tablero, que contiene el estado actual de la partida y el tamaño,
# y las funciones relacionadas, como mover ficha o comprobar el estado actual de la partida

class Board(object):

    #Tamaño del tablero
    HEIGHT = 6
    WIDTH = 7

    def __init__(self, specificBoard=None):

        # board contiene las columnas del tablero, una lista para cada una,
        # por lo que la estructura del tablero es una lista de 7 listas.
        # numMoves es el numero de fichas que hay en el tablero
        # lastMove es el ultimo movimiento que se ha realizado y tiene la forma (jugador que lo ha hecho, columna)
        
        # Copiamos un tablero (se puede usar para forzar un tipo de tablero y estudiar como reacciona cada jugador a una situacion concreta)
        if (specificBoard):
            self.board = [list(col) for col in specificBoard.board]
            self.numMoves = specificBoard.numMoves
            self.lastMove = specificBoard.lastMove
        else:
            self.board = [[] for x in range(self.WIDTH)]
            self.numMoves = 0
            self.lastMove = None

    # Pone una pieza en el tablero
    def makeMove(self, column):
        # Decide de que jugador va a ser la pieza en funcion de los movimientos hechos, 0 para el jugador 1 y 1 para el 2
        piece = self.numMoves % 2
        self.lastMove = (piece, column)
        self.numMoves += 1
        self.board[column].append(piece)

    # Genera una lista de los tableros posibles a partir del tablero actual
    def children(self):
        children = []
        # Para cada columna, si no está completa creamos un tablero hijo y le añadimos una ficha en esa columna
        for row in range(7):
            if len(self.board[row]) < 6:
                child = Board(self)
                child.makeMove(row)
                children.append((row, child))
        return children

    # Devuelve el estado del juego: -1 no ha acabado, 0 empate, 1-2 ha ganado alguien
    def isTerminal(self):
        tablero = self.convertIntoTable()
        if self.isFull():
            return 0
        else:
            for i in range(0,self.WIDTH):
                for j in range(0,self.HEIGHT):
                    if (i < self.WIDTH - 3) and (tablero[i][j]  == tablero[i+1][j] == tablero[i+2][j] == tablero[i+3][j]) and tablero[i][j] in (0,1):
                        return tablero[i][j] + 1
                    
                    if (j < self.HEIGHT - 3) and (tablero[i][j]  == tablero[i][j+1] == tablero[i][j+2] == tablero[i][j+3]) and tablero[i][j] in (0,1):
                        return tablero[i][j] + 1
                    
                    if (j < self.HEIGHT-3  and i < self.WIDTH-3) and tablero[i][j] == tablero[i+1][j + 1] == tablero[i+2][j + 2] == tablero[i+3][j + 3]\
                            and tablero[i][j] in (0,1):
                            return tablero[i][j] + 1
                    
                    if (j > 2 and i < self.WIDTH-3) and tablero[i][j] == tablero[i+1][j - 1] == tablero[i+2][j - 2] == tablero[i+3][j - 3]\
                            and tablero[i][j] in (0,1):
                            return tablero[i][j] + 1
        
        return -1

    # Cambia el tablero en forma de columnas por una tabla numpy del tamaño del tablero
    def convertIntoTable(self):
        newBoard = np.full([self.WIDTH, self.HEIGHT], -1)

        for row in range(self.HEIGHT - 1, -1, -1):
            for col in range(self.WIDTH - 1, -1, -1):
                # El try-except esta por si accedemos a una posicion que no tiene ficha aun
                try:
                     newBoard[col][row] = self.board[col][row]
                except IndexError:
                    pass
        return newBoard

    # Devuelve si el tablero está lleno o no
    def isFull(self):
        return self.numMoves == 42

    # Muestra el estado actual del tablero, O es jugador 0 y X jugador 1
    def printBoard(self):
        print("")
        print("+" + "---+" * self.WIDTH)
        # Empezamos a pintar por la primera fila, que en board es la ultima
        for row in range(self.HEIGHT - 1, -1, -1):
            rowValue = "|"
            for col in range(self.WIDTH):
                # Con esto sabemos si hay ficha o no en esa posicion y nos ahorramos el try-except
                if len(self.board[col]) > row:
                    rowValue += " " + ('X' if self.board[col][row] else 'O') + " |"
                else:
                    rowValue += "   |"
            print(rowValue)
            print("+" + "---+" * self.WIDTH)
        print("Actualmente hay ", self.numMoves, " fichas en el tablero")

    # Copia el tablero actual en otro
    def copiarTablero(self):
        newBoard = Board(specificBoard=self)

        return newBoard

    # Devuelve una lista con todas las columnas en las que podemos poner ficha
    def movimientosPosibles(self):
        posMovs = []

        for i in range(self.WIDTH):
            if len(self.board[i]) < self.WIDTH:
                posMovs.append(i)
        return posMovs

    # Cambia el tablero actual a uno que tenga el formato requerido por la NN
    def pasarAInputDelModelo(self):

        inputToModel = np.zeros((2,6,7))
        newboard = self.convertIntoTable()
        newboard = np.transpose(newboard)

        for col in range(self.WIDTH):
            for row in range(self.HEIGHT):
                if newboard[col][row] == 0:
                    inputToModel[0][col][row] = 1
                elif newboard[col][row] == 1:
                    inputToModel[1][col][row] = 1

        # El expand es porque la entrada de la red tiene la forma (1,2,6,7) aunque no se a que pertenece el 1
        inputToModel = np.expand_dims(inputToModel, axis=0)
        return inputToModel







