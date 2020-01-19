# TFG---Connect4IA
Implementacion de varios algoritmos de IA para jugar al Connect4 como TFG de Ingeniería Informática en la UAB

En este código se implementan los siguientes algoritmos:
      + Minimax (MM)
      + Minimax con poda alfabeta (AB)
      + Monte Carlo Tree Search (MCTS)
      + MCTS con la utilización de una red neuronal (Código derivado del algoritmo AlphaZero)
      
Para este último se ha hecho solo la implementacion que utiliza una red neuronal entrenada con el algoritmo descrito en el repositorio https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning.git



El código está formado por los siguientes ficheros:
      + main.py: En este fichero se crea la partida y se lleva a cabo el bucle principal de la misma. Antes de empezar hay que establecer varios parámetros que regulan el funcionamiento de los algortimos implementados. También es aquí donde se crea y se carga la red neuronal en caso de que se vaya a usar una
      + players.py: En este fichero se definen los tipos de jugadores que se pueden utilizar para la partida. Todos heredan de una clase central y luego tienen implementado su propio código para escoger movimiento. Los jugadores actuales son los siguientes:
                  + Random Player (en cada turno escoge la accion de forma totalmente aleatoria)
                  + Manual Player (en cada turno pregunta al usuario qué accion quiere escoger)
                  + Minimax Player
                  + Minimax AB Player
                  + MCTS Player (este jugador se utiliza tanto para el MCTS puro como para el MCTS mezclado con la red neuronal del AZ
      + board.py: En este fichero se define la clase del tablero y todas las funciones que se le pueden aplicar, como hacer un movimiento o mirar el estado actual de la partida      
      +loss.py, model.py: Ambos ficheros son del repositorio anteriormente mencionado y regulan la carga de la red y la función de pérdida
