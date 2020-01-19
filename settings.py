run_folder = './run/'
run_archive_folder = './run_archive/'

#### SELF PLAY
EPISODES = 5

# Numero de simulaciones que se hacen con el MCTS antes de tomar la decision
MCTS_SIMS = 30
MEMORY_SIZE = 30

# Turno a partir del cual se empieza a jugar deterministicamente
TURNS_UNTIL_TAU0 = 10

# Equilibrio entre exploracion y explotacion que se usa para decidir que nodo se va a abrir a continuacion en el MCTS
CPUCT = 1
EPSILON = 0.2

# Movida que influye en la fase de seleccion en el MCTS
ALPHA = 0.8

#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1

MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
    {'filters': 75, 'kernel_size': (4, 4)}
    , {'filters': 75, 'kernel_size': (4, 4)}
    , {'filters': 75, 'kernel_size': (4, 4)}
    , {'filters': 75, 'kernel_size': (4, 4)}
    , {'filters': 75, 'kernel_size': (4, 4)}
    , {'filters': 75, 'kernel_size': (4, 4)}
]

#### Episodios que se juegan oara validar
EVAL_EPISODES = 8

# Multiplicador que se añade a la puntucaion del mejor jugador para ver si el actual lo supera y se convierte en el nuevo mejor
SCORING_THRESHOLD = 1.3
