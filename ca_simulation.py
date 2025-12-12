# ca_simulation.py
"""

Simula un autómata celular 100x100 con 3 estados:
  - S0: reposo
  - S1: activación
  - S2: refractario

Reglas:
  1. S1 -> S2
  2. S2 -> S0
  3. S0 -> S1 si vecinos activos >= 3 con probabilidad P
     P = ActiveNeighbors / (8 * R)

R es un parámetro global (ligado a cp_dose / cp_time en el informe).
Se ejecutan varias simulaciones para distintos R y se grafica la
cantidad de celdas S1 en el tiempo.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

GRID_SIZE = 100
N_ITER = 800
RANDOM_STATE = 42

# Estados
S0 = 0  # Resting
S1 = 1  # Activation
S2 = 2  # Refractory


def _init_grid() -> np.ndarray:
    """
    Inicializa el grid en estado S0.
    """
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)


def _neighbors_active_count(grid: np.ndarray, i: int, j: int) -> int:
    """
    Cuenta vecinos activos (S1) usando vecindad de Moore (8 vecinos).
    """
    count = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni = i + di
            nj = j + dj
            if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                if grid[ni, nj] == S1:
                    count += 1
    return count


def run_ca_simulation(R: float) -> Tuple[np.ndarray, List[int]]:
    """
    Ejecuta la simulación del autómata para un valor de R.

    - Inicialmente todo S0.
    - En t=50 se activa un cluster 2x2 en el centro (S1).
    - Se aplican las reglas durante N_ITER.

    Retorna:
      - grid final
      - lista con el número de celdas S1 por iteración.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    grid = _init_grid()

    s1_counts = []

    for t in range(N_ITER):
        # En t=50: cluster 2x2 activo en el centro
        if t == 50:
            c = GRID_SIZE // 2
            grid[c : c + 2, c : c + 2] = S1

        new_grid = grid.copy()

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                state = grid[i, j]

                if state == S1:
                    # Decaimiento
                    new_grid[i, j] = S2
                elif state == S2:
                    # Vuelve a reposo
                    new_grid[i, j] = S0
                else:  # S0
                    active_neighbors = _neighbors_active_count(grid, i, j)
                    if active_neighbors >= 1:
                        # Probabilidad dependiente de R
                        P = active_neighbors / (8.0 * R)
                        P = min(max(P, 0.0), 1.0)
                        if rng.random() < P:
                            new_grid[i, j] = S1

        grid = new_grid
        s1_counts.append(int((grid == S1).sum()))

    return grid, s1_counts


def run_ca_experiments():
    """
    Ejecuta la simulación para varios valores de R y grafica
    el número de celdas S1 en el tiempo.

    Valores típicos:
      - R_bajo  ~ 0.3  (ruido se disipa)
      - R_medio ~ 0.5  (onda estable)
      - R_alto  ~ 0.7  (caos/fractal)
    """
    R_values = [0.3, 0.5, 0.7]
    plt.figure()

    for R in R_values:
        _, s1_counts = run_ca_simulation(R)
        plt.plot(s1_counts, label=f"R={R}")

    plt.xlabel("Iteración")
    plt.ylabel("Número de celdas en estado S1")
    plt.title("Evolución de activación para distintos valores de R")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ca_s1_counts_vs_time.png")
    plt.close()

    print("Simulaciones CA completadas. Gráfica guardada en ca_s1_counts_vs_time.png")

def save_ca_snapshot(R: float, t_snapshot: int = 200):
    """
    Ejecuta la simulación del autómata hasta un tiempo t_snapshot
    y guarda una imagen del grid en ese instante.

    Especialmente útil para visualizar patrones espaciales
    (ondas / caos) para R altos, por ejemplo R = 0.9.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    grid = _init_grid()

    for t in range(t_snapshot):
        # Activación inicial en t = 50 (igual que en run_ca_simulation)
        if t == 50:
            c = GRID_SIZE // 2
            grid[c:c+2, c:c+2] = S1

        new_grid = grid.copy()

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                state = grid[i, j]

                if state == S1:
                    # S1 -> S2
                    new_grid[i, j] = S2
                elif state == S2:
                    # S2 -> S0
                    new_grid[i, j] = S0
                else:  # S0
                    active_neighbors = _neighbors_active_count(grid, i, j)

                    
                    if active_neighbors >= 1:
                        P = active_neighbors / (8.0 * R)
                        P = min(max(P, 0.0), 1.0)
                        if rng.random() < P:
                            new_grid[i, j] = S1

        grid = new_grid

    # Comprobar que S1 no sea cero:
    print("Celdas S1 en el snapshot:", int((grid == S1).sum()))

    plt.figure(figsize=(6, 6))
    # Muestra solo las celdas activas (S1=1, resto=0) para que el patrón resalte
    plt.imshow((grid == S1).astype(int), cmap="inferno", interpolation="nearest")
    plt.title(f"Patrón Espacial del Autómata (R={R})")
    plt.colorbar(label="Estado (S1 = 1)")

    filename = f"ca_pattern_R{R}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(f"Imagen del patrón espacial guardada como {filename}")

