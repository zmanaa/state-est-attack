## Getting Started

1.  **Dependencies:** Ensure you have **Python 3.8 or newer** installed.
2.  **Installation:** Install the required libraries using the following
    ```bash
    pip install -r requirements.txt
    ```

## Execution

1.  Clone this repository:
    ```bash
    git clone https://github.com/zmanaa/state-est-attack.git
    cd state-est-attack
    ```
2.  Run the main simulation script:
    ```bash
    python3 main.py
    ```

### Output

The script will automatically create a new directory named `figures/` and save two PDF plots inside it:

* **`estimator.pdf`**: Time trajectories of the **true state (z)** and the **estimated state (zhat)** (Figure 3).
* **`comparison.pdf`**: Comparison of the norms of the true state (Norm(z)) and the estimation error (Norm(e)).# state-est-attack (Figure 4).
