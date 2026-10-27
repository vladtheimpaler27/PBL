import numpy as np

class LinUCB:
    def __init__(self,n_arms, d,alpha=0.5):
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
        self.alpha = alpha


    def choose(self, x):
        scores = []
        for A,b in zip(self.A, self.b):
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            scores.append(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
            return int(np.argmax(scores))

        def update(self, arm, x, reward):
            self.A[arm] += np.outer(x,x)
            self.b[arm] += reward * x    