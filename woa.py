import numpy as np
import matplotlib.pyplot as plt


class WOA:
    # 初始化
    def __init__(self, fitness, whale_num, max_iter, dim, LB, UB, b):
        self.fitness = fitness
        self.LB = LB
        self.UB = UB
        self.dim = dim
        self.whale_num = whale_num
        self.max_iter = max_iter
        self.b = b
        # Initialize the locations of whale
        self.X = np.zeros((whale_num, dim))
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(max_iter)
        self.gBest_X = np.zeros(dim)

        # 适应度函数

    def opt(self):
        for i in range(0, self.whale_num):
            for j in range(0, self.dim):
                self.X[i, j] = np.random.rand() * (self.UB[j] - self.LB[j]) + self.LB[j]
        iterations = []
        accuracy = []
        t = 0
        while t < self.max_iter:
            for i in range(self.whale_num):
                self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)  # Check boundries
                fitness = self.fitness(self.X[i, :])
                # Update the gBest_score and gBest_X
                if fitness < self.gBest_score:
                    self.gBest_score = fitness
                    self.gBest_X = self.X[i, :].copy()

            a = 2 * (self.max_iter - t) / self.max_iter
            # Update the location of whales
            for i in range(self.whale_num):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()

                A = 2 * a * R1 - a
                C = 2 * R2
                l = 2 * np.random.uniform() - 1

                if p >= 0.5:
                    D = abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gBest_X
                else:
                    if abs(A) < 1:
                        D = abs(C * self.gBest_X - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A * D
                    else:
                        rand_index = np.random.randint(low=0, high=self.whale_num)
                        X_rand = self.X[rand_index, :]
                        D = abs(C * X_rand - self.X[i, :])
                        self.X[i, :] = X_rand - A * D

            self.gBest_curve[t] = self.gBest_score
            t += 1
            iterations.append(t)

            accuracy.append(self.gBest_score)
        return iterations, accuracy
