from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        self.descent: BaseDescent = get_descent(descent_config)  # Получаем объект класса для градиентного спуска
        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков (включая столбец единиц для свободного члена).
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.
        """
        # Начальное значение потерь и инициализация весов
        self.loss_history.append(self.calc_loss(x, y))  # Записываем начальное значение функции потерь
        prev_weights = self.descent.w.copy()

        for iteration in range(self.max_iter):
            # Выполняем шаг градиентного спуска
            weights_diff = self.descent.step(x, y)
            self.loss_history.append(self.calc_loss(x, y))  # Записываем значение функции потерь

            # Проверка условий остановки
            if np.linalg.norm(weights_diff) < self.tolerance:  # Если евклидова норма разности весов меньше tolerance
                print(f'Остановка на итерации {iteration} по критерию нормы веса.')
                break

            if np.isnan(self.descent.w).any():  # Если в весах появились NaN
                print(f'Остановка на итерации {iteration} из-за NaN в весах.')
                break

            prev_weights = self.descent.w.copy()  # Обновляем предыдущие веса

        return self



    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)

    def r2_score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

