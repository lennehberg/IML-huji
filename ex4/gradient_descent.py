from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from base_module import BaseModule
from base_learning_rate import  BaseLR
from learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(**kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """
    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-17),
                 tol: float = 1e-5,
                 max_iter: int = 20000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """
        # initialize random starting weights
        tol_reached = False
        x_t = f.weights # x_0
        losses = []
        weights = []

        for t in range(self.max_iter_):
            # set module with weights
            f.weights = x_t
            weights.append(x_t)

            # compute loss to find best
            loss_t = f.compute_output(X=X, y=y)
            losses.append(loss_t)

            # record findings
            self.callback_(val=loss_t, weights=x_t)

            # if tolerance reached, saved the last weight and break
            if tol_reached:
                break

            # compute gradient of f(x_t)
            partial_f = f.compute_jacobian(X=X, y=y)
            # choose a sub-gradient at random from partial_f
            v_t =np.random.choice(partial_f)

            # get current step size
            step_t = self.learning_rate_.lr_step(t=t)
            # decent down the unknown, and hope for the best
            x_t_next = x_t - (step_t * v_t)
            # check tolerance
            if np.linalg.norm(x_t_next - x_t) < self.tol_:
                tol_reached = True
            x_t = x_t_next

        # return result according to out_type
        if self.out_type_ == 'last':
            return weights[-1]
        elif self.out_type_ == 'best':
            best_ind = np.argmin(losses)
            return weights[best_ind]
        else:
            return np.add(weights) / self.max_iter_
