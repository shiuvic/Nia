# encoding=utf8

"""Implementation of Griewank funcion."""

from niapy.problems.problem import Problem
from Problem.fitness_process import save

__all__ = ['Griewank']


class Griewank(Problem):
    r"""Implementation of Griewank function.

    Date: 2018

    Authors: Iztok Fister Jr. and Lucija Brezočnik

    License: MIT

    Function: **Griewank function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} - \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} -
            \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1$

        Equation:
            \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} -
            \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1 \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference paper:
    Jamil, M., and Yang, X. S. (2013).
    A literature survey of benchmark functions for global optimisation problems.
    International Journal of Mathematical Modelling and Numerical Optimisation,
    4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, algo=None, times=None, *args, **kwargs,):
        self.c = 0
        r"""Initialize Griewank problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bound of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bound of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)
        self.algo = algo
        self.times = times
    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} - \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1$'''

    def _evaluate(self, sol):
        fitness_val = sol[0] - sol[1] + 1.0
        save(fitness_val, self.algo, self.times, type=False)

        return fitness_val