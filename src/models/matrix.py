import numpy as np
from typing import List


class Matrix:
    """
    This class allows you to generate a matrix and perform mathematical operations on it. The
    elements can be specified as a 1D list or NumPy array. If only a single element is specified,
    the matrix will be filled with that element. The shape of the matrix is specified by the number
    of rows and columns. The matrix calculator can handle addition, subtraction, multiplication and
    division.

    Helper methods have been written for use by the neural network and genetic algorithm for
    crossover and activation functions etc.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        elements: 'int | float | np.ndarray | list | "Matrix"',
    ):
        """
        Create a matrix of specified shape using provided elements.

        Parameters:
            rows (int): Number of rows in matrix
            cols (int): Number of columns in matrix
            elements (int | float | np.ndarray | list | Matrix): Matrix elements to use
        """
        if isinstance(elements, (int, float)):
            self.matrix = np.ones((rows, cols)) * elements
        elif isinstance(elements, (list, np.ndarray, Matrix)):
            elements_to_reshape = (
                elements.matrix if isinstance(elements, Matrix) else np.array(elements)
            )
            try:
                self.matrix = np.matrix.reshape(elements_to_reshape, (rows, cols))
                if self.matrix.dtype not in ["float64", "int32"]:
                    raise TypeError(
                        "All provided elements in array must be of type int or float."
                    )
            except ValueError:
                raise ValueError(
                    f"Number of elements must be compatible with provided shape.\n{rows} rows, {cols} columns, expected {rows * cols} elements, got {np.size(elements_to_reshape)}."
                )
        else:
            raise TypeError(
                "Provided elements must be of type int, float, list, or NumPy array."
            )

    def __str__(self) -> str:
        """
        Print matrix in a readable way.
        """
        return str(self.matrix)

    def __repr__(self) -> str:
        """
        Return string representation of matrix.
        """
        return f"Matrix({self.matrix.shape[0]}, {self.matrix.shape[1]}, {self.matrix})"

    def __add__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        Add a scalar/matrix to matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to add to matrix
        """
        if isinstance(other, Matrix):
            try:
                new_matrix = self.matrix + other.matrix
            except ValueError:
                raise ValueError(
                    f"Matrices must have same shape (trying to add {self.matrix.shape[0]} and {other.matrix.shape[0]} rows, and {self.matrix.shape[1]} and {other.matrix.shape[1]} columns)."
                )
        elif isinstance(other, (int, float)):
            new_matrix = self.matrix + other

        return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)

    def __iadd__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        In-place addition of matrix and scalar/matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to add to matrix
        """
        return self + other

    def __radd__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        Reverse addition of matrix and scalar/matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to add to matrix
        """
        return self + other

    def __sub__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        Subtract a scalar/matrix from matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to subtract from matrix
        """
        if isinstance(other, Matrix):
            try:
                new_matrix = self.matrix - other.matrix
            except ValueError:
                raise ValueError(
                    f"Matrices must have same shape (trying to subtract {self.matrix.shape[0]} and {other.matrix.shape[0]} rows, and {self.matrix.shape[1]} and {other.matrix.shape[1]} columns)."
                )
        elif isinstance(other, (int, float)):
            new_matrix = self.matrix - other

        return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)

    def __isub__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        In-place subtraction of matrix and scalar/matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to subtract from matrix
        """
        return self - other

    def __rsub__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        Reverse subtraction of matrix and scalar/matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to subtract from matrix
        """
        return self - other

    def __neg__(self) -> "Matrix":
        """
        Flip sign of all elements in matrix.
        """
        new_matrix = self.matrix * (-1)
        return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)

    def __mul__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        Multiply matrix and scalar/matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to subtract from matrix
        """
        if isinstance(other, Matrix):
            try:
                new_matrix = np.matmul(self.matrix, other.matrix)
                new_matrix_object = Matrix(
                    self.matrix.shape[0], other.matrix.shape[1], new_matrix
                )
            except ValueError:
                raise ValueError(
                    f"Number of columns in first matrix must be equal to number of rows in second matrix (first matrix has {self.matrix.shape[1]} columns and second matrix has {other.matrix.shape[0]} rows."
                )
        elif isinstance(other, (int, float)):
            new_matrix = self.matrix * other
            new_matrix_object = Matrix(
                self.matrix.shape[0], self.matrix.shape[1], new_matrix
            )

        return new_matrix_object

    def __imul__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        In-place multiplication of matrix and scalar/matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to multiply matrix
        """
        return self * other

    def __rmul__(self, other: 'int | float | "Matrix"') -> "Matrix":
        """
        Reverse multiplication of matrix and scalar/matrix.

        Parameters:
            other (int, float, Matrix): Scalar/matrix to multiply matrix
        """
        return self * other

    def __truediv__(self, other: "int | float") -> "Matrix":
        """
        Divide matrix by scalar.

        Parameters:
            other (int, float): Scalar to divide matrix
        """
        if isinstance(other, (int, float)):
            try:
                new_matrix = self.matrix / other
            except ZeroDivisionError:
                raise ZeroDivisionError("Cannot divide by 0!")

            return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)
        else:
            raise ValueError("Matrix can only be divided by a scalar.")

    def __idiv__(self, other: "int | float") -> "Matrix":
        """
        In-place division of matrix and scalar.

        Parameters:
            other (int, float): Scalar to divide matrix
        """
        return self / other

    def to_array(self) -> "List[int] | List[float]":
        """
        Return matrix as list.

        Returns:
            arr (List(int | float)): List of matrix elements
        """
        arr = np.reshape(self.matrix, self.no_of_elements)
        arr = list(arr)
        return arr

    @property
    def no_of_elements(self) -> int:
        """
        Return the number of elements in the matrix.
        """
        return self.matrix.shape[0] * self.matrix.shape[1]

    @classmethod
    def column_matrix(cls, arr: List[float]) -> "Matrix":
        """
        Return a column matrix with provided elements.

        Parameters:
            arr (List(float)): List of elements to be in matrix
        """
        return cls(len(arr), 1, arr)

    @classmethod
    def random_matrix(cls, rows: int, cols: int, low: float, high: float) -> "Matrix":
        """
        Return a matrix with randomised elements of specified size.

        Parameters:
            rows (int): Number of rows in matrix
            cols (int): Number of columns in matrix
            low (float): Lower limit for random element
            high (float): Upper limit for random element
        """
        elements = [np.random.uniform(low, high) for _ in range(rows * cols)]
        return cls(rows, cols, elements)

    @classmethod
    def zeros_matrix(cls, rows: int, cols: int) -> "Matrix":
        """
        Return a matrix with all zeros of specified size.

        Parameters:
            rows (int): Number of rows in matrix
            cols (int): Number of columns in matrix
        """
        return cls(rows, cols, 0)

    @classmethod
    def crossover(
        cls,
        matrix: "Matrix",
        other_matrix: "Matrix",
        mutation_rate: float,
        low: float,
        high: float,
    ) -> "Matrix":
        """
        Generate a new matrix by mixing two matrices. Each element in the new matrix is randomly
        selected from the two matrices, but there is also a chance for the element to be random.

        Parameters:
            matrix (Matrix): Matrix to use to generate new matrix
            other_matrix (Matrix): Matrix to use to generate new matrix
            mutation_rate (float): Probability for element to be random, range [0, 1]
            low (float): Lower limit for random element
            high (float): Upper limit for random element
        """
        matrix_elements = Matrix.to_array(matrix)
        other_matrix_elements = Matrix.to_array(other_matrix)

        elements = []

        for i in range(matrix.no_of_elements):
            rng = np.random.uniform(0, 1)
            if rng < mutation_rate:
                elements.append(np.random.uniform(low, high))
            elif rng < (0.5 + mutation_rate / 2):
                elements.append(matrix_elements[i])
            else:
                elements.append(other_matrix_elements[i])

        return cls(matrix.matrix.shape[0], matrix.matrix.shape[1], elements)

    @classmethod
    def map(cls, matrix: "Matrix", map_function) -> "Matrix":
        """
        Map all elements in matrix using a mapping function.

        Parameters:
            matrix (Matrix): Matrix elements to map to new matrix
            map_function: Mathematical function f(x)
        """
        return cls(
            matrix.matrix.shape[0], matrix.matrix.shape[1], map_function(matrix.matrix)
        )
