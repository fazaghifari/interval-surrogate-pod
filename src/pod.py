import numpy as np
import scipy.io
import matplotlib.pyplot as plt


class POD():
    """
    Class for handling Proper Orthogonal Decomposition (POD) from snapshots.
    This class only have one method (for now):
        - fit(snapshot)

    `fit` method does not return anything. But you can access class variable such as:
        - self.U, self.S, self.V : Matrices U,S,V from SVD
        - self.U_trunc: Truncated U matrix if full_matrix = False
        - self.S_trunc: Truncated S matrix if full_matrix = False
        - self.pod_coeff: POD coefficient

    POD coefficient equal to $S \times V.T$ or $U.T \times snapshot$.
    If you want to use POD as a dimensionality reduction for Surrogate Model, then the one that the
    surrogate model (NN/GP/SVM/etc.) predict is the POD coefficient.
    """

    def __init__(self, tolerance = 0.01, full_matrix=False) -> None:
        """Initialize POD class
        
        Args:
            tolerance (float, optional): Tolerance limit of the basis truncation. Defaults to 0.01.
            full_matrix (bool, optional): Set the SVD to return full matrix or not. Defaults to False.
        """

        # Assert conditions
        assert tolerance >= 0 , f"Expected tolerance to be larger than equal to 0, got: {tolerance}"

        self.snapshot = 0
        self.n_point = 0 
        self.n_dim = 0
        self.U, self.S, self.V = None,None,None
        self.tolerance = tolerance
        self.full_matrix = full_matrix
        self.truncated_idx = None
        self.S_trunc = None
        self.U_trunc = None
        self.pod_coeff = None


    def fit(self, snapshot: (np.ndarray)):
        """Fit the POD to snapshot matrix.
        If the class input `full_matrix == True`, then the matrix is not truncated
        Matrix truncation depends on the tolerance

        Args:
            snapshot (np.array): Snapshot matrix 
        """
        
        # Assert snapshot dimension, must be equal to 2
        assert snapshot.ndim == 2, f"Snapshot dimension expected to be 2, got: {snapshot.ndim}"

        self.snapshot = snapshot
        self.n_point, self.n_dim = self.snapshot.shape

        # Compute Basic SVD using numpy
        self.U, self.S, self.V = np.linalg.svd(self.snapshot,full_matrices=self.full_matrix)

        if not self.full_matrix:
            # Truncate matrix
            temp = 0
            diagsum = np.sum(self.S)
            for idx, sigma in enumerate(self.S):
                temp += sigma
                ratio = temp/diagsum

                if ratio >= (1-self.tolerance):
                    self.truncated_idx = idx
                    break

            self.S_trunc = self.S[:self.truncated_idx+1]
            self.U_trunc = self.U[:,:self.truncated_idx+1]
            self.pod_coeff = self.U_trunc.T @ self.snapshot

        else:
            # Does not truncate matrix
            self.pod_coeff = self.U.T @ self.snapshot


if __name__ == "__main__":

    # Test POD
    # Load and extract data
    data = scipy.io.loadmat("data/cylinder_nektar_wake.mat")
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2

    # Reshape Data
    X_grid = X_star[:,0].reshape(50,100)
    Y_grid = X_star[:,1].reshape(50,100)
    u_grid = U_star[:,0,0].reshape(50,100)

    # Construct snapshot matrix, using u_x as the value field
    snapshot = U_star[:,0,:]

    # Initiate and fit POD to snapshot matrix
    # Set `full_matrix = False` for truncated SVD 
    pod = POD(tolerance=0.01, full_matrix=False)
    pod.fit(snapshot)

    # Try plotting first basis function of POD
    u1 = pod.U_trunc[:,0].reshape(50,100) # get first order basis
    fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
    # Plot full-order solution
    cf = ax1[0].contourf(X_grid,Y_grid,u_grid)
    cbar = fig1.colorbar(cf, ax=ax1[0])
    ax1[0].set_xlabel('X')
    ax1[0].set_ylabel('Y')
    ax1[0].set_title('Full Order')
    # Plot first basis function
    cf1 = ax1[1].contourf(X_grid,Y_grid,u1)
    cbar1 = fig1.colorbar(cf1, ax=ax1[1])
    ax1[1].set_xlabel('X')
    ax1[1].set_ylabel('Y')
    ax1[1].set_title('First Basis')

    plt.show()