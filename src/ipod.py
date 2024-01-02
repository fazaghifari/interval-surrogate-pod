import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from src.isvd import ISVD0, ISVD2, ISVD3, ISVD4
from src.isvd import tools


class IPOD():
    """
    Class for handling Interval Proper Orthogonal Decomposition (IPOD) from snapshots.
    This class only have one method (for now):
        - fit(snapshot)

    `fit` method does not return anything. But you can access class variable such as:
        - self.U, self.S, self.V : Matrices U,S,V from SVD
            - U,S,V matrices could be both interval or the average, depends on the decomposition strategy given
        - self.pod_coeff: POD coefficient
        - self.latent_vectors: The latent vectors
        - etc

    Interval POD coefficient can be different on each decomposition strategy:
        - Decomposition strategy "a" doesn't support POD coefficient currentlyt.
        - Decomposition strategy "b": Given U and V are averages and S is interval, Snapshot = L * V_avg. Where L = U_avg * S_int.
                                      Thus, pod_coeff = V_avg and latent_vectors = L.
        - Decomposition strategy "c": All U S and V are averages, Snapshot = L * V_avg, where L = U_avg * S_avg.
                                      Thus, pod_coeff = V_avg and latent_vectors = L.
    """

    def __init__(self, target_rank=5, decomp_strategy="b", isvd_choice=4) -> None:

        assert isvd_choice in [0,2,3,4], f"ISVD option is not available, available model = [0,2,3,4]"
        assert decomp_strategy in ["c", "b", "b_intv"], 'Decomposition strategy not available, available options: ["c", "b", "b_intv"]'

        self.snapshot = 0
        self.n_point = 0 
        self.n_dim = 0
        self.U, self.S, self.V = None,None,None
        self.target_rank = target_rank
        self.decomp_strategy = decomp_strategy
        self.isvd_choice = isvd_choice
        self.pod_coeff = None
        self.latent_vec = None

        if self.isvd_choice == 0:
            self.isvd = ISVD0()
        elif self.isvd_choice == 2:
            self.isvd = ISVD2(target_rank=target_rank)
        elif self.isvd_choice == 3:
            self.isvd = ISVD3(target_rank=target_rank)
        elif self.isvd_choice == 4:
            self.isvd = ISVD4(target_rank=target_rank)
        else:
            raise ValueError("ISVD option not available")

    
    def fit(self, snapshot: (np.ndarray)):
        """Fit the IPOD to snapshot matrix.
        If the class input `full_matrix == True`, then the matrix is not truncated
        Matrix truncation depends on the tolerance

        Args:
            snapshot (np.array): 3D snapshot matrix with dimension (n x m x 2)
                                 snapshot[:,:,0] should be the lower bound and snapshot[:,:,1] is the upper bound
        """
        
        # Assert snapshot dimension, must be equal to 2
        assert snapshot.ndim == 3, f"Snapshot dimension expected to be 3, got: {snapshot.ndim}"

        self.snapshot = snapshot
        self.n_point, self.n_dim, _ = self.snapshot.shape

        # Compute interval SVD
        self.U, self.S, self.V = self.isvd.fit_return(self.snapshot, decomp_strategy= self.decomp_strategy)

        # Calculate POD coeff and latent vectors
        if self.decomp_strategy == "c":
            self.pod_coeff = self.V
            self.latent_vec = self.U @ self.S
        elif self.decomp_strategy == "b" :
            self.pod_coeff = self.V
            self.latent_vec = tools.interval_matmul(np.dstack([self.U, self.U]), self.S)
        elif self.decomp_strategy == "b_intv":
            self.pod_coeff = tools.interval_matmul(self.S, np.dstack([self.V.T, self.V.T]))
            self.latent_vec = self.U
        else:
            pass
            