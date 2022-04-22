import os
import numpy as np
import scipy.io as sio
from scipy import sparse


def loadmat(path, to_dense=True):
    data = sio.loadmat(path)
    X = data["X"]
    y_true = data["y_true"].astype(np.int32).reshape(-1)

    if sparse.isspmatrix(X) and to_dense:
        X = X.toarray()

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    return X, y_true, N, dim, c_true


def load_Agg():
    this_directory = os.path.dirname(__file__)
    data_path = os.path.join(this_directory, "dataset/")
    name_full = os.path.join(data_path + "Agg.mat")
    X, y_true, N, dim, c_true = loadmat(name_full)
    return X, y_true, N, dim, c_true


def load_USPS():
    this_directory = os.path.dirname(__file__)
    data_path = os.path.join(this_directory, "dataset/")
    name_full = os.path.join(data_path + "USPS.mat")
    X, y_true, N, dim, c_true = loadmat(name_full)
    return X, y_true, N, dim, c_true


def turdata(N, tur, d=2, dis="gaussian"):
    """
    dis=["uniform", "gaussian"]
    """

    if dis == "gaussian":
        mu = np.repeat(0, d)
        sig = np.eye(d) * tur
        x = np.random.multivariate_normal(mu, sig, N)
    elif dis == "uniform":
        x = np.random.uniform(-tur/2, tur/2, (N, 2))
    return x


def rand_ring(r, N, tur, cx=0, cy=0, s=0, e=2 * np.pi, dis="gaussian"):
    """
    dis=["uniform", "gaussian"]
    """
    theta = np.linspace(s, e, N)
    x = np.vstack((r * np.cos(theta), r * np.sin(theta))).T
    x = x + turdata(N, tur, d=2, dis=dis)

    x[:, 0] += cx
    x[:, 1] += cy

    return x

def twospirals(N=2000, degrees=570, start=90, noise=0.2):
    X = np.zeros((N, 2), dtype=np.float64)
    deg2rad = np.pi/180
    start = start * deg2rad

    N1 = int(np.floor(N/2))
    N2 = N - N1

    n = start + np.sqrt(np.random.rand(N1)) * degrees * deg2rad
    X[:N1, 0] = -np.cos(n) * n + np.random.rand(N1) * noise
    X[:N1, 1] =  np.sin(n) * n + np.random.rand(N1) * noise

    n = start + np.sqrt(np.random.rand(N2)) * degrees * deg2rad
    X[N1:, 0] =  np.cos(n) * n + np.random.rand(N2) * noise
    X[N1:, 1] = -np.sin(n) * n + np.random.rand(N2) * noise

    y = np.ones(N, dtype=np.int32)
    y[:N1] = 0
    return X, y


def data_description(data_path, data_name, version, url):
    full_name = os.path.join(data_path, f"{data_name}_{version}.mat")
    X, y_true, N, dim, c_true = loadmat(full_name)

    # title and content
    T1 = "data_name"
    T2 = "# Samples"
    T3 = "# Features"
    T4 = "# Subjects"

    C1 = data_name
    C2 = str(X.shape[0])
    C3 = str(X.shape[1])
    C4 = str(c_true)

    n1 = max(len(T1), len(C1))
    n2 = max(len(T2), len(C2))
    n3 = max(len(T3), len(C3))
    n4 = max(len(T4), len(C4))

    y_df = pd.DataFrame(data=y_true, columns=["label"])
    ind_L = y_df.groupby("label").size()

    show_n = 5

    with open("{}{}_{}.txt".format(data_path, data_name, version), "a") as f:

        # version
        f.write("version = {}\n\n".format(version))

        # table
        f.write("{}  {}  {}  {}\n".format(
            T1.rjust(n1), T2.rjust(n2), T3.rjust(n3), T4.rjust(n4)))
        f.write("{}  {}  {}  {}\n\n".format(
            C1.rjust(n1), C2.rjust(n2), C3.rjust(n3), C4.rjust(n4)))

        # url
        f.write("url = {}\n\n".format(url))
        f.write("=================================\n")

        # content
        f.write("X[:, :2], {}, {}, {}\n".format(
            str(type(X))[8:-2], X.shape, str(type(X[0, 0]))[8:-2]))
        if isinstance(X, sparse.spmatrix):
            f.write("{}\n".format(X[:show_n, :2].toarray()))
        else:
            f.write("{}\n".format(X[:show_n, :2]))
        f.write("...\n\n")

        f.write("y_true, {}, {}, {}\n".format(
            str(type(y_true))[8:-2], y_true.shape, str(type(y_true[0]))[8:-2]))
        f.write("{}".format(y_true[:show_n]))
        f.write("...\n\n")

        f.write("distribution\n")
        f.write(ind_L[:50].to_string())
        f.write("\n\n")
