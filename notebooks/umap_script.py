from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import numpy as np
import umap

if __name__ == "__main__":
    df = np.load("./VGG16_bald_100_0.001_21_04_2023_15-16.npz")
    data = df["x"].astype(np.float16)

    # n = data.shape[0]  # how many rows we have in the dataset
    # chunk_size = 1000  # how many rows we feed to IPCA at a time, the divisor of n
    # ipca = IncrementalPCA(n_components=200, batch_size=16)

    # for i in tqdm(range(0, n // chunk_size)):
    #     ipca.partial_fit(data[i * chunk_size : (i + 1) * chunk_size])
    # ipca.partial_fit(data[(i + 1) * chunk_size :])

    # x_transform = np.ndarray(shape=(0, 200))
    # for i in tqdm(range(0, n // chunk_size)):
    #     partial_x_transform = ipca.transform(
    #         data[i * chunk_size : (i + 1) * chunk_size]
    #     )
    #     x_transform = np.vstack((x_transform, partial_x_transform))
    # partial_x_transform = ipca.transform(data[(i + 1) * chunk_size :])
    # x_transform = np.vstack((x_transform, partial_x_transform))

    # # x_transformed = ipca.transform(data)
    # del df
    # del data
    fitter = umap.UMAP(verbose=True, low_memory=True, n_neighbors=10).fit(data)
    embedding = fitter.embedding_

    np.savez_compressed(
        "embedding_200_VGG16_bald_100_0.001_21_04_2023_15-16", x=embedding
    )
