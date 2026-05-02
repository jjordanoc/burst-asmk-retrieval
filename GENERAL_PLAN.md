GOAL: Develop an image retrieval system followin Shi et al., a state of the art method that minimizes Mean Average Precision (mAP). You should do this in a single Python file, separated by comments (which will be exported to a jupyter notebook afterwards).

To implement the Shi et al. early burst detection combined with ASMK[cite: 2], you definitely do not need to write the heavy mathematical lifting from scratch. You will primarily rely on the standard scientific Python stack (`numpy`, `scipy`) along with some highly optimized classical computer vision libraries. 

Here is exactly what you need, broken down by the steps of the pipeline:

### 1. Feature Extraction (Hessian-Affine + RootSIFT)
The paper specifically uses the Hessian-affine detector and RootSIFT descriptors[cite: 2]. 
*   **The Tool:** `OpenCV` (`cv2`) and `NumPy`.
*   **The Heavy Lifting:** 
    *   Standard `cv2.SIFT_create().detectAndCompute()` will give you the DoG keypoints and SIFT descriptors. 
    *   To get **RootSIFT**[cite: 2], you just use `NumPy` to $L_1$-normalize the SIFT array and apply `np.sqrt()`.
    *   *Pro-Tip for Hessian-Affine:* Pure OpenCV does not have a great built-in Hessian-Affine detector out of the box. If you want the exact SOTA detector mentioned in the paper[cite: 2], look for a Python wrapper called **`pyhesaff`** or use the **`cyvlfeat`** library (a Python wrapper for VLFeat, the legendary classical CV library). If you can't get those installed, standard `cv2.SIFT` is an acceptable fallback, though slightly less robust to extreme angles.

### 2. Early Burst Detection (The Shi et al. Magic)
This is where you build the affinity matrix and group the features[cite: 2]. You do not need `skimage` or `cv2` for this; it is purely a graph/matrix math problem.
*   **The Tools:** `SciPy` and `NumPy`.
*   **The Heavy Lifting:**
    *   **Affinity Matrix:** Use `scipy.spatial.distance.pdist` and `scipy.spatial.distance.squareform`. This will calculate the pairwise distances between all descriptors in an image in one highly optimized C-level sweep.
    *   **Connected Components:** Once you apply a threshold to your matrix (e.g., setting distances below your threshold to 1 and above to 0), use **`scipy.sparse.csgraph.connected_components`**. This single function instantly does the graph traversal to find the bursty groups proposed by Shi et al[cite: 2].
    *   **Aggregation:** Use `NumPy` to take the `np.mean()` of the descriptors in each group and `l2`-normalize them. 

### 3. Vocabulary Generation & ASMK Indexing
Tolias's ASMK requires clustering millions of descriptors into a visual vocabulary and then indexing them[cite: 2].
*   **The Tool:** **`FAISS`** (by Meta AI).
*   **The Heavy Lifting:** `FAISS` is the absolute gold standard for classical vector similarity search. While developed by an AI lab, its core is just blisteringly fast C++ implementations of K-Means and Nearest Neighbors. 
    *   Use `faiss.Kmeans` to train your visual vocabulary (it will do in seconds what `sklearn.cluster.MiniBatchKMeans` takes minutes to do).
    *   Use FAISS inverted file structures (`IndexIVF`) to handle the actual database searching.

### Are there already implemented versions of the pipeline?

**For the ASMK part:** Yes! 
You do not need to implement ASMK from scratch. There is an open-source, highly optimized Python implementation of ASMK maintained by the original authors (Giorgos Tolias and his lab). 
*   If you search GitHub for **`gtolias/asmk`**, you will find the official repository. It handles the K-means, the inverted files, and the ASMK similarity metrics perfectly. You can literally just convert it into Python and feed your aggregated descriptors into it.

**For the Shi et al. Burst Detection part:** No direct library package.
Because the burst detection occurs on a per-image basis *before* indexing, there isn't a massive standalone library for it. However, as outlined above, the entire Shi et al. burst detection algorithm[cite: 2] is essentially 15-20 lines of `SciPy` code. 

