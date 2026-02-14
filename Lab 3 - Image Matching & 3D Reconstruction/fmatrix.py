import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import extract_and_match_SIFT,generate_3d_points
from plots import draw_matches

def find_fmatrix(pts1:np.ndarray, pts2:np.ndarray, normalize:bool = False) -> np.ndarray:
    '''Estimate the F matrix from the matching points pts1 and pts2

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes. 
        :param bool normalize: if True, points are normalized to improve stability.
        :returns np.ndarray: a 3x3 array representing the F matrix.

    '''
    # For better stability, normalize points to be centered at (0,0) with unit variance 
    if normalize:
        mean1 = np.mean(pts1, axis=1)
        std1 = np.std(pts1, axis=1)
        T1 = np.array([[1/std1[0], 0, -mean1[0]/std1[0]], [0, 1/std1[1], -mean1[1]/std1[1]], [0, 0, 1]])
        pts1 = T1 @ np.vstack((pts1, np.ones((1, np.size(pts1, 1)))))
        mean2 = np.mean(pts2, axis=1)
        std2 = np.std(pts2, axis=1)
        T2 = np.array([[1/std2[0], 0, -mean2[0]/std2[0]], [0, 1/std2[1], -mean2[1]/std2[1]], [0, 0, 1]])
        pts2 = T2 @ np.vstack((pts2, np.ones((1, np.size(pts2, 1)))))

    # Use image positions of matching pairs to create a matrix B
    xa = pts1[0, 0]
    ya = pts1[1, 0]
    xb = pts2[0, 0]
    yb = pts2[1, 0]
    B = np.array([[xb * xa, xb * ya, xb, yb * xa, yb * ya, yb, xa, ya, 1]])

    for i in range(1, np.shape(pts1)[1]):
        xa = pts1[0, i]
        ya = pts1[1, i]
        xb = pts2[0, i]
        yb = pts2[1, i]
        curr_eq = np.array([[xb * xa, xb * ya, xb, yb * xa, yb * ya, yb, xa, ya, 1]])
        B = np.vstack((B, curr_eq))

    # Compute square of B, C = transpose(B) * B
    C = np.transpose(B) @ B

    # Find eigenvector h of smallest eigenvalue lambda of C
    U, S, Vf = np.linalg.svd(C)
    f = Vf[-1, :]

    # Rearrange 9-vector h into a 3x3 homography H
    F = f.reshape(3, 3)

    # Undo the normalization to get F in the original coordinates
    if normalize:
        F = T2.T @ F @ T1
    return F


def fmatrix_error(F1:np.ndarray, F2:np.ndarray, focal:float) -> float:
    '''Computes the error between two F matrices.
        :param np.ndarray F1: a 3x3 matrix representing one of the F matrices.
        :param np.ndarray F2: a 3x3 matrix representing the second F matrix.
        :returns float: the error between the two F matrices.
    '''
    F1n = np.diag([focal, focal, 1.0]) @ F1 @ np.diag([focal, focal, 1.0])
    F2n = np.diag([focal, focal, 1.0]) @ F2 @ np.diag([focal, focal, 1.0])
    F1n = F1n/np.linalg.norm(F1n)
    F2n = F2n/np.linalg.norm(F2n)
    if np.sum(F1n*F2n)<0:
        F2n = -F2n
    return np.linalg.norm(F1n - F2n)


def count_fmatrix_inliers(F, pts1, pts2, thresh = 0.5):
    '''Given the matrix F, projects pts1 on the second image, counting the number of actual points in pts2 for which the projection error is smaller than the given threshold.

        :param np.ndarray F: a 3x3 matrix containing the F matrix.
        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param float thresh: the threshold to consider points as inliers.
        :returns int ninliers, np.ndarray errors:
            ninliers: the number of inliers.
            errors: a N_points array containing the errors; they are indexed as pts1 and pts2.
    
    '''
    Fp = F@np.vstack((pts1, np.ones((1,np.size(pts1, 1)))))
    pF = F.T@np.vstack((pts2, np.ones((1,np.size(pts2, 1)))))
    pFp = (Fp[0,:]*pts2[0,:] + Fp[1,:]*pts2[1,:] + Fp[2,:])**2
    l1 = Fp[0,:]**2 + Fp[1,:]**2 
    l2 = pF[0,:]**2 + pF[1,:]**2 
    errors = np.sqrt(pFp/l1 + pFp/l2)
    ninliers = np.sum(np.where(errors<thresh, 1, 0))
    return ninliers, errors


def find_fmatrix_RANSAC(pts1:np.ndarray, pts2:np.ndarray, niter:int = 100, thresh:float = 1.0):
    '''Computes the best F matrix for matching points pts1 and pts2, adopting RANSAC.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param int niter: the number of RANSAC iteration to run.
        :param float thresh: the maximum error to consider a point as an inlier while evaluating a RANSAC iteration.
        :returns np.ndarray Fbest, int ninliers, np.ndarray errors:
            Fbest: a 3x3 matrix representing the best F matrix found.
            ninliers: the number of inliers for the best F matrix found.
            errors: a N_points array containing the errors for the best F matrix found; they are indexed as pts1 and pts2.
    
    '''
    Fbest = None
    errors = None
    ninliers = 0

    # Loop niter times
    for i in range(niter):

        # Generate a minimum set of 8 random feature matches
        idx = np.random.choice(pts1.shape[1], 8, replace=False)
        pts1_subsample = pts1[:, idx]
        pts2_subsample = pts2[:, idx]

        # Find F using these matches
        F_curr = find_fmatrix(pts1_subsample, pts2_subsample, True)
        ninliers_curr, errors_curr = count_fmatrix_inliers(F_curr, pts1, pts2, thresh)

        if Fbest is None or ninliers_curr > ninliers:
            ninliers = ninliers_curr
            errors = errors_curr
            Fbest = F_curr

    return Fbest, ninliers, errors


def synthetic_example(RANSAC=False):
    focal = 1000
    pts1, pts2, F = generate_3d_points(num = 200, noutliers = 20, noise=0.5, focal=focal, spherical = True)
    draw_matches(pts1, pts2) 
    print('True F =\n', F/np.linalg.norm(F))
    if RANSAC:
        F1,ninliers,errors = find_fmatrix_RANSAC(pts1, pts2, niter=10000)
        F2 = find_fmatrix(pts1[:,errors<1], pts2[:,errors<1], normalize=True)
        print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
        print('RANSAC F =\n', F1/np.linalg.norm(F1))
        print('Final estimated F =\n', F2/np.linalg.norm(F2))
    else:
        F2 = find_fmatrix(pts1, pts2, normalize=True) 
        print('Estimated F =\n', F2/np.linalg.norm(F2))
    print('Error =', fmatrix_error(F, F2, focal))


def real_example():
    img1 = cv2.imread('images/desk1.jpg', 0)
    img2 = cv2.imread('images/desk2.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    draw_matches(pts1, pts2, img1, img2)
    F1,inliers,errors = find_fmatrix_RANSAC(pts1, pts2, 10000)
    draw_matches(pts1[:,errors<1], pts2[:,errors<1], img1, img2)

# Task 3
############################

def task_3_example_syn():
    focal = 1000
    pts1, pts2, F = generate_3d_points(num=100, noutliers=0, noise=0.5, focal=focal, spherical=True)
    draw_matches(pts1, pts2)
    print('True F =\n', F / np.linalg.norm(F))
    F2 = find_fmatrix(pts1, pts2, normalize=True)
    print('Estimated F =\n', F2 / np.linalg.norm(F2))
    print('Error =', fmatrix_error(F, F2, focal))

def task_3_example_real():
    # No ground truth or visualisation so kind of pointless
    img1 = cv2.imread("images/books1.jpg", flags=0)
    img2 = cv2.imread("images/books2.jpg", flags=0)

    pts1, pts2 = extract_and_match_SIFT(img1, img2, num=1000)
    F = find_fmatrix(pts1, pts2, normalize=True)
    print(F)

def task_3_outlier_test():
    focal = 1000
    outlier_levels = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 50, 60, 70, 80, 90, 100]
    errors_nonorm = []
    errors_norm = []
    for i in range(len(outlier_levels)):
        error_nonorm = 0.0
        error_norm = 0.0
        for j in range(100):
            pts1, pts2, F = generate_3d_points(num=100, noutliers=outlier_levels[i], noise=0.5, focal=focal, spherical=True)
            # No normalization
            F2 = find_fmatrix(pts1, pts2, normalize=False)
            error_nonorm += fmatrix_error(F, F2, focal)
            # No normalization
            F2 = find_fmatrix(pts1, pts2, normalize=True)
            error_norm += fmatrix_error(F, F2, focal)
        errors_nonorm.append(error_nonorm/100)
        errors_norm.append(error_norm/100)

    plt.figure(figsize=(8, 5))
    plt.plot(outlier_levels, errors_norm, marker='o')
    plt.plot(outlier_levels, errors_nonorm, marker='o')
    plt.xlabel("Outliers")
    plt.ylabel("Fundamental Matrix Error")
    plt.title("Fundamental Matrix Error vs. Outliers")
    plt.grid(True)
    plt.show()

def task_3RANSACN_example_syn():
    focal = 1000
    pts1, pts2, F = generate_3d_points(num = 200, noutliers = 20, noise=0.5, focal=focal, spherical = True)
    draw_matches(pts1, pts2)
    print('True F =\n', F/np.linalg.norm(F))
    F1,ninliers,errors = find_fmatrix_RANSAC(pts1, pts2, niter=10000, thresh=1.0)
    F2 = find_fmatrix(pts1[:,errors<1], pts2[:,errors<1], normalize=True)
    print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
    print('RANSAC F =\n', F1/np.linalg.norm(F1))
    print('Final estimated F =\n', F2/np.linalg.norm(F2))
    print('Error =', fmatrix_error(F, F2, focal))

def task_3RANSACN_example_real():
    img1 = cv2.imread('images/img1.jpg', 0)
    img2 = cv2.imread('images/img2.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    draw_matches(pts1, pts2, img1, img2)
    F1,inliers,errors = find_fmatrix_RANSAC(pts1, pts2, 10000)
    draw_matches(pts1[:,errors<1], pts2[:,errors<1], img1, img2)
    print(f'RANSAC inliers = {inliers}/{pts1.shape[1]}')

def task_3_iterations_test():
    focal = 1000
    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1100, 1200, 1300, 1400]
    errors = []
    for i in range(len(iterations)):
        error = 0.0
        for j in range(100):
            pts1, pts2, F = generate_3d_points(num=100, noutliers=50, noise=0.5, focal=focal, spherical=True)
            F2,_,_ = find_fmatrix_RANSAC(pts1, pts2, iterations[i])
            error += fmatrix_error(F, F2, focal)
        errors.append(error / 100)
        print(iterations[i], " iterations result in average error of: ", error/100)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, errors, marker='o')
    plt.xlabel("RANSAC Iterations")
    plt.ylabel("Fundamental Matrix Error")
    plt.title("Fundamental Matrix Error vs. RANSAC Iterations")
    plt.grid(True)
    plt.show()


if __name__=="__main__":
    np.set_printoptions(precision = 3)

    ## Task 3
    #task_3_example_syn()
    #task_3_example_real()
    #task_3_outlier_test()
    #task_3RANSACN_example_syn()
    task_3RANSACN_example_real()
    #task_3_iterations_test()

    ## Task 4

    ## Task 4