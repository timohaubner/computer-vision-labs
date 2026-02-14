import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_2d_points, extract_and_match_SIFT
from plots import draw_homography,draw_matches

def find_homography(pts1:np.ndarray, pts2:np.ndarray) -> np.ndarray:
    '''Find the homography matrix from matching points in two images.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :returns np.ndarray H: a 3x3 array representing the homography matrix H.

    '''

    # Use image positions of matching pairs to create a matrix A
    xa = pts1[0, 0]
    ya = pts1[1, 0]
    xb = pts2[0, 0]
    yb = pts2[1, 0]
    A = np.array([[xa, ya, 1, 0, 0, 0, -xa * xb, -ya * xb, -xb],
                  [0, 0, 0, xa, ya, 1, -xa * yb, -ya * yb, -yb]])

    for i in range(1, np.shape(pts1)[1]):
        xa = pts1[0, i]
        ya = pts1[1, i]
        xb = pts2[0, i]
        yb = pts2[1, i]
        curr_eq = np.array([[xa, ya, 1, 0, 0, 0, -xa * xb, -ya * xb, -xb],
                            [0, 0, 0, xa, ya, 1, -xa * yb, -ya * yb, -yb]])
        A = np.vstack((A, curr_eq))

    # Compute square of A, C = transpose(A) * A
    C = np.transpose(A) @ A

    # Find eigenvector h of smallest eigenvalue lambda of C
    U, S, Vh = np.linalg.svd(C)
    h = Vh[-1, :]

    # Rearrange 9-vector h into a 3x3 homography H
    H = h.reshape(3, 3)

    return H


def homography_error(H1:np.ndarray, H2:np.ndarray, focal:float = 1000) -> float:
    '''Computes the error between two homographies, wrt a known focal.
        :param np.ndarray H1: a 3x3 matrix representing one of the homographies.
        :param np.ndarray H2: a 3x3 matrix representing the second homography.
        :param float focal: the known focal length.
        :returns float: the error between the homographies.
    '''
    H_diff = H1/H1[2,2] - H2/H2[2,2]
    return np.linalg.norm(np.diag((1/focal,1/focal,1)) @ H_diff @ np.diag((focal,focal,1)))


def count_homography_inliers(H:np.ndarray, pts1:np.ndarray, pts2:np.ndarray, thresh:float = 1.0) -> tuple[int,np.ndarray]:
    '''Given the homography H, projects pts1 on the second image, counting the number of actual points in pts2 for which the projection error is smaller than the given threshold.

        :param np.ndarray H: a 3x3 matrix containing the homography matrix.
        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param float thresh: the threshold to consider points as inliers.
        :returns int ninliers, np.ndarray errors:
            ninliers: the number of inliers.
            errors: a N_points array containing the errors; they are indexed as pts1 and pts2.
    
    '''
    Hp1 = H @ np.vstack((pts1, np.ones((1, np.size(pts1, axis=1)))))
    errors = np.sqrt(np.sum((Hp1[0:2,:]/Hp1[2,:] - pts2)**2, axis=0))
    ninliers = np.sum(np.where(errors<thresh**2, 1, 0))
    return ninliers, errors


def find_homography_RANSAC(pts1:np.ndarray, pts2:np.ndarray, niter:int = 100, thresh:float = 1.0) ->tuple[np.ndarray,int,np.ndarray]:
    '''Computes the best homography for matching points pts1 and pts2, adopting RANSAC.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param int niter: the number of RANSAC iteration to run.
        :param float thresh: the maximum error to consider a point as an inlier while evaluating a RANSAC iteration.
        :returns np.ndarray Hbest, int ninliers, np.ndarray errors:
            Hbest: a 3x3 matrix representing the best homography found.
            ninliers: the number of inliers for the best homography found.
            errors: a N_points array containing the errors for the best homography found; they are indexed as pts1 and pts2.
    
    '''

    Hbest = None
    errors = None
    ninliers = 0

    # Loop niter times
    for i in range(niter):

        # Generate a minimum set of 4 random feature matches
        idx = np.random.choice(pts1.shape[1], 4, replace=False)
        pts1_subsample = pts1[:, idx]
        pts2_subsample = pts2[:, idx]

        # Find homography H using these matches
        H_curr = find_homography(pts1_subsample, pts2_subsample)
        ninliers_curr, errors_curr = count_homography_inliers(H_curr, pts1, pts2, thresh)

        if Hbest is None or ninliers_curr > ninliers:
            ninliers = ninliers_curr
            errors = errors_curr
            Hbest = H_curr


    return Hbest, ninliers, errors


def synthetic_example(RANSAC = False):
    focal = 1000
    pts1, pts2, H = generate_2d_points(num = 100, noutliers = 0, noise=0.5, focal = focal)
    draw_matches(pts1, pts2)
    print('True H =\n', H)
    if RANSAC:
        H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
        H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
        print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
        print('RANSAC H =\n', H1)
        print('Final estimated H =\n', H2)
    else:
        H2 = find_homography(pts1, pts2)
        print('Estimated H =\n', H2)
    print('Error =', homography_error(H, H2, focal))


def real_example():
    img1 = cv2.imread('images/img1.jpg', 0)
    img2 = cv2.imread('images/img2.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
    H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
    print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
    print('RANSAC H =\n', H1)
    print('Final estimated H =\n', H2)
    draw_homography(img1, img2, H2)


# Task 1
############################

def task_1_example_syn():
    focal = 1000
    pts1, pts2, H = generate_2d_points(num=20, noutliers=0, noise=0.5, focal=focal)
    draw_matches(pts1, pts2)
    print('True H =\n', H)
    H2 = find_homography(pts1, pts2)
    print('Estimated H =\n', H2)
    print('Error =', homography_error(H, H2, focal))

def task_1_example_real():
    img1 = cv2.imread("images/books1.jpg", flags=0)
    img2 = cv2.imread("images/books2.jpg", flags=0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num=20)
    H = find_homography(pts1, pts2)
    draw_matches(pts1, pts2, img1, img2)
    print(H)
    draw_homography(img1, img2, H)

def task_1_noise_test():
    focal = 1000
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    errors = []
    for i in range(len(noise_levels)):
        error = 0.0
        for j in range(10):
            pts1, pts2, H = generate_2d_points(num=100, noutliers=0, noise=noise_levels[i], focal=focal)
            H2 = find_homography(pts1, pts2)
            error += homography_error(H, H2, focal)
        errors.append(error/10)

    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, errors, marker='o')
    plt.xlabel("Noise Level")
    plt.ylabel("Homography Error")
    plt.title("Homography Error vs. Noise Level")
    plt.grid(True)
    plt.show()

def task_1_outlier_test():
    focal = 1000
    outlier_levels = [0, 5, 10, 15, 20, 25, 30]
    errors = []
    for i in range(len(outlier_levels)):
        error = 0.0
        for j in range(10):
            pts1, pts2, H = generate_2d_points(num=100, noutliers=outlier_levels[i], noise=0.5, focal=focal)
            H2 = find_homography(pts1, pts2)
            error += homography_error(H, H2, focal)
        errors.append(error/10)

    plt.figure(figsize=(8, 5))
    plt.plot(outlier_levels, errors, marker='o')
    plt.xlabel("Outliers")
    plt.ylabel("Homography Error")
    plt.title("Homography Error vs. Outliers")
    plt.grid(True)
    plt.show()

# Task 2
############################

def task_2_example_syn():
    focal = 1000
    pts1, pts2, H = generate_2d_points(num = 100, noutliers = 20, noise=0.5, focal = focal)
    draw_matches(pts1, pts2)
    print('True H =\n', H)
    H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
    H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
    print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
    print('RANSAC H =\n', H1)
    print('Final estimated H =\n', H2)
    print('Error =', homography_error(H, H2, focal))

def task_2_example_real():
    img1 = cv2.imread('images/img3.jpg', 0)
    img2 = cv2.imread('images/img4.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
    H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
    print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
    print('RANSAC H =\n', H1)
    print('Final estimated H =\n', H2)
    draw_homography(img1, img2, H2)

def task_2_outlier_test():
    focal = 1000
    outlier_levels = [5, 10, 20, 30, 40, 50, 60, 70, 80]
    errors = []
    for i in range(len(outlier_levels)):
        error = 0.0
        # Formula for iterations adapted from slides
        epsilon = max(outlier_levels[i] / 100, 0.2)
        iter = int(np.ceil(np.log(1 - 0.99) / np.log(1 - (1 - epsilon) ** 4)))
        print(iter, "iterations for outlier amount of: ", outlier_levels[i])
        for j in range(100):
            pts1, pts2, H = generate_2d_points(num=100, noutliers=outlier_levels[i], noise=0.0, focal=focal)
            H2, _, _ = find_homography_RANSAC(pts1, pts2, iter)
            error += homography_error(H, H2, focal)
        errors.append(error/100)

    plt.figure(figsize=(8, 5))
    plt.plot(outlier_levels, errors, marker='o')
    plt.xlabel("Outliers")
    plt.ylabel("Homography Error")
    plt.title("Homography Error vs. Outliers with RANSAC")
    plt.grid(True)
    plt.show()

def task_2_iterations_test():
    focal = 1000
    iterations = [50, 100, 150, 200, 250, 300, 350]
    errors = []
    for i in range(len(iterations)):
        error = 0.0
        for j in range(100):
            pts1, pts2, H = generate_2d_points(num=100, noutliers=50, noise=0.5, focal=focal)
            H2, _, _ = find_homography_RANSAC(pts1, pts2, iterations[i])
            error += homography_error(H, H2, focal)
        errors.append(error / 100)
        print(iterations[i], " iterations result in average error of: ", error/100)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, errors, marker='o')
    plt.xlabel("RANSAC Iterations")
    plt.ylabel("Homography Error")
    plt.title("Homography Error vs. RANSAC Iterations")
    plt.grid(True)
    plt.show()

def task_2_example_not_flat():
    img1 = cv2.imread('images/books1.jpg', 0)
    img2 = cv2.imread('images/books2.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    H1, ninliers, errors = find_homography_RANSAC(pts1, pts2, niter=10000, thresh=1.0)
    H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
    print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
    print('RANSAC H =\n', H1)
    print('Final estimated H =\n', H2)
    draw_homography(img1, img2, H2)

# Task 3
############################




if __name__=="__main__":
    np.set_printoptions(precision = 3)

    # Task 1
    #task_1_example_syn()
    #task_1_example_real()
    #task_1_noise_test()
    #task_1_outlier_test()

    ## Task 2
    #task_2_example_syn()
    #task_2_example_real()
    #task_2_outlier_test()
    #task_2_iterations_test()
    task_2_example_not_flat()
