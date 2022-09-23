import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def combine_images(img1, img2, h):
    nrow1, ncol1 = img1.shape[0], img1.shape[1]
    nrow2, ncol2 = img2.shape[0], img2.shape[1]

    # img1 on the right/bottom, img2 on the left/top
    points1 = np.array([[0, 0], [0, nrow1],
                           [ncol1, nrow1],
                           [ncol1, 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = np.array([[0, 0], [0, nrow2],
                           [ncol2, nrow2],
                           [ncol2, 0]], dtype=np.float32)
    points2 = points2.reshape((-1, 1, 2))

    points_tmp = cv2.perspectiveTransform(points1, h)
    points_combine = np.concatenate((points2, points_tmp), axis=0)

    # compute translation for overlap region
    [x_min, y_min] = (points_combine.min(axis=0).ravel() - 0.5).astype(np.int32)
    [x_max, y_max] = (points_combine.max(axis=0).ravel() + 0.5).astype(np.int32)
    h_translation = np.float64([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # align the images and add that
    output_img = cv2.warpPerspective(img1, h_translation.dot(h),
                                     (x_max - x_min, y_max - y_min))

    # align the images and add that
    img2_wp = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    img2_wp[-y_min: (-y_min + nrow2), -x_min:(-x_min + ncol2), :] = img2

    # remove black stitching lines
    mask1 = cv2.threshold(output_img, 0, 255, cv2.THRESH_BINARY)[1]
    mask1_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_ERODE, mask1_kernel)
    output_img[mask1 == 0] = 0
    output_img_final = np.where(output_img != 0, output_img, img2_wp)

    return output_img_final

class Stitcher:
    """ function to stitch a series of images along one direction"""

    def __init__(self, n_features=10000, detect_method='orb',
                 matcher_method = 'flann', affine_only=True,
                 verbose=False, x_move_default=None, y_move_default=None):

        # define feature detection method
        if detect_method == 'sift':
            self.detector = cv2.SIFT_create(n_features)
            distance = cv2.NORM_L1
            index_params = dict(algorithm=0,
                                trees = 5)

        elif detect_method == 'orb':
            self.detector = cv2.ORB_create(n_features)
            distance = cv2.NORM_HAMMING
            index_params = dict(algorithm=6,
                                table_number=6,
                                key_size=12, #20
                                multi_probe_level=1) #2

        # define the hyperparameters for knn feature matcher for ratio testing
        self.lowe = 0.7
        self.knn_clusters = 2
        self.matcher_method = matcher_method
        self.affine_only = affine_only
        self.min_feature_counts = 4

        # define methods for matching features
        if 'bf' in self.matcher_method:
            self.good_match_percent = 0.5
            self.matcher = cv2.BFMatcher(distance, crossCheck=False)
        else:
            self.matcher = cv2.FlannBasedMatcher(index_params, {'checks': 50})

        self.dst_image = None
        self.dst_image_gray = None
        self.verbose = verbose

        # the default pixel movement in horizontal and vertical directions
        self.x_move_default = x_move_default
        self.y_move_default = y_move_default

    def compute_matches(self, img1, img2, direction, fraction):

        # detect features
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        # crop out the relevant fraction, which is likely containing the overlap region for feature detection
        if direction == 'horizontal':
            # define search roi if it's stiching horizontally
            # img1 is on the right and img2 is on the left
            img1_sub = img1[:,:int(w1*fraction)]
            img2_sub = img2[:, -int(w1*fraction):]
            self.h_offset = [int(w2-w1*fraction), 0]

        elif direction == 'vertical':
            # define search roi if it's stiching vertically
            # img1 is on the bottom and img2 is on the top
            img1_sub = img1[:int(h1*fraction),:]
            img2_sub = img2[-int(h1*fraction):, :]
            self.h_offset = [0, int(h2-h1*fraction)]

        img1_mask = None
        img2_mask = None

        keypoints1_sub, descriptors1_sub = self.detector.detectAndCompute(img1_sub, mask=img1_mask)
        keypoints2_sub, descriptors2_sub = self.detector.detectAndCompute(img2_sub, mask=img2_mask)

        # if one of the input images doesn't contain more than 3 valid features
        # (require 3 valida feature pairs to compute affine transform)

        if descriptors1_sub is None or descriptors2_sub is None or \
            len(descriptors1_sub) < self.min_feature_counts or \
                len(descriptors2_sub) < self.min_feature_counts:
            return None, None, None

        if self.verbose:
            print(f'features img1 = {len(descriptors1_sub)}')
            print(f'features img2 = {len(descriptors2_sub)}')

        if 'bf' in self.matcher_method:
            matches_sub = self.matcher.match(descriptors1_sub, descriptors2_sub, None)
            numGoodMatches = int(len(matches_sub) * self.good_match_percent)
            good_matches_sub = matches_sub[:numGoodMatches]
        else:
            # apply ratio testing feature matching using FLANN or BF
            matches_sub = self.matcher.knnMatch(descriptors1_sub, descriptors2_sub, k=self.knn_clusters)

            # extract good matches with lowe test
            good_matches_sub = []
            for match_sub in matches_sub:
                try:
                    match1_sub, match2_sub = match_sub
                except:
                    continue
                if match1_sub.distance < self.lowe * match2_sub.distance:
                    good_matches_sub.append(match1_sub)

        if len(good_matches_sub) < self.min_feature_counts:
            return None, None, None

        if self.verbose:
            print(f'good matched features = {len(good_matches_sub)}')


        points1_sub = np.array([keypoints1_sub[good_match_sub.queryIdx].pt for good_match_sub in good_matches_sub],
                           dtype=np.float32)
        points2_sub = np.array([keypoints2_sub[good_match_sub.trainIdx].pt for good_match_sub in good_matches_sub],
                           dtype=np.float32)

        points1_sub = points1_sub.reshape((-1, 1, 2))
        points2_sub = points2_sub.reshape((-1, 1, 2))

        return points1_sub, points2_sub, len(good_matches_sub)


    def add_image(self, image, direction='horizontal', fraction=1/4, store_dst=True):

        self.direction = direction
        # convert image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.dst_image is None:
            # no stitching happening for the first image
            self.dst_image = image
            self.dst_image_gray = image_gray
            return self.dst_image, None

        # detect and match features between images to be stitched
        # specify the direction to define the roi for feature detection
        # horizontal : left to right; vertical: top to bottom
        matches_src_sub, matches_dst_sub, n_matches_sub = self.compute_matches(image_gray,
                                    self.dst_image_gray,
                                    direction,
                                    fraction
                                    )

        # register/align and blend images
        if matches_src_sub is not None and matches_dst_sub is not None and n_matches_sub is not None:
            # find homography: map from source image to dst image
            if self.affine_only:
                h, mask = cv2.estimateAffinePartial2D(matches_src_sub, matches_dst_sub, method=cv2.RANSAC,
                                                      ransacReprojThreshold=5.0)
                h[:, -1] += self.h_offset
                h = np.vstack([h,[0,0,1]])
            else:
                h, mask = cv2.findHomography(matches_src_sub, matches_dst_sub, cv2.RANSAC, 5.0)
                h[:-1, -1] += self.h_offset

        else:
            print('Not enough good features for matching, use direct concatenation')
            assert False

        if self.verbose:
            print(f'H transform = {h}')

        # align and blend images
        dst_image = combine_images(image, self.dst_image, h)
        # convert rgb dst image to grayscale for next iteration
        dst_image_gray = cv2.cvtColor(dst_image, cv2.COLOR_RGB2GRAY)
        if store_dst:
            self.dst_image = dst_image
            self.dst_image_gray = dst_image_gray

        return dst_image, h


def ImageStitching(image_folder, board_image_patches, stitch_order='horizontal_vertical',
                   detector_method='sift', matcher_method='flann', overlap_fraction=0.25,
                   verbose=True, save=True, innerloop_nfeatures = 2500, outterloop_nfeatures = 5000,
                   ):
    """ wrapper function to stitch all images in the folder into two board images: board_image_rgb, board_image_white """

    inner_direction, outter_direction = stitch_order.split('_')

    # stitch board image rgb first
    # start inner stitching loop to stitch all images along the same row or column

    x_move_default = int(2368)
    y_move_default = int(2155)
    direct_concat_inner = False
    direct_concat_outter = False

    inner_stitched_images = []
    all_inner_h_lists = []
    for i in range(len(board_image_patches)):
        # initialise stitcher
        inner_stitcher = Stitcher(detect_method=detector_method, matcher_method=matcher_method,
                                n_features=innerloop_nfeatures, verbose=verbose)

        inner_image_files = board_image_patches[i]
        inner_h_list_i = []
        try:
            for j, image_file in enumerate(inner_image_files):

                # if i == 0 and j == 0:
                #     assert False
                # read image to be stitched
                image_file_path = os.path.join(image_folder, image_file)
                image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

                # stitch new image to the existing image along inner_direction
                stitched_image, h = inner_stitcher.add_image(image, direction=inner_direction,
                                                                    fraction=overlap_fraction)
                if j > 0:
                    inner_h_list_i.append(h)

                # display stitched row image
                if verbose:
                    stitched_image_view = cv2.resize(stitched_image, dsize=(0, 0), fx=0.1, fy=0.1)
                    stitched_image_view = cv2.cvtColor(stitched_image_view, cv2.COLOR_BGR2RGB)
                    plt.figure()
                    plt.imshow(stitched_image_view, cmap='gray')
                    plt.title(f'inner stitched image {image_file}')
                    plt.show()

                    print(f'finished stitching: {image_file}')
        except:
            direct_concat_inner = True

        # store stitched images in inner loop
        inner_stitched_images.append(stitched_image)
        all_inner_h_lists.append(inner_h_list_i)

    if direct_concat_inner == True:
        print('Not enough good features, concatenate pictures')
        inner_stitched_images = []
        # compute the pixel movement from previously computed high quality transformation matrices or default value
        if inner_direction == 'horizontal':
            if len(all_inner_h_lists) == 0:
                x_move_pixel_inner = int(x_move_default)
            else:
                x_move_pixel_from_goodh = []
                for inner_h_list_i in all_inner_h_lists:
                    if len(inner_h_list_i) > 0:
                        for j, h in enumerate(inner_h_list_i):
                            if abs(h[1, 2]) < 500:
                                x_move_pixel_from_goodh.append(h[0, 2] / (j + 1))

                # x_move_pixel_from_goodh = [[h[0, 2] / (j + 1) for j, h in enumerate(inner_h_list_i) if
                #                        abs(h[1, 2]) < 500] for inner_h_list_i in all_inner_h_lists if len(inner_h_list_i)>0]
                x_move_pixel_from_goodh_arr = np.array(x_move_pixel_from_goodh).flatten()
                x_move_pixel_inner = int(np.mean(x_move_pixel_from_goodh_arr))
            print(f'{inner_direction}: {x_move_pixel_inner}')
        else:
            if len(all_inner_h_lists) == 0:
                y_move_pixel_inner = int(y_move_default)
            else:
                y_move_pixel_from_goodh = [[h[1, 2] / (j + 1) for j, h in enumerate(inner_h_list_i) if
                                       abs(h[0, 2]) < 500] for inner_h_list_i in all_inner_h_lists]
                y_move_pixel_from_goodh_arr = np.array(y_move_pixel_from_goodh).flatten()
                y_move_pixel_inner = int(np.mean(y_move_pixel_from_goodh_arr))
            print(f'{inner_direction}: {y_move_pixel_inner}')

        for i in range(len(board_image_patches)):
            inner_image_files = board_image_patches[i]

            for j, image_file in enumerate(inner_image_files):

                # read image to be stitched
                image_file_path = os.path.join(image_folder, image_file)
                image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
                if j == 0:
                    stitched_image = image
                else:
                    if inner_direction == 'horizontal':
                        x_offset = x_move_pixel_inner * j
                        # directly concat image horizontally
                        stitched_image = np.hstack([stitched_image[:, :x_offset], image])
                    else:
                        y_offset = y_move_pixel_inner * j
                        # directly concat image vertically
                        stitched_image = np.vstack([stitched_image[:y_offset, :], image])

                # display stitched row image
                if verbose:
                    stitched_image_view = cv2.resize(stitched_image, dsize=(0, 0), fx=0.1, fy=0.1)
                    stitched_image_view = cv2.cvtColor(stitched_image_view, cv2.COLOR_BGR2RGB)
                    plt.figure()
                    plt.imshow(stitched_image_view, cmap='gray')
                    plt.title(f'inner stitched image {image_file}')
                    plt.show()

            # store stitched images in inner loop
            inner_stitched_images.append(stitched_image)

    # start outter stitching loop to stitch all stitched rows or columns into one board image
    outter_stitcher = Stitcher(detect_method=detector_method, matcher_method=matcher_method,
                               n_features=outterloop_nfeatures, verbose=verbose)
    outter_h_lists = []
    for k, stitched_image in enumerate(inner_stitched_images):

        # stitch new row/column image together
        try:
            # if k == 1:
            #     assert False
            stitched_board, outter_h = outter_stitcher.add_image(stitched_image, direction=outter_direction,
                                                           fraction=overlap_fraction)
            if k > 0:
                outter_h_lists.append(outter_h)
        except:
            print('Not enough good features at outter loop, concatenate all row and columns')
            direct_concat_outter = True

        # display stitched row image
        if verbose:
            stitched_board_view = cv2.resize(stitched_board, dsize=(0, 0), fx=0.1, fy=0.1)
            stitched_board_view = cv2.cvtColor(stitched_board_view, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(stitched_board_view, cmap='gray')
            plt.title(f'outter stitched image {k}')
            plt.show()

    if direct_concat_outter == True:

        if direct_concat_inner == False:
            inner_stitched_images = []

            # compute the pixel movement from previously computed high quality transformation matrices or default value
            if inner_direction == 'horizontal':
                if len(all_inner_h_lists) == 0:
                    x_move_pixel_inner = int(x_move_default)
                else:
                    x_move_pixel_from_goodh = [[h[0, 2] / (j + 1) for j, h in enumerate(inner_h_list_i) if
                                           abs(h[1, 2]) < 500] for inner_h_list_i in all_inner_h_lists]
                    x_move_pixel_from_goodh_arr = np.array(x_move_pixel_from_goodh).flatten()
                    x_move_pixel_inner = int(np.mean(x_move_pixel_from_goodh_arr))
                print(f'{inner_direction}: {x_move_pixel_inner}')
            else:
                if len(all_inner_h_lists) == 0:
                    y_move_pixel_inner = int(y_move_default)
                else:
                    y_move_pixel_from_goodh = [[h[1, 2] / (j + 1) for j, h in enumerate(inner_h_list_i) if
                                           abs(h[0, 2]) < 500] for inner_h_list_i in all_inner_h_lists]
                    y_move_pixel_from_goodh_arr = np.array(y_move_pixel_from_goodh).flatten()
                    y_move_pixel_inner = int(np.mean(y_move_pixel_from_goodh_arr))
                print(f'{inner_direction}: {y_move_pixel_inner}')

            for i in range(len(board_image_patches)):
                inner_image_files = board_image_patches[i]

                for j, image_file in enumerate(inner_image_files):

                    # read image to be stitched
                    image_file_path = os.path.join(image_folder, image_file)
                    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
                    if j == 0:
                        stitched_image = image
                    else:
                        if inner_direction == 'horizontal':
                            x_offset = x_move_pixel_inner * j
                            # directly concat image horizontally
                            stitched_image = np.hstack([stitched_image[:, :x_offset], image])
                        else:
                            y_offset = y_move_pixel_inner * j
                            # directly concat image vertically
                            stitched_image = np.vstack([stitched_image[:y_offset, :], image])

                    # display stitched row image
                    if verbose:
                        stitched_image_view = cv2.resize(stitched_image, dsize=(0, 0), fx=0.1, fy=0.1)
                        stitched_image_view = cv2.cvtColor(stitched_image_view, cv2.COLOR_BGR2RGB)
                        plt.figure()
                        plt.imshow(stitched_image_view, cmap='gray')
                        plt.title(f'inner stitched image {image_file}')
                        plt.show()

                # store stitched images in inner loop
                inner_stitched_images.append(stitched_image)

        if outter_direction == 'horizontal':
            if len(outter_h_lists) == 0:
                x_move_pixel_outter = int(x_move_default)
            else:
                x_move_pixel_from_goodh = [h[0, 2] / (j + 1) for j, h in enumerate(outter_h_lists) if
                                       abs(h[1, 2]) < 500]
                x_move_pixel_outter = int(np.mean(x_move_pixel_from_goodh))
            print(f'{outter_direction}: {x_move_pixel_outter}')

        else:
            if len(outter_h_lists)== 0:
                y_move_pixel_outter = int(y_move_default)
            else:
                y_move_pixel_from_goodh = [h[1, 2] / (j + 1) for j, h in enumerate(outter_h_lists) if
                                       abs(h[0, 2]) < 500]
                y_move_pixel_outter = int(np.mean(y_move_pixel_from_goodh))
            print(f'{outter_direction}: {y_move_pixel_outter}')

        for k, stitched_image in enumerate(inner_stitched_images):

            if k == 0:
                stitched_board = stitched_image
            else:
                if outter_direction == 'horizontal':
                    x_offset = x_move_pixel_outter * k
                    # directly concat image horizontally
                    stitched_board = np.hstack([stitched_board[:, :x_offset], stitched_image])
                else:
                    y_offset = y_move_pixel_outter * k
                    # directly concat image vertically
                    stitched_board = np.vstack([stitched_board[:y_offset, :], stitched_image])

            # display stitched row image
            if verbose:
                stitched_board_view = cv2.resize(stitched_board, dsize=(0, 0), fx=0.1, fy=0.1)
                stitched_board_view = cv2.cvtColor(stitched_board_view, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.imshow(stitched_board_view, cmap='gray')
                plt.title(f'outter stitched image {k}')
                plt.show()

    if save:
        # save final stitched board image
        output_name = os.path.join(image_folder, f"board_rgb1.jpg")
        cv2.imwrite(output_name, stitched_board)

    board_image_rgb = stitched_board
    print('finished rgb stitching')

    # stitch board image white based on board image rgb transformation matrices
    board_image_white_patches = [[image_patch.replace('rgb', 'white') for image_patch in inner_patches] for inner_patches in board_image_patches]

    inner_stitched_images = []
    if direct_concat_inner:
        # start inner stitching loop to stitch all images along the same row or column
        for i in range(len(board_image_white_patches)):

            inner_image_files = board_image_white_patches[i]
            stitched_image = None

            for j, image_file in enumerate(inner_image_files):

                # read image to be stitched
                image_file_path = os.path.join(image_folder, image_file)
                image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

                if stitched_image is None:
                    # no stitching happen for the first image
                    stitched_image = image

                else:
                    if inner_direction == 'horizontal':
                        x_offset = x_move_pixel_inner * j
                        # directly concat image horizontally
                        stitched_image = np.hstack([stitched_image[:, :x_offset], image])
                    else:
                        y_offset = y_move_pixel_inner * j
                        # directly concat image vertically
                        stitched_image = np.vstack([stitched_image[:y_offset, :], image])

                # display stitched row image
                if verbose:
                    stitched_image_view = cv2.resize(stitched_image, dsize=(0, 0), fx=0.1, fy=0.1)
                    stitched_image_view = cv2.cvtColor(stitched_image_view, cv2.COLOR_BGR2RGB)
                    plt.figure()
                    plt.imshow(stitched_image_view, cmap='gray')
                    plt.title(f'inner stitched image {image_file}')
                    plt.show()

                    print(f'finished stitching: {image_file}')

            # store stitched images in inner loop
            inner_stitched_images.append(stitched_image)

    else:
        # start inner stitching loop to stitch all images along the same row or column
        for i in range(len(board_image_white_patches)):

            inner_image_files = board_image_white_patches[i]
            stitched_image = None
            inner_h_lists_i = all_inner_h_lists[i]

            for j, image_file in enumerate(inner_image_files):

                # read image to be stitched
                image_file_path = os.path.join(image_folder, image_file)
                image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

                if stitched_image is None:
                    # no stitching happen for the first image
                    stitched_image = image

                else:
                    h_ij = inner_h_lists_i[j-1]
                    # stitch image based on transformation matrices learnt from rgb image
                    stitched_image = combine_images(image, stitched_image, h_ij)

                # display stitched row image
                if verbose:
                    stitched_image_view = cv2.resize(stitched_image, dsize=(0, 0), fx=0.1, fy=0.1)
                    stitched_image_view = cv2.cvtColor(stitched_image_view, cv2.COLOR_BGR2RGB)
                    plt.figure()
                    plt.imshow(stitched_image_view, cmap='gray')
                    plt.title(f'inner stitched image {image_file}')
                    plt.show()

                print(f'finished stitching: {image_file}')

            # store stitched images in inner loop
            inner_stitched_images.append(stitched_image)

    if direct_concat_outter:
        # start outter stitching loop to stitch all stitched rows or columns into one board image
        stitched_board = None
        for k, stitched_image in enumerate(inner_stitched_images):

            if stitched_board is None:
                stitched_board = stitched_image
            else:
                if outter_direction == 'horizontal':
                    x_offset = x_move_pixel_outter * k
                    # directly concat image horizontally
                    stitched_board = np.hstack([stitched_board[:, :x_offset], stitched_image])
                else:
                    y_offset = y_move_pixel_outter * k
                    # directly concat image vertically
                    stitched_board = np.vstack([stitched_board[:y_offset, :], stitched_image])
                # display stitched image
                if verbose:
                    stitched_board_view = cv2.resize(stitched_board, dsize=(0, 0), fx=0.1, fy=0.1)
                    stitched_board_view = cv2.cvtColor(stitched_board_view, cv2.COLOR_BGR2RGB)
                    plt.figure()
                    plt.imshow(stitched_board_view, cmap='gray')
                    plt.title(f'outter stitched image {k}')
                    plt.show()

    else:
        # start outter stitching loop to stitch all stitched rows or columns into one board image
        stitched_board = None
        for k, stitched_image in enumerate(inner_stitched_images):

            if stitched_board is None:
                stitched_board = stitched_image
            else:
                # stitch new row/column image together
                h_outter_k = outter_h_lists[k-1]
                stitched_board = combine_images(stitched_image, stitched_board, h_outter_k)

            # display stitched image
            if verbose:
                stitched_board_view = cv2.resize(stitched_board, dsize=(0, 0), fx=0.1, fy=0.1)
                stitched_board_view = cv2.cvtColor(stitched_board_view, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.imshow(stitched_board_view, cmap='gray')
                plt.title(f'outter stitched image {k}')
                plt.show()

    if save:
        # save final stitched board image
        output_name = os.path.join(image_folder, f"board_white1.jpg")
        cv2.imwrite(output_name, stitched_board)

    board_image_white = stitched_board
    print('finished white stitching')

    return board_image_rgb, board_image_white


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MPB3 for SMT Defect Classification')
    parser.add_argument('-path', '--image_path', type=str,
                        help='folder location for images to be stitched ')
    parser.add_argument('-nr', '--n_rows', default=4, type=int,
                        help='number of rows')
    parser.add_argument('-nc', '--n_column', default=4, type=int,
                        help='number of columns')

    args = parser.parse_args()
    image_path = args.image_path
    n_rows = args.n_rows
    n_cols = args.n_columns
    n_total = int(n_rows*n_cols)
    board_image_patches = np.array([f'rgb_{i}.jpg' for i in range(n_total)]).reshape([n_rows, n_cols])
    print(f'start {image_path} with {n_rows} rows and {n_cols} columns')
    board_image_rgb, board_image_white = ImageStitching(image_path, board_image_patches,
                                                        stitch_order='horizontal_vertical',
                                                        detector_method='sift', matcher_method='flann',
                                                        overlap_fraction=0.3,
                                                        verbose=False, innerloop_nfeatures=5000,
                                                        outterloop_nfeatures=5000,
                                                        )
