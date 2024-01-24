"""

Fit a line/curve to the segmentation mask
https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
TODO: Move this to utils

__author__ = "P Rajan"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""
from sklearn.linear_model import (LinearRegression, HuberRegressor,
                              	RANSACRegressor, TheilSenRegressor)

from sklearn.preprocessing import StandardScaler

import numpy as np
import cv2
import time
import os
import datetime as datetime
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
sys.path.append(os.path.join('..', 'Utils'))

def line_fit():
    model_rgrsn = HuberRegressor(epsilon=1)
    N_SAMPLES = 500
    N_OUTLIERS = 25

    X, y, coef = datasets.make_regression(
        n_samples=N_SAMPLES,
        n_features=1,
        n_informative=1,
        noise=20,
        coef=True,
        random_state=42
    )
    coef_list = [["original_coef", float(coef)]]

    # add outliers          	 
    np.random.seed(42)
    X[:N_OUTLIERS] = 10 + 0.75 * np.random.normal(size=(N_OUTLIERS, 1))
    y[:N_OUTLIERS] = -15 + 20 * np.random.normal(size=N_OUTLIERS)

    plt.scatter(X, y)
    lr = LinearRegression().fit(X, y)
    coef_list.append(["linear_regression", lr.coef_[0]])

    plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)

    fit_df = pd.DataFrame(
        index = plotline_X.flatten(),
        data={"linear_regression": lr.predict(plotline_X)}
    )
    fix, ax = plt.subplots()
    fit_df.plot(ax=ax)
    plt.scatter(X, y, c="k")
    plt.title("Linear regression on data with outliers")
    print('Done')

def fit_line(mask):
    
    mask[mask==2] = 0
    mask[mask==3] = 0
    
    y, X = np.where(mask==1)
    # # y = np.expand_dims(y, axis=1)
    X = np.expand_dims(X, axis=1)
    
    # plt.scatter(X, y)
    lr = LinearRegression().fit(X, y)
    hr = HuberRegressor().fit(X, y)
    rr = RANSACRegressor(random_state=42).fit(X, y)   

    XX = np.squeeze(X)
    outlier_mask = np.logical_not(rr.inlier_mask_)

    plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)
    plotline_y = hr.predict(plotline_X)

    hr_x = plotline_X 
    hr_y = plotline_X * hr.coef_ + hr.intercept_
    
    plt.plot(plotline_X, plotline_y, 'bo')
    plt.plot(hr_x, hr_y, 'g-')
    plt.title("Regression result") 
    plt.show()
    st = [int(X[0]), int(y[0])]
    ed = [int(X[-1]), int(y[-1])]
    return st, ed, hr.coef_, hr.intercept_

def line_fit_mask():
    
    # Load the annotaion mask
    root_dir = "E:\\Data\\Needles\\Current_Data\\testData\\"
    test_green = ['G1_test']#, 'NNConfRoom_test']

    result_dir_root = "E:\\Data\\Needles\\Linefit_Results_{}".format("1")

    model_rgrsn = HuberRegressor(epsilon=1)
    
    for test in test_green:
        TESTDIR = os.path.join(root_dir, test)
        RESULTDIR = os.path.join(result_dir_root, test)
        if not os.path.isdir(RESULTDIR):
            os.makedirs(RESULTDIR, exist_ok=True)
        RESULTDIR_MASK = os.path.join(RESULTDIR, 'mask')
        if not os.path.isdir(RESULTDIR_MASK):
            os.makedirs(RESULTDIR_MASK, exist_ok=True)

        testImages = [f for f in os.listdir(TESTDIR) if f.endswith(".png") or f.endswith(".jpg")]
        for img in testImages:
            image = cv2.imread(os.path.join(TESTDIR, img))
            # image = cv2.resize(image, dsize=(704, 1280))
            img_png = img.replace('jpg', 'png')
            mask_gt = cv2.imread(os.path.join(TESTDIR, 'annotations_hubtipshaft', img_png), cv2.IMREAD_GRAYSCALE) #annotations_hubtipshaft
            # mask_gt_cv = cv2.resize(mask_gt_cv, dsize=(704, 1280))
            
            mask_gt[mask_gt==2] = 0
            mask_gt[mask_gt==3] = 0
            
            y, X = np.where(mask_gt==1)
            # y = np.expand_dims(y, axis=1)
            X = np.expand_dims(X, axis=1)
            
            # plt.scatter(X, y)
            lr = LinearRegression().fit(X, y)
            hr = HuberRegressor().fit(X, y)
            rr = RANSACRegressor(random_state=42).fit(X, y)

            # coef_list.append(["linear_regression", lr.coef_[0]])

            plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)
            plotline_y = rr.predict(plotline_X)
            fit_df = pd.DataFrame(
                index = plotline_X.flatten(),
                data={"regression": plotline_y}
            )
            fix, ax = plt.subplots()
            fit_df.plot(ax=ax)
            plt.scatter(X, y, c="k")
            plt.title("Regression result")

            # Overlay the result on the mask image
            plotline_X = [int(i) for i in plotline_X]
            plotline_y = [int(i) for i in plotline_y]
            np.expand_dims(X, axis=1)
            mask_gt[plotline_y, plotline_X] = 10

            splash_line = color_masks(image, mask, mask_l)
            # splash = cv2.cvtColor(splash, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(RESULTDIR, img), splash_line)
            cv2.imwrite(os.path.join(RESULTDIR_MASK, img), mask_l*255)
 

if __name__ == "__main__":
    # line_fit_mask()

    image_dir = "Z:\\Needles\\Lumena\\test_data\\PR\\masks_cropped"
    img_name = "im_20230720-142855.560876_R.png"

    img = cv2.imread(os.path.join(image_dir, img_name), cv2.IMREAD_GRAYSCALE)    
    plt.imshow(img,'gray')
    plt.show()
    a, b, c, d = fit_line(img)

    print(c, d)