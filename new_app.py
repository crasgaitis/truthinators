import statistics
import numpy as np
# assume we have 0-5 (6 different emotions) from both face emotion and text emotion models
# check if the two models agree
# probability distribution - 
# [0.1, 0.1, 0.2, 0.4, 0., 0.6] .. 6 of these values]
# 

# def check_if_true(sensitivity. model1_out, model2_out):
# sensitivity should be a scalar from 0 to 1
# returns 1 for shock, 0 for no shock
def check_if_true(sensitivity, model1_out, model2_out):
    shock = 0
    stdeviation1 = np.std(model1_out)
    stdeviation2 = np.std(model2_out)
    avg = 100.0 / 6.0
    model1_bool = model1_out[model1_out <= np.abs(avg - stdeviation1)]
    model2_bool = model2_out[model2_out <= np.abs(avg - stdeviation2)]

    threshhold = sensitivity * 6

    avg_length = (model1_bool + model2_bool) / 2.0
    if(avg_length < threshhold):
        shock = 1

    return shock






