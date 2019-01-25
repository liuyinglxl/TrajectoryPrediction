# -*- coding: utf-8 -*-
# this is the module of my functions

import pandas as pd
from pandas import Series,DataFrame
from numpy import *

def deg_to_rad(deg):
    '''函数：输入度，返回弧度'''
    return deg*pi/180

# 传入预测的观测值的期望
def next_v_to_predict_next_position(x,y,v,theta,u=0,diata=0):
    '''
    用状态对应的v和上一点的theta进行计算
    返回得出的下一点的x,y值
    '''
    theta_rad = deg_to_rad(theta)
    diata_rad = deg_to_rad(diata)
    vx = v*cos(theta_rad)
    vy = v*sin(theta_rad)
    ux = u*cos(diata_rad)
    uy = u*sin(diata_rad)
    return x+vx+ux,y+vy+uy



# 向量化
vec_next_v_to_predict_next_position = vectorize(next_v_to_predict_next_position)

def distance(true_x,true_y,predict_x,predict_y):
    return sqrt((true_x-predict_x)**2+(true_y-predict_y)**2)

# 向量化
vec_distance = vectorize(distance)


def predict_next_n_points_arima1(last_point_df,n,predictions_v):
    """
    仅配合over_error1使用
    x,y,v,theta 原始点参数
    n 为int 预测步数
    model 为使用的模型
    state_to_v 为state与v的对应关系
    返回prediction_next_n_points ，为(x,y)
    """
#     print last_point_df
    x = array(last_point_df.x)
#     print x
    y = array(last_point_df.y)
    v = array(last_point_df.v)
    theta = array(last_point_df.theta)
    next_n_points_location = []
    for step in range(n):
        v = predictions_v[:,step]
        next_position = vec_next_v_to_predict_next_position(x,y,v,theta)
        next_n_points_location.append(next_position)
        x = next_position[0]
        y = next_position[1]
           
    return array(next_n_points_location)

def over_error_arima1(ctdf,predictions_v,last_point_num=1999,steps=20,len_of_trail=200):
    tmp = ctdf[ctdf.index%len_of_trail==last_point_num]
    next_n_points_location = predict_next_n_points_arima1(tmp,steps,predictions_v)
    X = next_n_points_location[:,0,:].flatten()
    Y = next_n_points_location[:,1,:].flatten()
    tmpX = array(ctdf[ctdf.index%len_of_trail == last_point_num + 1].x)
    for i in range(2,steps+1):
        tmpX = append(tmpX,array(ctdf[ctdf.index%len_of_trail == last_point_num+i].x))
    tmpY = array(ctdf[ctdf.index%len_of_trail == last_point_num].y)
    for i in range(2,steps+1):
        tmpY = append(tmpY,array(ctdf[ctdf.index%len_of_trail == last_point_num+i].y))
    
    error = mean(vec_distance(tmpX,tmpY,X,Y))  # 已修改为经纬度
    return error