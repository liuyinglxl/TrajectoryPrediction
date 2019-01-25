#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "gps.h"
#include<math.h>
#include <memory.h>

int NUM = 800;
//int step = 29;

#define PI                      3.1415926
#define EARTH_RADIUS            6378.137        //地球近似半径

float radian(float d);
float get_distance(float lat1, float lng1, float lat2, float lng2);

// 求弧度
float radian(float d)
{
    return d * PI / 180.0;   //角度1˚ = π / 180
}

//计算距离
float get_distance(float lat1, float lng1, float lat2, float lng2)
{
    float radLat1 = radian(lat1);
    float radLat2 = radian(lat2);
    float dlat = radLat1 - radLat2;
    float dlon = radian(lng1) - radian(lng2);

    double a=sin(dlat/2)*sin(dlat/2) + cos(radLat1)*cos(radLat2)*sin(dlon/2)*sin(dlon/2);
    double c=2*atan2(sqrt(a),sqrt(1-a)) ;
    double d=EARTH_RADIUS*c;
    double dst=d*1000;
    //float dst = 2 * asin((sqrt(pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * pow(sin(b / 2), 2) )));

    //dst = dst * EARTH_RADIUS;
    //dst= round(dst * 10000) / 10000;
    return dst;
}

struct DataSet {
    int n1;
    int n2;
    double lat;
    double lon;
};

void test_read_lat_long_track(int step)
{
  FILE* file = fopen("/mnt/data2/TITS_data/beijing/beijing_train", "r");
  //FILE* file1 = fopen("out_train.txt", "w");
  //------------------------------------------------------
  int n1,n2;
  double lat,lon;
  double error=0;
  double loss=0;
  int flag=0;
  struct DataSet dataSet[NUM];

  int h;
  //FILE *P[2000];


  KalmanFilter f = alloc_filter_velocity2d(300.0);   //新建一个滤波器，括号里给定噪声参数，可以改变噪声参数
  for(int index = 0; index < 10000; index++)    //set index < 2000，index代表组数
  {
      for(int i = 0;i < NUM; i++)     //读入一组数据，2000个，原来给的数据多了90个。
        {
        read_lat_long(file,&n1, &n2, &lat, &lon);
        struct DataSet dataItem = {n1, n2, lat, lon};
        dataSet[i] = dataItem;
        }
          //------------------------------------------------------

          assert(file);
          int i,j,k,l,m;

        for(int j=0;j<770;j++)                      //每个ID为一组，组内的计算
        {
            double result_lat = 0;
            double result_lon = 0;

            //char out[35];
            //sprintf(out, "kalman_truck_%d.txt", j+10);
            //printf("%s\n",out);
            //if(index==0) P[j] = fopen(out, "w");
            //else P[j] = fopen(out, "a");

            for(k=j;k<=j+29;k++)                           //前10组数据已知，只更新滤波器
            {
              update_velocity2d(f, dataSet[k].lat, dataSet[k].lon, 5);//括号中最后一个参数可以调整
            }

            get_lat_long(f, &result_lat, &result_lon);
            //int m=0;
            //fprintf(P[j],"%d %d %f %f\n",index,m,result_lat,result_lon);//结果写入文件，文件名与j一致
            //m+=1;
           // if (j>99&&j<200)
           //     {
                error+=sqrt(pow((result_lat-dataSet[j+30].lat),2)+pow((result_lon-dataSet[j+30].lon),2)); //计算误差，欧氏距离
                loss+=get_distance(dataSet[k+1].lat,dataSet[k+1].lon,result_lat,result_lon);//计算meter误差
                flag+=1;
            //    }


              for(l=j+30;l<=j+step;l++)                      //预测未来20步的结果
              {
                  update_velocity2d(f, result_lat, result_lon, 5);
                  get_lat_long(f, &result_lat, &result_lon);
                  if(l<j+step)
                  {
                  //fprintf(P[j], "%d %d %f %f\n",index,m,result_lat,result_lon);//结果写入文件
                  //m+=1;
               //   if (j>99&&j<200)
               // {
                    error+=sqrt(pow((result_lat-dataSet[j+30].lat),2)+pow((result_lon-dataSet[j+30].lon),2)); //计算误差，欧氏距离
                    loss+=get_distance(dataSet[k+1].lat,dataSet[k+1].lon,result_lat,result_lon);//计算meter误差
                    flag+=1;
                //    }
                }
              }
              printf("%d %d\n", index,j);
              //fclose(P[j]);
            }

  }
  fclose(file);
  error=error/flag;
  loss=loss/flag;
printf("step= %d\n",step);
printf("error= %g\n",error);
printf("loss= %f \n",loss);
}

int main(void) {
//test_read_lat_long_track(19);
test_read_lat_long_track(39);
//test_read_lat_long_track(39);
return 0;
}
