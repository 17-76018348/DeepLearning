#include <stdio.h>
#include <stdlib.h>

final int DATA_NUM = 5



double *tmp();

struct Plus{
	double x;
	double y;

}
struct Minus{
	double x;
	double y;
}
struct Mul{
	double x;
	double y;
}
struct Square{
	double x;
}
struct Cost{
	double x;
}

int main(void)
{
	int data_x[5] = {1,2,3,4,5};
	int data_y[5] = {1,2,3,4,5};
	
}

double forward(int *x, int *y, theta)
{
	double z1 = mul_for()
}

double plus_for(double *x, double *y)
{
	int idx;
	double *ls = malloc(sizeof(double) * DATA_NUM);
	for(idx = 0; i<DATA_NUM; i++)
	{
		ls[idx] = x[idx] + y[idx];
	}
	return 
}
double* plus_back(double dL)
{
	double *ls = malloc(sizeof(double) * 2);
	ls[0] = dL;
	ls[1] = dL;
	return ls;	
}

double minus_for(double x, double y)
{
	return x - y;
}
double* minus_back(double dL)
{
	double *ls = malloc(sizeof(double) * 2);
	ls[0] = dL;
	ls[1] = -1 * dL;
	return ls;
}

double mul_for(double x, double y)
{
	return x * y;
}





double *tmp()
{
	double *x1 = malloc(sizeof(double) * 2);
	x1[0] = 1;
	x1[1] = 2;
	return x1;
}

