#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "SimAnneal.h"

double SA(double *x_given, double *y_given, double *z_given, double *r_given, int imax_given)
{
    double pa, prob, sumx = 0, sumy = 0, sumz = 0;
    double cold, cmin = 1e+20, cnew, xnew, ynew, znew;
    double T = 1, a = 0.999;
    int i, iter = 0, O = 5;
    double rangex, rangey, rangez, ranger;
    // FILE *fp, *fp2;
    imax = imax_given;

    for (i = 0; i < imax; i++)
    {
        x[i] = x_given[i];
        y[i] = y_given[i];
        z[i] = z_given[i];
        r[i] = r_given[i];
        sumx += x[i];
        sumy += y[i];
        sumz += z[i];
    }

    x[imax] = sumx / imax;
    y[imax] = sumy / imax;
    z[imax] = sumz / imax;
    cold = CostCal(x[imax], y[imax], z[imax], 1);

    // fp = fopen ("Data.csv","w");
    // if(fp == NULL){
    // printf("Couldn't open file\n");
    // return 1;
    // }

    // fp2 = fopen ("Cost.csv","w");
    // if(fp2 == NULL){
    // printf("Couldn't open file\n");
    // return 0;
    // }
    while (T > 1e-5)
    {
        iter++;
        i = 0;
        while (i < O)
        {
            maxmin(&rangex, &rangey, &rangez, &ranger);
            move(&xnew, &ynew, &znew, rangex, rangey, rangez, ranger, T);
            cnew = CostCal(xnew, ynew, znew, iter);
            if (cnew < cmin)
            {
                cmin = cnew;
                xmin = xnew;
                ymin = ynew;
                zmin = znew;
                cold = cnew;
                x[imax] = xnew;
                y[imax] = ynew;
                z[imax] = znew;
                for (int j = 0; j < imax; j++)
                    rmin[j] = r[j];
                break;
            }
            prob = exp((cold - cnew) / T);
            pa = ((double)rand() / (double)RAND_MAX);

            if (prob > pa)
            {
                x[imax] = xnew;
                y[imax] = ynew;
                z[imax] = znew;
                // Diff(x[imax], y[imax], z[imax]);
                cold = cnew;
                // fprintf(fp, "%f,%f,%f\n", x[imax], y[imax], z[imax]);
                // fprintf(fp2, "%f\n", cold);
            }
            i++;
        }
        T *= a;
    }
    x[imax] = xmin;
    y[imax] = ymin;        
    z[imax] = zmin;
    return cmin;
}

double CostCal(double xp, double yp, double zp, int t)
{
    int i;
    double cost = 0, lisq = 0, l[1024], mc = 0, pc = 0, pcin = 0, penalty = 0;
    double w0 = 642, w1 = 5e3, w2 = 1;

    for (i = 0; i < imax; i++)
    {
        lisq = pow((xp - x[i]), 2) + pow((yp - y[i]), 2) + pow((zp - z[i]), 2);
        l[i] = sqrt(lisq);
        mc += l[i] * pow(r[i], 2);
        penalty += pow(get_max(0, r[i] - 2), 2) + pow(get_max(0, 0.5 - r[i]), 2);
    }
    cost = w0 * mc + w1 * penalty + (w2 * l[0] / pow(r[0], 4));
    

    for (i = 1; i < imax; i++)
    {
        pcin += pow(r[i], 4) / l[i];
    }
    pc = 1.0 / pcin;
    cost += w2 * pc;
    return cost;
}

int move(double *xnew,double *ynew,double *znew, double rangex, double rangey, double rangez, double ranger, double T)
{
    *xnew = x[imax] + (((2 * (double)rand() / (double)RAND_MAX) - 1) * 0.05 * rangex);
    *ynew = y[imax] + (((2 * (double)rand() / (double)RAND_MAX) - 1) * 0.05 * rangey);
    *znew = z[imax] + (((2 * (double)rand() / (double)RAND_MAX) - 1) * 0.05 * rangez);

    double r_sum = 0;
    for (int i = 1; i < imax; i++)
    {
        r[i] += (((2 * (double)rand() / (double)RAND_MAX) - 1) * 0.005 * (T * ranger));
        r_sum += pow(r[i], 3);
    }
    r[0] = pow(r_sum, 1.0 / 3.0);
    return 0;
}


int maxmin(double *rangex, double *rangey, double *rangez, double *ranger)
{
    *rangex = 100;
    *rangey = 100;
    *rangez = 100;
    *ranger = 2;
    return 0;
    double xmax = x[0], ymax = y[0], zmax = z[0], rmax = r[0], xmin = x[0], ymin = y[0], zmin = z[0], rmin = r[0];
    for (int i = 0; i < imax; i++)
    {
        if (x[i] > xmax)
            xmax = x[i];
        if (y[i] > ymax)
            ymax = y[i];
        if (z[i] > zmax)
            zmax = z[i];
        if (r[i] > rmax)
            rmax = r[i];
        if (x[i] < xmin)
            xmin = x[i];
        if (y[i] < ymin)
            ymin = y[i];
        if (z[i] < zmin)
            zmin = z[i];
        if (r[i] < rmin)
            rmin = r[i];
    }
    *rangex = xmax - xmin;
    *rangey = ymax - ymin;
    *rangez = zmax - zmin;
    *ranger = rmax - rmin;
    return 0;
}

// double Diff(double xn, double yn, double zn)
// {
//     int i;
//     double dco[4] = {0, 0, 0, 0};

//     for (i = 1;i <= 3;i++)
//     {
//         dco[i] = 0.0001;
//         d1[i] = (CostCal(xn + dco[1], yn + dco[2], zn + dco[3]) - CostCal(xn - dco[1], yn - dco[2], zn - dco[3])) / 2 * dco[i];
//         d2[i] = (CostCal(xn + dco[1], yn + dco[2], zn + dco[3]) - 2 * CostCal(xn, yn, zn) + CostCal(xn - dco[1], yn - dco[2], zn - dco[3])) / pow(dco[i], 2);
//         dco[i] = 0;
//     }
//     return 0;
// }

double get_coord(char c)
{
    if (c == 'x')
        return x[imax];
    if (c == 'y')
        return y[imax];
    if (c == 'z')
        return z[imax];
    return 0;
}

double* get_r()
{
    return rmin;
}

double get_max(double a, double b)
{
    if (a > b)
        return a;
    else
        return b;
}
