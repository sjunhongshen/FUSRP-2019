#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "SimAnneal.h"

double SA(double *x, double *y, double *z, int imax)
{
    double pa, prob, sumx=0, sumy=0, sumz=0;
    double cold, cmin = 1e+20, cnew, xnew, ynew, znew;
    double T = 1, a = 0.999;
    int i, iter=0, O = 2;
    double rangex, rangey, rangez;
    FILE *fp, *fp2;


    for (i=0; i<imax; i++)
    {
        sumx = sumx + x[i];
        sumy = sumy + y[i];
        sumz = sumz + z[i];
    }
    x[imax] = sumx/imax;
    y[imax] = sumy/imax;
    z[imax] = sumz/imax;
    cold=CostCal(x[imax], y[imax], z[imax]);

    maxmin(&rangex, &rangey, &rangez);

    fp = fopen ("Data.csv","w");
    if(fp == NULL){
    printf("Couldn't open file\n");
    return 1;
    }

    fp2 = fopen ("Cost.csv","w");
    if(fp2 == NULL){
    printf("Couldn't open file\n");
    return 0;
    }
while (T>1e-2)
{
    iter++;
    printf("iter = %d\n", iter);
    i=0;
    while (i<O)
    {
        move(&xnew, &ynew, &znew, rangex, rangey, rangez, T);
        cnew = CostCal(xnew,ynew,znew);

        if (cnew<cmin){
            cmin = cnew;
            xmin = xnew;
            ymin = ynew;
            zmin = znew;
        }
        prob = exp((cold-cnew)/T);
        pa = ((double)rand() / (double)RAND_MAX);

        if (prob>pa){
            x[imax]=xnew;
            y[imax]=ynew;
            z[imax]=znew;
            Diff(x[imax], y[imax], z[imax]);
            cold= cnew;
            i++;
            printf("%d\t%f\t%f\n", i, T, cnew);
            fprintf(fp, "%f,%f,%f\n", x[imax], y[imax], z[imax]);
            fprintf(fp2, "%f\n", cold);
        }
    }
    T = T*a;
}
if (cmin<cnew){
    cnew = cmin;
    x[imax] = xmin;
    y[imax] = ymin;
    z[imax] = zmin;
}
return cnew;
}

double CostCal (double xp, double yp, double zp)
{
    int i;
    double cost=0, lisq, c, l[100], r[100], po, mc= 0, pc = 0, pcin;

    r[0] = 30;
    po = r[0]/(double)(imax-1);

for (i=0; i<imax; i++)
{
    if (i != 0){
    r[i] = pow (po, 1.0/3.0);
    }
    lisq= pow((xp-x[i]),2) + pow((yp-y[i]),2) + pow((zp-z[i]),2);
    l[i] = sqrt (lisq);
    mc += l[i]*pow(r[i], 2);
}
    cost = mc + (1e+3*l[0]*pow(r[0], -4));

for (i=1; i<imax; i++)
{
    pcin = pow(r[i], 4)/l[i];
}
    pc = 1.0/ pcin;
    cost += (1e+3*pc);
    return cost;
}

int move(double *xnew,double *ynew,double *znew, double rangex, double rangey, double rangez, double T)
{
    *xnew = x[imax]+ (((2*(double)rand()/(double)RAND_MAX)-1)*0.05*(T*rangex));
// xnew - new x coordinate, x[imax] - previous x coordinate, ((2*(double)rand()/(double)RAND_MAX)-1) - generates random no. between 0 and 1
// T-temperature, rangex - entire range of points along x
    *ynew = y[imax]+ (((2*(double)rand()/(double)RAND_MAX)-1)*0.05*(T*rangey));
    *znew = z[imax]+ (((2*(double)rand()/(double)RAND_MAX)-1)*0.05*(T*rangez));
}


int maxmin(double *rangex, double *rangey, double *rangez)
{
    int i;
    double xmax=0, ymax=0, zmax=0, xmin = x[0], ymin= y[0], zmin= z[0];
    for (i=0; i<imax; i++)
    {
        if (x[i]>xmax)
            xmax = x[i];
        if (y[i]>ymax)
            ymax = y[i];
        if (z[i]>zmax)
            zmax = z[i];
        if (x[i]<xmin)
            xmin = x[i];
        if (y[i]<ymin)
            ymin = y[i];
        if (z[i]<zmin)
            zmin = z[i];
    }
    *rangex= xmax - xmin;
    *rangey= ymax - ymin;
    *rangez= zmax - zmin;
}

double Diff(double xn, double yn, double zn)
{
    int i;
    double dco[4]={0,0,0,0};

    for (i=1;i<=3;i++)
    {
        dco[i]=0.0001;
        d1[i] = (CostCal(xn+dco[1], yn+dco[2], zn+dco[3])-CostCal(xn-dco[1], yn-dco[2], zn-dco[3]))/2*dco[i];
        d2[i] = (CostCal(xn+dco[1], yn+dco[2], zn+dco[3])-2*CostCal(xn, yn, zn)+CostCal(xn-dco[1], yn-dco[2], zn-dco[3]))/pow(dco[i],2);
        dco[i]= 0;
    }
}
