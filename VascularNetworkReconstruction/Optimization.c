#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double cost();
int move(double *, double *, double *, double, double, double, double);
double SimAnneal(double);
int maxmin (double *, double *, double *);
double x[1000], y[1000], z[1000], xmin, ymin, zmin;
int imax;

int main()
{
	int i;
	double cold, cmin, sumx=0, sumy=0, sumz=0;
	char file_name[100];
	FILE *fp;

printf("Filename: ");
scanf("%[^\n]", file_name);
fp = fopen (file_name,"r");
if(fp == NULL){
    printf("Couldn't open file\n");
    return 1;
}
i=0;
while(!feof(fp))
{
	i++;
	fscanf(fp,"%lf%*c%lf%*c%lf", &x[i], &y[i], &z[i]);
}
imax=i;

    for (i=1; i<imax; i++)
    {
        sumx = sumx + x[i];
        sumy = sumy + y[i];
        sumz = sumz + z[i];
    }
    x[imax] = sumx/imax;
    y[imax] = sumy/imax;
    z[imax] = sumz/imax;
cold=cost();

cmin = SimAnneal(cold);
for (i=1; i<=imax; i++)
{
	printf("%d\t%f %f %f\n", i, x[i],y[i],z[i]);

}
printf("\n\n\nThe optimum location of branching point is -\nx = %f\ty = %f\t z = %f\nCost = %f", x[imax], y[imax], z[imax], cmin);
}

double SimAnneal(double cold)
{
    double pa, prob;
    double cmin = 1e+20, cnew, xnew, ynew, znew;
    double T = 1, a = 0.999;
    int i, iter=0, O = 2;
    double rangex, rangey, rangez;
    maxmin(&rangex, &rangey, &rangez);

while (T>1e-3)
{
    iter++;
    printf("iter = %d\n", iter);
    i=0;
    while (i<O)
    {
        move(&xnew, &ynew, &znew, rangex, rangey, rangez, T);
        cnew = cost();
        if (cnew<cmin){
            cmin = cnew;
            xmin = xnew;
            ymin = ynew;
            zmin = znew;
        }
        prob = exp((cold-cnew)/T);
        pa = ((double)rand() / (double)RAND_MAX) ;

        if (prob>pa){
            x[imax]=xnew;
            y[imax]=ynew;
            z[imax]=znew;
            cold= cnew;
            i++;
            printf("%d\t%f\t%f\n", i, T, cnew);
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

double cost ()
{
    int i;
    double cost=0, lisq, c, li, r0 = 30, po, ri;

    po = r0/(double)(imax-2);
    ri = pow (po, 1/3);
for (i=1; i<imax; i++)
{
    lisq= pow((x[imax]-x[i]),2) + pow((y[imax]-y[i]),2) + pow((z[imax]-z[i]),2);
    li = sqrt (lisq);
    if (i==1)
        c= li*(pow(r0,2)+ pow(r0, -4));
    else
        c= li*(pow(ri,2)+ pow(ri, -4));
    cost += c;
}
    return cost;
}

int move(double *xnew,double *ynew,double *znew, double rangex, double rangey, double rangez, double T)
{
    int i;
    *xnew = x[imax]+ (((2*(double)rand()/(double)RAND_MAX)-1)*0.01*sqrt(T*rangex));
    *ynew = y[imax]+ (((2*(double)rand()/(double)RAND_MAX)-1)*0.01*sqrt(T*rangey));
    *znew = z[imax]+ (((2*(double)rand()/(double)RAND_MAX)-1)*0.01*sqrt(T*rangez));
}


int maxmin(double *rangex, double *rangey, double *rangez)
{
    int i;
    double xmax=0, ymax=0, zmax=0, xmin = 1e+50, ymin=1e+50, zmin= 1e+50;
    for (i=1; i<imax; i++)
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
