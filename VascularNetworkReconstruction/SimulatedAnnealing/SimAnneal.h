#ifndef SA_H_
#define SA_H_

double SA(double *, double *, double *, double *, int);
double CostCal(double, double, double);
int move(double *, double *, double *, double, double, double, double, double);
int maxmin (double *, double *, double *, double *);
double Diff(double, double, double);
double get_coord(char);
double* get_r(void);
double get_max(double, double);


double x[1024], y[1024], z[1024], r[1024], rmin[1024], d1[6], d2[6], xmin, ymin, zmin;
int imax;

#endif