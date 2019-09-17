#include <Python.h>
#include <numpy/arrayobject.h>
#include "SimAnneal.h"

static char SimAnneal_docstring[] =
    "This module provides an interface for optimization using simulated annealing in C.";
static char SA_docstring[] =
    "Optimize a given problem using given input data";

static PyObject *SimAnneal_SA(PyObject *self, PyObject *args);

static PyMethodDef SimAnnealMethods[] = {
    {"SA", SimAnneal_SA, METH_VARARGS, SA_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef SimAnnealModule = {
   PyModuleDef_HEAD_INIT,
   "SimAnneal",
    SimAnneal_docstring,
   -1,
    SimAnnealMethods
};

PyMODINIT_FUNC PyInit_SimAnneal(void)
{
    return PyModule_Create(&SimAnnealModule);
}


static PyObject *SimAnneal_SA(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *y_obj, *z_obj, *r_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOO", &x_obj, &y_obj, &z_obj, &r_obj))
        return NULL;
	
	// printf("\nInput parsed successfully!\n");

    PyArrayObject *x_array = (PyArrayObject*) x_obj;
    PyArrayObject *y_array = (PyArrayObject*) y_obj;
    PyArrayObject *z_array = (PyArrayObject*) z_obj;
    PyArrayObject *r_array = (PyArrayObject*) r_obj;

    /* How many data points are there? */
    int imax = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *x = (double*)PyArray_DATA(x_array);
    double *y = (double*)PyArray_DATA(y_array);
    double *z = (double*)PyArray_DATA(z_array);
    double *r = (double*)PyArray_DATA(r_array);

    /* Call the external C function to compute the cost. */
    double cost = SA(x, y, z, r, imax);
    double x_coord = get_coord('x');
    double y_coord = get_coord('y'); 
    double z_coord = get_coord('z');
    double* r_new = get_r();

    /* Clean up. */
    // Py_DECREF(x_array);
    // Py_DECREF(y_array);
    // Py_DECREF(z_array);
    // Py_DECREF(r_array);

    if (cost < 0.0) 
    {
        PyErr_SetString(PyExc_RuntimeError, "SA returned an impossible value.");
        return NULL;
    }

    PyObject* retList = PyList_New(imax + 4);
    for (int i = 0; i < imax; i++)
    {

        PyList_SetItem(retList, i, Py_BuildValue("d", r_new[i]));
    }
    PyList_SetItem(retList, imax, Py_BuildValue("d", x_coord));
    PyList_SetItem(retList, imax + 1, Py_BuildValue("d", y_coord));
    PyList_SetItem(retList, imax + 2, Py_BuildValue("d", z_coord));
    PyList_SetItem(retList, imax + 3, Py_BuildValue("d", cost));

    return retList;
}
