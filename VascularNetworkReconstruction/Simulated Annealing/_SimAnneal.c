#include <Python.h>
#include <numpy/arrayobject.h>
#include "SimAnneal.h"

static char module_docstring[] =
    "This module provides an interface for optimization using simulated annealing in C.";
static char SimAnneal_docstring[] =
    "Optimize a given problem using given input data";

static PyObject *SimAnneal_SA(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"SA", SimAnneal_SA, METH_VARARGS, SimAnneal_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_SimAnneal(void)
{
    PyObject *m = Py_InitModule3("_SimAnneal", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


static PyObject *SimAnneal_SA(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *y_obj, *z_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOO", &x_obj, &y_obj, &z_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *z_array = PyArray_FROM_OTF(z_obj, NPY_DOUBLE,NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL || z_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(z_array);
        return NULL;
    }

    /* How many data points are there? */
    int imax = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *x = (double*)PyArray_DATA(x_array);
    double *y = (double*)PyArray_DATA(y_array);
    double *z = (double*)PyArray_DATA(z_array);

    /* Call the external C function to compute the chi-squared. */
    double cost = SA(x, y, z, imax);

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(z_array);

    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "SA returned an impossible value.");
        return NULL;
    }

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", cost);
    return ret;
}