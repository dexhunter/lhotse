#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/* Count PyObject allocations while calling a zero-argument callable.
 * The hook forwards to the previous object allocator, so it is legal to
 * install after Py_Initialize. The callback itself does not allocate.
 */

static uint64_t alloc_count;
static int installed;
static PyMemAllocatorEx orig_obj;

static void *counting_malloc(void *ctx, size_t size) {
    alloc_count++;
    return orig_obj.malloc(orig_obj.ctx, size);
}

static void *counting_calloc(void *ctx, size_t nelem, size_t elsize) {
    alloc_count++;
    return orig_obj.calloc(orig_obj.ctx, nelem, elsize);
}

static void *counting_realloc(void *ctx, void *ptr, size_t new_size) {
    alloc_count++;
    return orig_obj.realloc(orig_obj.ctx, ptr, new_size);
}

static void counting_free(void *ctx, void *ptr) {
    orig_obj.free(orig_obj.ctx, ptr);
}

static void install(void) {
    if (installed) {
        return;
    }
    PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &orig_obj);
    PyMemAllocatorEx hooked = orig_obj;
    hooked.malloc = counting_malloc;
    hooked.calloc = counting_calloc;
    hooked.realloc = counting_realloc;
    hooked.free = counting_free;
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &hooked);
    installed = 1;
}

static void uninstall(void) {
    if (!installed) {
        return;
    }
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &orig_obj);
    installed = 0;
}

static PyObject *count_call(PyObject *self, PyObject *args) {
    PyObject *fn;
    if (!PyArg_ParseTuple(args, "O", &fn)) {
        return NULL;
    }
    install();
    alloc_count = 0;
    PyObject *result = PyObject_CallNoArgs(fn);
    uint64_t n = alloc_count;
    uninstall();
    if (result == NULL) {
        return NULL;
    }
    PyObject *packed = PyTuple_New(2);
    if (packed == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(packed, 0, result);
    PyObject *count = PyLong_FromUnsignedLongLong(n);
    if (count == NULL) {
        Py_DECREF(packed);
        return NULL;
    }
    PyTuple_SET_ITEM(packed, 1, count);
    return packed;
}

static PyMethodDef methods[] = {
    {"count_call", count_call, METH_VARARGS,
     "Call fn() and return (result, object_allocation_count)."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "alloccount", NULL, -1, methods};

PyMODINIT_FUNC PyInit_alloccount(void) { return PyModule_Create(&moduledef); }
