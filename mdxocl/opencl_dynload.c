/*
 * opencl_dynload.c — Runtime dynamic loading of OpenCL library
 *
 * Loads OpenCL.dll (Windows) or libOpenCL.so (Linux/FreeBSD) at runtime.
 * If the library is not present, opencl_dynload_init() returns -1 and
 * mdxfind runs CPU-only without error.
 */

#if defined(OPENCL_GPU)

#define CL_TARGET_OPENCL_VERSION 120

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdio.h>
#include <CL/cl.h>

/* Function pointer definitions */
cl_int (*p_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
cl_int (*p_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
cl_int (*p_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
cl_context (*p_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *, void (CL_CALLBACK *)(const char *, const void *, size_t, void *), void *, cl_int *);
cl_command_queue (*p_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
cl_program (*p_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
cl_int (*p_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void (CL_CALLBACK *)(cl_program, void *), void *);
cl_int (*p_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
cl_kernel (*p_clCreateKernel)(cl_program, const char *, cl_int *);
cl_int (*p_clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);
cl_mem (*p_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
cl_int (*p_clReleaseMemObject)(cl_mem);
cl_int (*p_clReleaseKernel)(cl_kernel);
cl_int (*p_clReleaseProgram)(cl_program);
cl_int (*p_clReleaseCommandQueue)(cl_command_queue);
cl_int (*p_clReleaseContext)(cl_context);
cl_int (*p_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
cl_int (*p_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
cl_int (*p_clEnqueueFillBuffer)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
cl_int (*p_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
cl_int (*p_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
cl_int (*p_clFinish)(cl_command_queue);

static int dynload_done = 0;

#ifdef _WIN32
#define DLSYM(h, name) (void *)GetProcAddress(h, name)
#else
#define DLSYM(h, name) dlsym(h, name)
#endif

int opencl_dynload_init(void) {
    if (dynload_done) return (p_clGetPlatformIDs != NULL) ? 0 : -1;
    dynload_done = 1;

#ifdef _WIN32
    HMODULE lib = LoadLibraryA("OpenCL.dll");
#elif defined(__APPLE__)
    void *lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY);
#else
    void *lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libOpenCL.so", RTLD_LAZY);
#endif

    if (!lib) return -1;

    p_clGetPlatformIDs = DLSYM(lib, "clGetPlatformIDs");
    p_clGetDeviceIDs = DLSYM(lib, "clGetDeviceIDs");
    p_clGetDeviceInfo = DLSYM(lib, "clGetDeviceInfo");
    p_clCreateContext = DLSYM(lib, "clCreateContext");
    p_clCreateCommandQueue = DLSYM(lib, "clCreateCommandQueue");
    p_clCreateProgramWithSource = DLSYM(lib, "clCreateProgramWithSource");
    p_clBuildProgram = DLSYM(lib, "clBuildProgram");
    p_clGetProgramBuildInfo = DLSYM(lib, "clGetProgramBuildInfo");
    p_clCreateKernel = DLSYM(lib, "clCreateKernel");
    p_clGetKernelWorkGroupInfo = DLSYM(lib, "clGetKernelWorkGroupInfo");
    p_clCreateBuffer = DLSYM(lib, "clCreateBuffer");
    p_clReleaseMemObject = DLSYM(lib, "clReleaseMemObject");
    p_clReleaseKernel = DLSYM(lib, "clReleaseKernel");
    p_clReleaseProgram = DLSYM(lib, "clReleaseProgram");
    p_clReleaseCommandQueue = DLSYM(lib, "clReleaseCommandQueue");
    p_clReleaseContext = DLSYM(lib, "clReleaseContext");
    p_clEnqueueWriteBuffer = DLSYM(lib, "clEnqueueWriteBuffer");
    p_clEnqueueReadBuffer = DLSYM(lib, "clEnqueueReadBuffer");
    p_clEnqueueFillBuffer = DLSYM(lib, "clEnqueueFillBuffer");
    p_clSetKernelArg = DLSYM(lib, "clSetKernelArg");
    p_clEnqueueNDRangeKernel = DLSYM(lib, "clEnqueueNDRangeKernel");
    p_clFinish = DLSYM(lib, "clFinish");

    /* Verify critical functions resolved */
    if (!p_clGetPlatformIDs || !p_clGetDeviceIDs || !p_clCreateContext ||
        !p_clCreateCommandQueue || !p_clEnqueueNDRangeKernel || !p_clFinish) {
        p_clGetPlatformIDs = NULL; /* mark as failed */
        return -1;
    }

    return 0;
}

#endif /* OPENCL_GPU */
