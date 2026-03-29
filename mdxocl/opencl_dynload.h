/*
 * opencl_dynload.h — Runtime dynamic loading of OpenCL library
 *
 * Loads OpenCL.dll / libOpenCL.so at runtime via LoadLibrary/dlopen.
 * If the library is not found, GPU is simply unavailable.
 * This allows mdxfind to start and run CPU-only on systems without OpenCL.
 */

#ifndef OPENCL_DYNLOAD_H
#define OPENCL_DYNLOAD_H

#if defined(OPENCL_GPU)

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

/* Load OpenCL library at runtime. Returns 0 on success, -1 if not found. */
int opencl_dynload_init(void);

/* Function pointers — use these instead of calling OpenCL directly */
extern cl_int (*p_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
extern cl_int (*p_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
extern cl_int (*p_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
extern cl_context (*p_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *, void (CL_CALLBACK *)(const char *, const void *, size_t, void *), void *, cl_int *);
extern cl_command_queue (*p_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
extern cl_program (*p_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
extern cl_int (*p_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void (CL_CALLBACK *)(cl_program, void *), void *);
extern cl_int (*p_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
extern cl_kernel (*p_clCreateKernel)(cl_program, const char *, cl_int *);
extern cl_int (*p_clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);
extern cl_mem (*p_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
extern cl_int (*p_clReleaseMemObject)(cl_mem);
extern cl_int (*p_clReleaseKernel)(cl_kernel);
extern cl_int (*p_clReleaseProgram)(cl_program);
extern cl_int (*p_clReleaseCommandQueue)(cl_command_queue);
extern cl_int (*p_clReleaseContext)(cl_context);
extern cl_int (*p_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
extern cl_int (*p_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
extern cl_int (*p_clEnqueueFillBuffer)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
extern cl_int (*p_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
extern cl_int (*p_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
extern cl_int (*p_clFinish)(cl_command_queue);

/* Redirect all clXxx calls to function pointers */
#define clGetPlatformIDs p_clGetPlatformIDs
#define clGetDeviceIDs p_clGetDeviceIDs
#define clGetDeviceInfo p_clGetDeviceInfo
#define clCreateContext p_clCreateContext
#define clCreateCommandQueue p_clCreateCommandQueue
#define clCreateProgramWithSource p_clCreateProgramWithSource
#define clBuildProgram p_clBuildProgram
#define clGetProgramBuildInfo p_clGetProgramBuildInfo
#define clCreateKernel p_clCreateKernel
#define clGetKernelWorkGroupInfo p_clGetKernelWorkGroupInfo
#define clCreateBuffer p_clCreateBuffer
#define clReleaseMemObject p_clReleaseMemObject
#define clReleaseKernel p_clReleaseKernel
#define clReleaseProgram p_clReleaseProgram
#define clReleaseCommandQueue p_clReleaseCommandQueue
#define clReleaseContext p_clReleaseContext
#define clEnqueueWriteBuffer p_clEnqueueWriteBuffer
#define clEnqueueReadBuffer p_clEnqueueReadBuffer
#define clEnqueueFillBuffer p_clEnqueueFillBuffer
#define clSetKernelArg p_clSetKernelArg
#define clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel
#define clFinish p_clFinish

#endif /* OPENCL_GPU */
#endif /* OPENCL_DYNLOAD_H */
