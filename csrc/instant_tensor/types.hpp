#pragma once

#include <atomic>
#include <any>
#include <cstdint>
#include <functional>
#include <instant_tensor/common.hpp>
#include <instant_tensor/dl_binding/cuda_binding.hpp>
#include <instant_tensor/dl_binding/nccl_binding.hpp>
#include <instant_tensor/executor/function_executor.hpp>

namespace instanttensor {

using namespace cuda_binding;
using namespace nccl_binding;

using chunk_id_t = ssize_t;

using SingleThreadTaskExecutor = SingleWorkerFunctionExecutor<MAX_IO_DEPTH, MAX_IO_DEPTH>;
using ThreadPoolTaskExecutor = MultiWorkerFunctionExecutor<MAX_IO_DEPTH, MAX_IO_DEPTH>;

struct IOCompletion {
    int request_id;
    std::exception_ptr error;
    bool retry = false;
};

enum class IOSubmitStatus {
    // Dispatched to the kernel/worker; completion will arrive through polling.
    Submitted,
    Retry,
    // The IO stage is complete; CUDA/NCCL still follows the normal pipeline.
    Completed,
};

using IOExecutorBase = SingleWorkerDriverExecutor<std::function<IOSubmitStatus()>, std::any,
                                                MAX_IO_DEPTH, MAX_IO_DEPTH>;

class IOExecutor;
class CUDAExecutor;

// NOTE: edit 
enum Backend {
    AIO,
    AIO_BUFFERED,
    URING,
    URING_BUFFERED,
    CUFILE,
    MMAP,
    FGDS,
};

struct BackendStatus {
    bool available;
    string reason;
    string warning;
};

inline bool is_valid_backend(Backend backend) {
    return backend >= Backend::AIO && backend <= Backend::FGDS;
}

inline string backend_to_string(Backend backend) {
    switch(backend) {
        case Backend::AIO: return "AIO";
        case Backend::AIO_BUFFERED: return "AIO_BUFFERED";
        case Backend::URING: return "URING";
        case Backend::URING_BUFFERED: return "URING_BUFFERED";
        case Backend::CUFILE: return "CUFILE";
        case Backend::MMAP: return "MMAP";
        case Backend::FGDS: return "FGDS";
    }
    return "UNKNOWN";
}

enum Op {
    OPEN,
    CLOSE,
    GET_TENSOR_PTR,
    STREAM_TENSOR,
};

struct OpenArgs {
    vector<string> filenames;
    int device_idx;
    ncclComm_t group_communicator;
    int rank;
    int world_size;
    size_t buffer_size;
    size_t chunk_size;
    size_t concurrency;
    size_t io_depth;
    Backend backend;
    vector<pair<size_t, size_t>> tensor_offsets;
    vector<bool> tensor_keep;
    OpenArgs(const vector<string> &filenames, int device_idx, ncclComm_t group_communicator, int rank,
        int world_size, size_t buffer_size, size_t chunk_size, size_t concurrency, size_t io_depth, Backend backend, const vector<pair<size_t, size_t>>& tensor_offsets,
        const vector<bool>& tensor_keep)
        : filenames(filenames), device_idx(device_idx), group_communicator(group_communicator), rank(rank), world_size(world_size),
        buffer_size(buffer_size), chunk_size(chunk_size), concurrency(concurrency), io_depth(io_depth), backend(backend), tensor_offsets(tensor_offsets), tensor_keep(tensor_keep)
        {}
};

struct CloseArgs {
    CloseArgs() {}
};

struct GetTensorArgs {
    size_t tensor_index;
    GetTensorArgs(size_t tensor_index) : tensor_index(tensor_index) {}
};

// Stream a contiguous slice of an oversize tensor (larger than the staging
// buffer) from the file into a caller-provided destination in bounded host
// windows, reusing the asynchronous I/O pipeline. The destination `dst` is a
// HOST memory block (e.g. a shared-memory region) covering the WHOLE tensor;
// this rank writes only its own disjoint [slice_offset, slice_offset+slice_size)
// range. Across all ranks the shared block then holds the full tensor, so a
// tensor that cannot fit in per-rank GPU memory can still be materialized once
// in shared RAM and sliced by consumers (e.g. vLLM) for GPU placement.
//
// Only host-staging backends (aio/uring/mmap) are supported here, because the
// bytes must be moved into host `dst`.
struct StreamTensorArgs {
    size_t file_index;
    size_t file_offset;  // absolute byte offset of the tensor base in the file
    size_t slice_offset; // byte offset *within the tensor* where THIS rank's slice starts
    size_t slice_size;   // number of tensor bytes written by THIS rank
    void *dst;           // base host pointer of the destination (full tensor)
    size_t dst_capacity; // writable bytes at dst (>= file_offset-relative size)
    StreamTensorArgs(size_t file_index, size_t file_offset, size_t slice_offset,
                     size_t slice_size, void *dst, size_t dst_capacity)
        : file_index(file_index), file_offset(file_offset), slice_offset(slice_offset),
          slice_size(slice_size), dst(dst), dst_capacity(dst_capacity) {}
};

struct FreeTensorArgs {
    size_t tensor_index;
    FreeTensorArgs(size_t tensor_index) : tensor_index(tensor_index) {}
};

struct RPCRequest {
    int id;
    int op;
    std::any args;
};

struct RPCResponse {
    int id;
    std::any result;
};

struct TensorMetadate {
    size_t size;
    size_t file_index;
    size_t file_offset;
    size_t device_buffer_offset;
    // Empty tensors anchor both IDs to the previous emitted chunk, or -1 if none exists.
    chunk_id_t first_chunk_id;
    chunk_id_t last_chunk_id;
    // Furthest prefetchable chunk without overwriting the tensor's first chunk.
    chunk_id_t prefetch_chunk_id;
};

struct ChunkIOState {
    int io_request_id;
    // Rank-local read range and staging window, fixed before submission and across retries.
    size_t rank_size;
    size_t rank_file_offset;
    size_t window_offset;
    size_t bytes_completed;
    // Loader-owned handle for completion of IO, CUDA and optional NCCL work.
    CUDAExecutor* cuda_executor;
    int cuda_request_id;
    // Streaming reads (oversize tensors) target a caller-provided destination
    // instead of the shared ring buffer. When stream_dst is non-null,
    // post_read_chunk copies the tensor bytes to that destination.
    void *stream_dst = nullptr;
    size_t stream_dst_offset = 0; // byte offset into stream_dst for this window
    size_t stream_read_offset = 0; // offset within the host staging window to the tensor bytes
    size_t stream_size = 0;        // number of tensor bytes delivered to stream_dst
};

struct IORequest {
    IOExecutor* executor;
    int wait_handle;
    bool loaded_to_device;
};

struct Chunk {
    size_t size; // size of the chunk in bytes
    size_t file_index;
    size_t file_offset;
    size_t device_buffer_offset;
    ChunkIOState io_state;
};

struct HostBufferCacheEntry {
    void *ptr;
    size_t size;
    std::function<void(void*)> deleter;
};

struct FileInfo {
    string filename;
    int fd;
    off_t size;
};

// Transient submission parameters; persistent IO parameters live in chunk.io_state.
struct ChunkIOParams {
    chunk_id_t chunk_id;
    const Chunk &chunk;
    const FileInfo &file;
    size_t window_idx;
    void *rank_dst;
    void *all_dst;
    cudaEvent_t event;
};

} // namespace instanttensor
