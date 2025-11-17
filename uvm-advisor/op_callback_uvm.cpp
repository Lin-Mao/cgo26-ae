#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <sanitizer.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>
#include <tuple>
#include <vector>
#include <map>
#include <regex>
#include <unordered_map>
#include <cassert>

#define CUDA_SAFECALL(call)                                      \
{                                                                \
    call;                                                        \
    cudaError err = cudaGetLastError();                          \
    if (cudaSuccess != err) {                                    \
        fprintf(                                                 \
            stderr,                                              \
            "[CUDA ERROR] '%s' in '%s:%i' - error: %s.\n",       \
            #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        fflush(stderr);                                          \
        assert(false);                                           \
    }                                                            \
}


#define SANITIZER_UVM_MEMORY_FLAG 0x6
#define LARGE_TENSOR_THRESHOLD 1048576


typedef struct {
    uint64_t op_id = 0;
    uint64_t kernel_id = 0;
    uint64_t mem_id = 0;
    uint64_t ten_id = 0;
} op_key_t;

static op_key_t op_key;
static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> id_2_tensor_map;
static std::unordered_map<uint64_t, uint64_t> ptr_to_ten_id;
static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> id_2_memory_map;


typedef enum {
    NO_PREFETCH = 0,
    OBJECT_GRANULARITY = 1,
    TENSOR_GRANULARITY = 2,
} PrefetchMode_t;
// ENV: PREFETCH_MODE
static PrefetchMode_t prefetch_mode = TENSOR_GRANULARITY;


static std::vector<cudaStream_t> prefetch_streams;
static int num_prefetch_streams = 3;
static uint64_t stream_index = 0;


using MemTenEntry = std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;
std::unordered_map<int, MemTenEntry> prefetch_schedule;

static void parse_profile_output(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line;

    int current_op_id = -1;

    while (std::getline(infile, line)) {
        // Parse op line
        if (line.rfind("Op -", 0) == 0) {
            size_t pos = line.find("op_id:");
            if (pos != std::string::npos) {
                current_op_id = std::stoi(line.substr(pos + 7));
            }
        }

        // Parse MemAlloc
        else if (line.find("MemAlloc") != std::string::npos) {
            std::vector<uint64_t> mem_allocs;
            std::regex entry_regex(R"((\d+):)");
            auto begin = std::sregex_iterator(line.begin(), line.end(), entry_regex);
            auto end = std::sregex_iterator();
            for (auto i = begin; i != end; ++i) {
                std::smatch match = *i;
                uint64_t id = (uint64_t) std::stoi(match[1]);
                mem_allocs.emplace_back(id);
            }
            prefetch_schedule[current_op_id].first = mem_allocs;
        }

        // Parse TenAlloc
        else if (line.find("TenAlloc") != std::string::npos) {
            std::vector<uint64_t> ten_allocs;
            std::regex entry_regex(R"((\d+):)");
            auto begin = std::sregex_iterator(line.begin(), line.end(), entry_regex);
            auto end = std::sregex_iterator();
            for (auto i = begin; i != end; ++i) {
                std::smatch match = *i;
                uint64_t id = (uint64_t) std::stoi(match[1]);
                ten_allocs.emplace_back(id);
            }
            prefetch_schedule[current_op_id].second = ten_allocs;
        }
    }
}


static void no_prefetch(uint64_t op_id) {
    // do nothing
}


static void prefetch_at_tensor_granularity(uint64_t op_id) {
    if (prefetch_schedule.find(op_id) == prefetch_schedule.end()) {
        return;
    }

    auto ten_allocs = prefetch_schedule[op_id].second;

    for (auto ten_alloc : ten_allocs) {
        auto tensor = id_2_tensor_map[ten_alloc];
        void* ptr = (void*) tensor.first;
        size_t size = tensor.second;
        // printf("[prefetch] op_id: %lu, ten_id: %lu, ptr: %lu (%p), size: %zu\n", op_id, ten_alloc, (uint64_t)ptr, ptr, size);
        CUDA_SAFECALL(cudaMemPrefetchAsync(ptr, size, 0, prefetch_streams[stream_index]));
        stream_index = (stream_index + 1) % num_prefetch_streams;
    }
}


static void prefetch_at_object_granularity(uint64_t op_id) {
    if (prefetch_schedule.find(op_id) == prefetch_schedule.end()) {
        return;
    }
    auto mem_allocs = prefetch_schedule[op_id].first;

    for (auto mem_alloc : mem_allocs) {
        auto memory = id_2_memory_map[mem_alloc];
        void* ptr = (void*) memory.first;
        size_t size = memory.second;
        // printf("[prefetch] ptr: %lu (%p), size: %zu\n", (uint64_t)ptr, ptr, size);
        CUDA_SAFECALL(cudaMemPrefetchAsync(ptr, size, 0, prefetch_streams[stream_index]));
        stream_index = (stream_index + 1) % num_prefetch_streams;
    }
}


static void operator_start(const at::RecordFunction& fn, at::ObserverContext* ctx) {
    op_key.op_id++;
    if (prefetch_mode == TENSOR_GRANULARITY) {
        prefetch_at_tensor_granularity(op_key.op_id);
    } else if (prefetch_mode == OBJECT_GRANULARITY) {
        prefetch_at_object_granularity(op_key.op_id);
    } else if (prefetch_mode == NO_PREFETCH) {
        no_prefetch(op_key.op_id);
    }
}


static void operator_end(const at::RecordFunction& fn, at::ObserverContext* ctx) {
}


static void tensor_malloc(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
    // printf("[malloc] ptr: %lu (%p), alloc_size: %ld, device: %d\n", (uint64_t)ptr, ptr, alloc_size, device.index());
    if (alloc_size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }
    op_key.ten_id++;
    id_2_tensor_map[op_key.ten_id] = std::make_pair((uint64_t)ptr, alloc_size);
    ptr_to_ten_id[(uint64_t)ptr] = op_key.ten_id;
}

static void tensor_free(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
    // printf("[free] ptr: %lu (%p), alloc_size: %ld, device: %d\n", (uint64_t)ptr, ptr, alloc_size, device.index());
    if ((-alloc_size) <= LARGE_TENSOR_THRESHOLD) {
        return;
    }
    auto ten_id = ptr_to_ten_id[(uint64_t)ptr];
    id_2_tensor_map.erase(ten_id);
    ptr_to_ten_id.erase((uint64_t)ptr);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct OperatorCallbackContext : at::ObserverContext {};

class OperatorCallback {
public:
    static OperatorCallback& getInstance() {
        static OperatorCallback instance;
        return instance;
    }

private:
    OperatorCallback() {
        auto callback = at::RecordFunctionCallback(
            &OperatorCallback::startCallbackStatic,
            &OperatorCallback::endCallbackStatic
        ).scopes({at::RecordScope::FUNCTION});

        handle_ = at::addGlobalCallback(callback);
    }

    static std::unique_ptr<at::ObserverContext> startCallbackStatic(const at::RecordFunction& fn) {
        return getInstance().onStart(fn);
    }

    static void endCallbackStatic(const at::RecordFunction& fn, at::ObserverContext* ctx) {
        getInstance().onEnd(fn, ctx);
    }

    std::unique_ptr<at::ObserverContext> onStart(const at::RecordFunction& fn) {
        auto ctx = std::make_unique<OperatorCallbackContext>();
        operator_start(fn, ctx.get());
        return ctx;
    }

    void onEnd(const at::RecordFunction& fn, at::ObserverContext* ctx) {
        operator_end(fn, ctx);
    }

    at::CallbackHandle handle_;
};


class TensorCallback{
public:
    static TensorCallback& getInstance() {
        static TensorCallback instance;
        if (!instance.is_enabled) {
            instance.is_enabled = true;
            auto profiler = instance.new_memory_reporting_info();
            c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, profiler);
        }
        return instance;
    }

private:
    TensorCallback() = default;
    ~TensorCallback() = default;

    class MemoryReportingInfo : public c10::MemoryReportingInfoBase {
    public:
        MemoryReportingInfo() = default;
        bool memoryProfilingEnabled() const override {
            return true;
        }

        #if TORCH_VERSION_MAJOR >= 2
        void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                            size_t total_reserved, c10::Device device) override {
            if (device.is_cuda()) {
                if (alloc_size > 0) {
                    TensorCallback::getInstance().onMalloc(ptr, alloc_size, total_allocated, total_reserved, device);
                } else {
                    TensorCallback::getInstance().onFree(ptr, alloc_size, total_allocated, total_reserved, device);
                }
            }
        }
        #else
        void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                            int64_t total_reserved, c10::Device device) override {
            if (device.is_cuda()) {
                if (alloc_size > 0) {
                    TensorCallback::getInstance().onMalloc(ptr, alloc_size, total_allocated, total_reserved, device);
                } else {
                    TensorCallback::getInstance().onFree(ptr, alloc_size, total_allocated, total_reserved, device);
                }
            }
        }
        #endif
    };

    bool is_enabled = false;

    std::shared_ptr<MemoryReportingInfo> new_memory_reporting_info() {
        return std::make_shared<MemoryReportingInfo>();
    }

    void onMalloc(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
        tensor_malloc(ptr, alloc_size, total_allocated, total_reserved, device);
    }

    void onFree(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
        tensor_free(ptr, alloc_size, total_allocated, total_reserved, device);
    }
};

static int init_prefetch_streams() {
    for (int i = 0; i < num_prefetch_streams; i++) {
        cudaStream_t prefetch_stream = nullptr;
        CUDA_SAFECALL(cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking));
        prefetch_streams.push_back(prefetch_stream);
    }

    parse_profile_output("uvm_advisor_opt.log");
    return 0;
}


void cs_callback(
    void* userdata,
    Sanitizer_CallbackDomain domain,
    Sanitizer_CallbackId cbid,
    const void* cbdata)
{
    if (domain == SANITIZER_CB_DOMAIN_RESOURCE) {
        auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
        if (cbid == SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC) {
            if (pModuleData->flags != SANITIZER_UVM_MEMORY_FLAG) {
                return;
            }
            op_key.mem_id++;
            id_2_memory_map[op_key.mem_id] = std::make_pair((uint64_t)pModuleData->address, pModuleData->size);
        }
    }
}

static int compute_sanitizer() {
    Sanitizer_SubscriberHandle handle;
    sanitizerSubscribe(&handle, cs_callback, nullptr);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RESOURCE);
    return 0;
}


static volatile int debug_flag = 0;
static int init_all_instances() {
    // check if DEBUG_FLAG env is set
    const char* debug_flag_str = std::getenv("DEBUG_FLAG");
    if (debug_flag_str) {
        debug_flag = std::stoi(debug_flag_str);
    }
    while (debug_flag) {}

    // get prefetch mode from environment variable
    const char* prefetch_mode_str = std::getenv("PREFETCH_MODE");
    if (prefetch_mode_str) {
        prefetch_mode = static_cast<PrefetchMode_t>(std::stoi(prefetch_mode_str));
    }
    if (prefetch_mode == NO_PREFETCH) {
        printf("PREFETCH_MODE: NO PREFETCH\n");
    } else if (prefetch_mode == OBJECT_GRANULARITY) {
        printf("PREFETCH_MODE: OBJECT GRANULARITY\n");
    } else if (prefetch_mode == TENSOR_GRANULARITY) {
        printf("PREFETCH_MODE: TENSOR GRANULARITY\n");
    }
    fflush(stdout);

    static auto& operator_instance = OperatorCallback::getInstance();
    static auto& tensor_instance = TensorCallback::getInstance();
    static auto init_sanitizer = compute_sanitizer();
    static auto init_streams = init_prefetch_streams();
    return 0;
}

static int global_init_instance = init_all_instances();
