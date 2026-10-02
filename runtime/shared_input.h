#pragma once
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdexcept>
#include <cstddef>

class SharedInput {
public:
    void* data=nullptr;
    size_t size=0;
    SharedInput(int fd, size_t bytes): size(bytes) {
        struct stat st{};
        if (fstat(fd,&st)!=0 || st.st_size!=static_cast<off_t>(bytes))
            throw std::runtime_error("shared raw-frame size mismatch");
        data=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        if (data==MAP_FAILED) { data=nullptr; throw std::runtime_error("shared raw-frame mmap failed"); }
    }
    ~SharedInput() { if(data) munmap(data,size); }
    SharedInput(const SharedInput&)=delete;
    SharedInput& operator=(const SharedInput&)=delete;
};
